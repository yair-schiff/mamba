"""Bidirectional Mamba module.

"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class BidirectionalMamba(Mamba):
    """Bidirectional Mamba module.

    Weights for in_proj, out_proj, and conv1d are shared between forward and backward passes.
    The SSM parameters, i.e., A and projections for B, C, and dt are not shared between forward and backward passes.
    SSM is run on forward and backward passes separately, and the outputs are combined according to
    `bidirectional_strategy`.
    """
    def __init__(
        self,
        d_model,
        bidirectional_strategy="add",
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        # NOTE: fast_path (i.e., using mamba_ssm.ops.selective_scan_interface.mamba_inner_fn) is not yet supported
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Mamba, self).__init__()  # nn.Module.__init__()
        if bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional_strategy = bidirectional_strategy
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        if use_fast_path:
            print("WARNING: fast_path (i.e., mamba_ssm.ops.selective_scan_interface.mamba_inner_fn) not yet supported!")
            use_fast_path = False
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_rev = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_rev = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_rev = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_rev.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_rev.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_rev.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        self.dt_proj_rev.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.A_log_rev = nn.Parameter(A_log.detach().clone())
        self.A_log_rev._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        self.D_rev = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_rev._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        if inference_params is not None:
            raise NotImplementedError("Passing inference_params not yet supported!")

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A_rev = -torch.exp(self.A_log_rev.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            raise NotImplementedError("fast_path not yet supported!")
        else:
            x, z = xz.chunk(2, dim=1)
            x_rev = x.flip(dims=(-1,))  # since we rearranged above, seq length is in last dim
            z_rev = z.flip(dims=(-1,))

            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
                x_rev = self.act(self.conv1d_rev(x_rev)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                x_rev = causal_conv1d_fn(
                    x=x_rev,
                    weight=rearrange(self.conv1d_rev.weight, "d 1 w -> d w"),
                    bias=self.conv1d_rev.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

            # Run forward pass SSM
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )

            # Run backward pass SSM
            x_dbl_rev = self.x_proj_rev(rearrange(x_rev, "b d l -> (b l) d"))  # (bl d)
            dt_rev, B_rev, C_rev = torch.split(
                x_dbl_rev, [self.dt_rank, self.d_state, self.d_state],
                dim=-1
            )
            dt_rev = self.dt_proj_rev.weight @ dt_rev.t()
            dt_rev = rearrange(dt_rev, "d (b l) -> b d l", l=seqlen)
            B_rev = rearrange(B_rev, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_rev = rearrange(C_rev, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y_rev = selective_scan_fn(
                x_rev,
                dt_rev,
                A_rev,
                B_rev,
                C_rev,
                self.D_rev.float(),
                z=z_rev,
                delta_bias=self.dt_proj_rev.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )

            # Combine forward and backward passes
            if self.bidirectional_strategy == "add":
                y = y + y_rev.flip(dims=(-1,))
            elif self.bidirectional_strategy == "ew_multiply":
                y = y * y_rev.flip(dims=(-1,))
            y = rearrange(y, "b d l -> b l d")

            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        raise NotImplementedError("`step` not supported for BidirectionalMamba!")
