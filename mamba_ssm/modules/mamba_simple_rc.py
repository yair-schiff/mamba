"""RCPS version of Mamba Block.

"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mamba.mamba_ssm.modules.rc_wrapper import RCPSAddNormWrapper, RCPSWrapper

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class RCBlock(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ):
        """
        RCPS version of simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        We maintain the slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mixer = RCPSWrapper(mixer_cls(dim))
        norm_f = norm_cls(dim)
        self.norm = norm_f if fused_add_norm else RCPSAddNormWrapper(norm_f)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual)).
            inference_params: inference parameters for mixer.
        """
        if not self.fused_add_norm:
            hidden_states, residual = self.norm(hidden_states, residual=residual)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn

            hidden_states_fwd, residual_fwd = fused_add_norm_fn(
                hidden_states[..., hidden_states.shape[-1] // 2:],
                self.norm.weight,
                self.norm.bias,
                residual=residual[..., hidden_states.shape[-1] // 2:] if residual is not None else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

            hidden_states_rc, residual_rc = fused_add_norm_fn(
                hidden_states[..., :hidden_states.shape[-1] // 2].flip(dims=[-2, -1]),
                self.norm.weight,
                self.norm.bias,
                residual=residual[..., :hidden_states.shape[-1] // 2].flip(dims=[-2, -1]) if residual is not None else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            residual = torch.cat([residual_fwd, residual_rc.flip(dims=[-2, -1])], dim=-1)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
