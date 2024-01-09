"""RCPS version of Mamba Block.

"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mamba.mamba_ssm.modules.rc_wrapper import RCPSWrapperKeepDim, RCPSAddNormWrapperKeepDim

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
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
        prenorm Transformer block that was defined.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # Cannot use fused add norm because we need to control residual connection in RCPS manner
        if fused_add_norm:
            print("WARNING: `fused_add_norm` is not supported in RCBlock. Defaulting to `False`.")
        self.fused_add_norm = False
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mixer = RCPSWrapperKeepDim(mixer_cls(dim), dim=dim, **factory_kwargs)
        # Divide dim by 2 because we concatenate fwd and rc along channel dim, and so we project down on each strand
        self.norm = RCPSAddNormWrapperKeepDim(norm_cls(dim // 2), dim=dim, **factory_kwargs)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual)).
            inference_params: inference parameters for mixer.
        """
        residual, hidden = self.norm(hidden_states, residual)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
