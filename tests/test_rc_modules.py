"""Tests for RCPS modules.

"""

from functools import partial

import pytest
import torch
from torch import nn

from mamba.mamba_ssm.modules.mamba_simple import Mamba
from mamba.mamba_ssm.modules.mamba_simple_rc import RCBlock
from mamba.mamba_ssm.modules.rc_wrapper import (
    RCPSEmbedding, RCPSWrapper, RCPSWrapperKeepDim, RCPSAddNormWrapper, RCPSAddNormWrapperKeepDim
)

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize("d_model", [256])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rcps_embedding(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cpu")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"A": 0, "C": 1, "G": 2, "T": 3}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {str_to_id[k]: str_to_id[v] for k, v in complement_map.items()}

    # Generate random sequences
    input_ids = torch.randint(low=0, high=4, size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC equivariance of embedding layer
    factory_kwargs = {"device": device, "dtype": dtype}
    embedding = RCPSEmbedding(vocab_size=4, d_model=d_model, complement_map=complement_map, **factory_kwargs).to(device)
    out_embed = embedding(input_ids)
    rc_out_embed = torch.flip(embedding(rc_input_ids), dims=[-2, -1])
    assert tuple(out_embed.size()) == (batch_size, seq_len, d_model)
    assert tuple(rc_out_embed.size()) == (batch_size, seq_len, d_model)
    assert torch.allclose(out_embed.detach(), rc_out_embed.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rcps_wrapper(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    factory_kwargs = {"device": device, "dtype": dtype}
    module = nn.Sequential(
        nn.Linear(d_model, d_model, bias=False, **factory_kwargs),
        nn.ReLU(),
        nn.Linear(d_model, d_model*2, bias=True, **factory_kwargs),
        nn.ReLU(),
        nn.Linear(d_model * 2, d_model, bias=True, **factory_kwargs)
    )

    # Test RC equivariance of wrapper
    rcps_module = RCPSWrapper(module).to(device)
    out = rcps_module(x)
    rc_out = torch.flip(rcps_module(rc_x), dims=[-2, -1])
    assert tuple(out.size()) == (batch_size, seq_len, d_model * 2)
    assert tuple(rc_out.size()) == (batch_size, seq_len, d_model * 2)
    assert torch.allclose(out.detach(), rc_out.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rcps_wrapper_keep_dim(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    factory_kwargs = {"device": device, "dtype": dtype}
    module = nn.Sequential(
        nn.Linear(d_model, d_model, bias=False, **factory_kwargs),
        nn.ReLU(),
        nn.Linear(d_model, d_model*2, bias=True, **factory_kwargs),
        nn.ReLU(),
        nn.Linear(d_model * 2, d_model, bias=True, **factory_kwargs)
    )

    # Test RC equivariance of wrapper
    rcps_module = RCPSWrapperKeepDim(module, dim=d_model, **factory_kwargs).to(device)
    out = rcps_module(x)
    rc_out = torch.flip(rcps_module(rc_x), dims=[-2, -1])
    assert tuple(out.size()) == (batch_size, seq_len, d_model)
    assert tuple(rc_out.size()) == (batch_size, seq_len, d_model)
    assert torch.allclose(out.detach(), rc_out.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rcps_add_norm_wrapper(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    factory_kwargs = {"device": device, "dtype": dtype}
    norm = RMSNorm(d_model, eps=1e-5, **factory_kwargs)

    # Test RC equivariance of wrapper
    rcps_module = RCPSAddNormWrapper(norm).to(device)
    out = rcps_module(x)
    rc_out = tuple([torch.flip(r, dims=[-2, -1]) for r in rcps_module(rc_x)])
    for f, r in zip(out, rc_out):
        assert tuple(f.size()) == (batch_size, seq_len, d_model * 2)
        assert tuple(r.size()) == (batch_size, seq_len, d_model * 2)
        assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rcps_add_norm_wrapper_keep_dim(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    factory_kwargs = {"device": device, "dtype": dtype}
    norm = nn.LayerNorm(d_model // 2, eps=1e-5, **factory_kwargs)

    # Test RC equivariance of wrapper
    rcps_module = RCPSAddNormWrapperKeepDim(norm, dim=d_model, **factory_kwargs).to(device)
    out = rcps_module(x)
    rc_out = tuple([torch.flip(r, dims=[-2, -1]) for r in rcps_module(rc_x)])
    for f, r in zip(out, rc_out):
        assert tuple(f.size()) == (batch_size, seq_len, d_model)
        assert tuple(r.size()) == (batch_size, seq_len, d_model)
        assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("rms_norm", [False, True])
def test_rcps_block(batch_size, seq_len, d_model, dtype, rms_norm):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Instantiate model
    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5, **factory_kwargs
    )
    mamba_block = RCBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=False,
        residual_in_fp32=True,
        **factory_kwargs,
    ).to(device)

    # Generate random sequence
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    # Test RC equivariance of wrapped mamba layer
    out = mamba_block(x)
    rc_out = tuple([torch.flip(r, dims=[-2, -1]) for r in mamba_block(rc_x)])
    for f, r in zip(out, rc_out):
        assert tuple(f.size()) == (batch_size, seq_len, d_model)
        assert tuple(r.size()) == (batch_size, seq_len, d_model)
        assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)