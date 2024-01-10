"""Tests for RCPS models.

"""

import pytest
import torch

from mamba.mamba_ssm.models.config_mamba import MambaConfig
from mamba.mamba_ssm.models.mixer_seq_simple_rc import RCPSMixerModel, RCPSMambaLMHeadModel


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1024, 2048])
@pytest.mark.parametrize("n_layer", [1, 2])
@pytest.mark.parametrize("d_model", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_rcps_backbone(batch_size, seq_len, n_layer, d_model, dtype, bidirectional):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"A": 0, "C": 1, "G": 2, "T": 3}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {str_to_id[k]: str_to_id[v] for k, v in complement_map.items()}

    # Setup MambaConfig
    initializer_cfg = {"initializer_range": 0.02, "rescale_prenorm_residual": True, "n_residuals_per_layer": 1}
    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=len(str_to_id),
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
    )
    factory_kwargs = {"device": device, "dtype": dtype}

    # Instantiate model
    backbone = RCPSMixerModel(
        d_model=d_model,
        n_layer=config.n_layer,
        vocab_size=config.vocab_size,
        complement_map=complement_map,
        ssm_cfg=config.ssm_cfg,
        rms_norm=config.rms_norm,
        initializer_cfg=initializer_cfg,
        fused_add_norm=config.fused_add_norm,
        residual_in_fp32=config.residual_in_fp32,
        bidirectional=bidirectional,
        bidirectional_strategy="add",
        **factory_kwargs,
    ).to(device)

    # Generate random sequences
    input_ids = torch.randint(low=0, high=4, size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC equivariance of rc backbone
    out = backbone(input_ids)
    rc_out = backbone(rc_input_ids)
    # Hidden state size should double
    assert tuple(out.size()) == (batch_size, seq_len, d_model * 2)
    assert tuple(rc_out.size()) == (batch_size, seq_len, d_model * 2)
    assert torch.allclose(out.detach(), torch.flip(rc_out.detach(), dims=[1, 2]), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1024, 2048])
@pytest.mark.parametrize("n_layer", [1, 4])
@pytest.mark.parametrize("d_model", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_rcps_mamba_lm(batch_size, seq_len, n_layer, d_model, dtype, bidirectional):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }

    # Setup MambaConfig
    initializer_cfg = {"initializer_range": 0.02, "rescale_prenorm_residual": True, "n_residuals_per_layer": 1}
    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=len(str_to_id),
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
    )
    factory_kwargs = {"device": device, "dtype": dtype}

    # Instantiate model
    mamba_lm = RCPSMambaLMHeadModel(
        config=config,
        complement_map=complement_map,
        initializer_cfg=initializer_cfg,
        **factory_kwargs,
    ).to(device)

    # Generate random sequences
    input_ids = torch.randint(low=2, high=len(str_to_id), size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC equivariance of rc backbone
    out = mamba_lm(input_ids)
    rc_out = mamba_lm(rc_input_ids)
    if config.vocab_size % config.pad_vocab_size_multiple != 0:
        config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
    assert tuple(out.logits.size()) == (batch_size, seq_len, config.vocab_size)
    assert tuple(rc_out.logits.size()) == (batch_size, seq_len, config.vocab_size)
    assert torch.allclose(out.logits.detach(), torch.flip(rc_out.logits.detach(), dims=[1, 2]), rtol=rtol, atol=atol)
