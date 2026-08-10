from pathlib import Path

import numpy as np
import torch

from cogtrack.models.sutrack.network.clip_text import (
    CLIPTextTower,
    convert_clip_text_weights_to_fp16,
)
from cogtrack.models.sutrack.preprocessing import (
    ImagePreprocessor,
    clip_box,
    hann2d,
    sample_target,
    transform_image_to_crop,
)
from cogtrack.models.sutrack.runtime import ConfigNode, _resolve_file


def test_config_node_and_environment_path_resolution(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "model.pth.tar"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("TEST_BUILTIN_SUTRACK", str(checkpoint))
    params = {"runtime": {"project_root": str(tmp_path), "model_root": str(tmp_path)}}

    node = ConfigNode({"MODEL": {"ENCODER": {"TYPE": "fastitpnb"}}})
    assert node.MODEL.ENCODER.TYPE == "fastitpnb"
    assert _resolve_file(
        "${TEST_BUILTIN_SUTRACK}", params=params, prefer_model_root=True
    ) == checkpoint


def test_preprocessing_and_geometry_are_device_agnostic():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[10:25, 20:35] = 255
    patch, scale = sample_target(image, [20, 10, 15, 15], 2.0, 32)
    assert patch.shape == (32, 32, 3)
    assert scale > 0

    tensor = ImagePreprocessor(torch.device("cpu")).process(
        np.concatenate((patch, patch), axis=2)
    )
    assert tensor.shape == (1, 6, 32, 32)
    annotation = transform_image_to_crop([20, 10, 15, 15], [20, 10, 15, 15], scale, 32)
    assert annotation.shape == (4,)
    assert torch.isfinite(annotation).all()
    assert hann2d(4, torch.device("cpu")).shape == (1, 1, 4, 4)
    assert clip_box([-5, -5, 20, 20], 40, 60) == [0.0, 0.0, 15.0, 15.0]


def test_minimal_clip_text_tower_forward_shape():
    tower = CLIPTextTower(
        embed_dim=8,
        context_length=4,
        vocab_size=16,
        width=8,
        layers=1,
        heads=2,
    ).eval()
    tokens = torch.tensor([[1, 2, 15, 0]], dtype=torch.long)
    with torch.inference_mode():
        output = tower.encode_text(tokens)
    assert output.shape == (1, 8)
    assert torch.isfinite(output).all()


def test_clip_fp16_conversion_matches_official_mixed_dtype_layout():
    tower = CLIPTextTower(
        embed_dim=8,
        context_length=4,
        vocab_size=16,
        width=8,
        layers=1,
        heads=2,
    )
    convert_clip_text_weights_to_fp16(tower)

    block = tower.transformer.resblocks[0]
    assert tower.dtype == torch.float16
    assert tower.text_projection.dtype == torch.float16
    assert block.attn.in_proj_weight.dtype == torch.float16
    assert block.mlp.c_fc.weight.dtype == torch.float16
    assert tower.token_embedding.weight.dtype == torch.float32
    assert tower.positional_embedding.dtype == torch.float32
    assert tower.ln_final.weight.dtype == torch.float32
