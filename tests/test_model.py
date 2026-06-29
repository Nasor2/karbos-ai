"""Tests de smoke para la arquitectura DA-VIT."""

import torch

from model import DCSA, DAViTModel, DAViTStage


def test_dcsa_forward():
    """DCSA produce output con mismo shape que input."""
    block = DCSA(64)
    x = torch.randn(1, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape


def test_davit_stage_forward():
    """DAViTStage con downsample reduce spatial dims y duplica canales."""
    stage = DAViTStage(64, num_blocks=2, downsample=True)
    x = torch.randn(1, 4096, 64)  # (B, L, C) where L=64*64
    out, H, W = stage(x, 64, 64)
    # Downsample: H/2=32, W/2=32, channels*2=128
    assert out.shape == (1, 1024, 128)  # (B, 32*32, 128)
    assert H == 32 and W == 32


def test_davit_model_output_shape():
    """DAViTModel tiny produce output a 1/4 de resolución (64x64 para input 512x512)."""
    model = DAViTModel("tiny", num_classes=4)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    # El FPN head produce output a la resolución del feature map más pequeño
    assert out.shape[0] == 1
    assert out.shape[1] == 4  # num_classes


def test_davit_model_num_params():
    """DAViTModel tiny tiene ~4.95M parámetros."""
    model = DAViTModel("tiny", num_classes=4)
    num_params = sum(p.numel() for p in model.parameters())
    assert 4_000_000 < num_params < 6_000_000, f"Got {num_params} params"


def test_davit_model_eval_mode():
    """Modelo en modo eval produce output determinista."""
    model = DAViTModel("tiny", num_classes=4)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_davit_invalid_variant():
    """Variante inválida lanza ValueError."""
    try:
        DAViTModel("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_davit_model_4_classes():
    """Modelo con 4 clases produce 4 canales de salida."""
    model = DAViTModel("tiny", num_classes=4)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape[1] == 4
