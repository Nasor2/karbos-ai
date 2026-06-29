"""Tests para el pipeline de inferencia."""

import numpy as np
from PIL import Image

from inference import compute_composition, confidence_statistics, decode_mask, preprocess


def test_preprocess_returns_correct_shapes():
    """preprocess retorna tensor [1,3,512,512], tamaño original, imagen PIL."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
    tensor, orig_size, pil_img = preprocess(img)
    assert tensor.shape == (1, 3, 512, 512)
    assert orig_size == (200, 100)  # (W, H)
    assert isinstance(pil_img, Image.Image)


def test_preprocess_normalization():
    """preprocess normaliza con mean/std de ImageNet."""
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    tensor, _, _ = preprocess(img)
    # Valor negro normalizado: (0 - mean) / std
    assert tensor.mean() < 0  # Should be negative since mean > 0


def test_decode_mask_colors():
    """decode_mask asigna colores correctos por clase."""
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[0:5, :] = 0  # Vitrinite (rojo)
    mask[5:10, :] = 1  # Inertinite (amarillo)
    rgb = decode_mask(mask)
    arr = np.array(rgb)
    assert arr[2, 2, 0] == 255  # Rojo
    assert arr[2, 2, 1] == 0
    assert arr[7, 2, 0] == 255  # Amarillo
    assert arr[7, 2, 1] == 255


def test_compute_composition():
    """compute_comcomposition retorna porcentajes correctos."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[0:60, :] = 0  # 60% Vitrinita
    mask[60:80, :] = 1  # 20% Inertinita
    mask[80:90, :] = 2  # 10% Liptinita
    mask[90:100, :] = 3  # 10% Fondo
    comp = compute_composition(mask)
    assert comp["Vitrinita"] == 60.0
    assert comp["Inertinita"] == 20.0
    assert comp["Liptinita"] == 10.0
    assert comp["Fondo"] == 10.0
    assert abs(sum(comp.values()) - 100.0) < 0.1


def test_confidence_statistics():
    """confidence_statistics calcula métricas correctamente."""
    conf = np.ones((100, 100), dtype=np.float32)
    conf[0:30, :] = 0.5  # 30% baja confianza
    stats = confidence_statistics(conf)
    assert stats["low_pct"] == 30.0
    assert stats["high_pct"] == 70.0
    # mean = 0.7*1.0 + 0.3*0.5 = 0.85
    assert stats["mean"] == 0.85


def test_confidence_all_high():
    """confidence_statistics con confianza toda alta."""
    conf = np.ones((100, 100), dtype=np.float32) * 0.95
    stats = confidence_statistics(conf)
    assert stats["low_pct"] == 0.0
    assert stats["high_pct"] == 100.0
