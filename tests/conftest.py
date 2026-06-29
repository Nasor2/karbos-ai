"""Fixtures compartidos para tests de Karbos AI."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_coal_image():
    """Imagen sintética 512x512 simulando sección pulida de carbón."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[50:200, 50:200] = [255, 0, 0]    # Vitrinita (rojo)
    img[50:200, 250:400] = [255, 255, 0]  # Inertinita (amarillo)
    img[300:450, 50:200] = [0, 0, 255]    # Liptinita (azul)
    return Image.fromarray(img)


@pytest.fixture
def tiny_image():
    """Imagen pequeña para tests rápidos."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
