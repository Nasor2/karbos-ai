"""Pipeline de inferencia para segmentación de macerales de carbón.

Carga del modelo, preprocesamiento, predicción y post-procesamiento.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    CHECKPOINT_PATH,
    CLASS_COLORS,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLDS,
    DEVICE,
    IMG_SIZE,
    MEAN,
    STD,
)
from model import DAViTModel


def load_model(
    checkpoint_path: str = CHECKPOINT_PATH,
    device: str = DEVICE,
) -> DAViTModel:
    """Carga el modelo DA-VIT desde un checkpoint.

    Args:
        checkpoint_path: Ruta al archivo .pth del checkpoint.
        device: Dispositivo ('cpu' o 'cuda').

    Returns:
        Modelo en modo evaluación.
    """
    model = DAViTModel("tiny", num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def preprocess(
    image_input, img_size: int = IMG_SIZE
) -> tuple[torch.Tensor, tuple[int, int], Image.Image]:
    """Preprocesa una imagen para inferencia.

    Args:
        image_input: Ruta (str), file-like object, o PIL Image.
        img_size: Tamaño de resolución (default 512).

    Returns:
        Tupla de (tensor [1,3,H,W], tamaño original (W,H), imagen PIL original).
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")
    original_size = img.size  # (W, H)
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0), original_size, img


@torch.no_grad()
def predict(
    model: DAViTModel,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta inferencia y retorna máscara de clase + confianza.

    Args:
        model: Modelo DA-VIT cargado.
        tensor: Tensor preprocesado [1, 3, H, W].

    Returns:
        Tupla de (máscara de clases [H, W], confianza [H, W]).
    """
    device = next(model.parameters()).device
    logits = model(tensor.to(device))
    probs = F.softmax(logits, dim=1)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    confidence = probs.max(dim=1)[0].squeeze(0).cpu().numpy()
    return mask, confidence


def decode_mask(mask: np.ndarray) -> Image.Image:
    """Convierte índices de clase a imagen RGB para visualización.

    Args:
        mask: Máscara de clases [H, W].

    Returns:
        Imagen PIL RGB con colores por maceral.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_COLORS)):
        rgb[mask == c] = CLASS_COLORS[c]
    return Image.fromarray(rgb)


def compute_composition(mask: np.ndarray) -> dict[str, float]:
    """Calcula composición maceral porcentual.

    Args:
        mask: Máscara de clases [H, W].

    Returns:
        Diccionario {nombre_clase: porcentaje}.
    """
    total = mask.size
    composition = {}
    for i, name in enumerate(CLASS_NAMES):
        count = (mask == i).sum()
        composition[name] = round(count / total * 100, 1)
    return composition


def confidence_statistics(confidence: np.ndarray) -> dict[str, float]:
    """Calcula estadísticas de confianza del modelo.

    Args:
        confidence: Mapa de confianza [H, W].

    Returns:
        Diccionario con media, %baja confianza, %alta confianza.
    """
    high = CONFIDENCE_THRESHOLDS["high"]
    low = CONFIDENCE_THRESHOLDS["medium"]
    return {
        "mean": round(float(confidence.mean()), 3),
        "low_pct": round(float((confidence < low).sum() / confidence.size * 100), 1),
        "high_pct": round(float((confidence >= high).sum() / confidence.size * 100), 1),
    }
