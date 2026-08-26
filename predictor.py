"""Prediction wrapper - simplified interface for model inference.

This module provides a high-level interface for coal maceral segmentation
without exposing the internal model architecture or processing details.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLASS_COLORS, CLASS_NAMES, IMG_SIZE, MEAN, STD


def preprocess(image_input, img_size: int = IMG_SIZE):
    """Preprocess an image for inference.

    Args:
        image_input: Path (str), file-like object, or PIL Image.
        img_size: Resolution size (default 512).

    Returns:
        Tuple of (tensor [1,3,H,W], original size (W,H), original PIL Image).
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")
    original_size = img.size
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0), original_size, img


@torch.no_grad()
def predict(model, tensor):
    """Run inference and return class mask + confidence.

    Args:
        model: Loaded model.
        tensor: Preprocessed tensor [1, 3, H, W].

    Returns:
        Tuple of (class mask [H, W], confidence map [H, W]).
    """
    device = next(model.parameters()).device
    logits = model(tensor.to(device))
    probs = F.softmax(logits, dim=1)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    confidence = probs.max(dim=1)[0].squeeze(0).cpu().numpy()
    return mask, confidence


def decode_mask(mask: np.ndarray) -> Image.Image:
    """Convert class indices to RGB visualization.

    Args:
        mask: Class mask [H, W].

    Returns:
        PIL RGB image with colors per maceral.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_COLORS)):
        rgb[mask == c] = CLASS_COLORS[c]
    return Image.fromarray(rgb)


def compute_composition(mask: np.ndarray) -> dict:
    """Calculate maceral composition percentage.

    Args:
        mask: Class mask [H, W].

    Returns:
        Dictionary {class_name: percentage}.
    """
    total = mask.size
    composition = {}
    for i, name in enumerate(CLASS_NAMES):
        count = (mask == i).sum()
        composition[name] = round(count / total * 100, 1)
    return composition


def confidence_statistics(confidence: np.ndarray) -> dict:
    """Calculate model confidence statistics.

    Args:
        confidence: Confidence map [H, W].

    Returns:
        Dictionary with mean, low_pct, high_pct.
    """
    high = 0.90
    medium = 0.75
    return {
        "mean": round(float(confidence.mean()), 3),
        "low_pct": round(float((confidence < medium).sum() / confidence.size * 100), 1),
        "high_pct": round(float((confidence >= high).sum() / confidence.size * 100), 1),
    }
