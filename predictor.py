"""Prediction wrapper - ONNX-based inference for model predictions.

This module provides a high-level interface for coal maceral segmentation
using ONNX Runtime for inference. No PyTorch dependency required.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from config import CLASS_COLORS, CLASS_NAMES, IMG_SIZE

# --- Confidence Thresholds (for UI display) ---
CONFIDENCE_THRESHOLDS = {
    "high": 0.90,
    "medium": 0.75,
    "low": 0.0,
}

# --- ImageNet Normalization ---
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)


def preprocess(image_input, img_size: int = IMG_SIZE):
    """Preprocess an image for inference.

    Args:
        image_input: Path (str), file-like object, or PIL Image.
        img_size: Resolution size (default 512).

    Returns:
        Tuple of (numpy array [1,3,H,W], original size (W,H), original PIL Image).
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")
    original_size = img.size  # (W, H)
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    tensor = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1)  # CHW
    tensor = (tensor - MEAN) / STD
    return tensor[np.newaxis, ...], original_size, img  # BCHW


def predict(session, tensor, original_size):
    """Run inference and return class mask + confidence.

    Args:
        session: ONNX Runtime InferenceSession.
        tensor: Preprocessed numpy array [1, 3, H, W].
        original_size: Original image size (W, H).

    Returns:
        Tuple of (class mask [H, W], confidence map [H, W]).
    """
    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0]  # [1, 4, 64, 64]

    # Squeeze batch dimension: [4, 64, 64]
    logits_2d = logits.squeeze(0)

    # Resize to original size using scipy
    orig_h, orig_w = original_size[1], original_size[0]
    scale_h = orig_h / logits_2d.shape[1]
    scale_w = orig_w / logits_2d.shape[2]
    logits_resized = zoom(logits_2d, (1, scale_h, scale_w), order=1)

    # Get class mask (argmax)
    mask = logits_resized.argmax(axis=0).astype(np.int32)

    # Get confidence (softmax then max)
    # Subtract max for numerical stability
    logits_shifted = logits_resized - logits_resized.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
    confidence = probs.max(axis=0)

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
    high = CONFIDENCE_THRESHOLDS["high"]
    medium = CONFIDENCE_THRESHOLDS["medium"]
    return {
        "mean": round(float(confidence.mean()), 3),
        "low_pct": round(float((confidence < medium).sum() / confidence.size * 100), 1),
        "high_pct": round(float((confidence >= high).sum() / confidence.size * 100), 1),
    }
