"""Model loader - loads pre-trained model without exposing architecture.

This module provides a simplified interface to load the pre-trained
coal maceral segmentation model. The model architecture is based on
the DA-VIT paper (arXiv:2506.12712).
"""

import os
import urllib.request
from pathlib import Path

import torch

CHECKPOINT_PATH = "best_mIoU.pth"
CHECKPOINT_URL = (
    "https://github.com/Nasor2/coal-maceral-segmentation/"
    "releases/download/v1.0.0/best_mIoU.pth"
)


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: str = "cpu"):
    """Load pre-trained model from checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load the model on ('cpu' or 'cuda').

    Returns:
        Loaded model in evaluation mode.
    """
    # Lazy import to avoid loading architecture at module level
    from model import DAViTModel

    model = DAViTModel("tiny", num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def ensure_checkpoint():
    """Download model checkpoint if it doesn't exist."""
    if os.path.exists(CHECKPOINT_PATH):
        return True
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        return True
    except Exception:
        return False
