"""Model loader - loads ONNX model for inference.

This module provides a simplified interface to load the pre-trained
coal maceral segmentation model in ONNX format.
"""

import onnxruntime as ort

CHECKPOINT_PATH = "best_mIoU.onnx"


def load_model(model_path: str = CHECKPOINT_PATH):
    """Load ONNX model for inference.

    Args:
        model_path: Path to the .onnx model file.

    Returns:
        ONNX Runtime InferenceSession.
    """
    session = ort.InferenceSession(model_path)
    return session
