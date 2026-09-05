"""Model loader - loads encrypted ONNX model for inference.

This module decrypts and loads the pre-trained coal maceral segmentation
model. The model is encrypted with Fernet (AES-128-CBC) and stored as
best_mIoU.onnx.enc. Decryption happens at runtime using a key from
Streamlit secrets.
"""

import tempfile
import os

import onnxruntime as ort
import streamlit as st
from cryptography.fernet import Fernet

ENCRYPTED_MODEL_PATH = "best_mIoU.onnx.enc"


def _get_fernet():
    """Get Fernet instance from Streamlit secrets."""
    try:
        key = st.secrets["MODEL_KEY"]
        return Fernet(key.encode() if isinstance(key, str) else key)
    except KeyError:
        raise ValueError(
            "MODEL_KEY not found in Streamlit secrets. "
            "Please add it in Streamlit Cloud → Settings → Secrets.\n"
            "Format: MODEL_KEY = \"your-key-here\""
        )
    except Exception as e:
        raise ValueError(f"Invalid MODEL_KEY in secrets: {e}")


def load_model(model_path: str = ENCRYPTED_MODEL_PATH):
    """Load encrypted ONNX model for inference.

    Args:
        model_path: Path to the encrypted .onnx.enc file.

    Returns:
        ONNX Runtime InferenceSession.
    """
    fernet = _get_fernet()

    with open(model_path, "rb") as f:
        encrypted_data = f.read()

    decrypted_data = fernet.decrypt(encrypted_data)

    # Write to temporary file (ONNX Runtime requires a file path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(decrypted_data)

        session = ort.InferenceSession(tmp_path)
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return session
