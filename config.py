"""Configuration for Karbos AI.

Simplified configuration with only UI-related constants.
Model architecture and training details are proprietary.
"""

import numpy as np

# --- Class Names and Colors ---
CLASS_NAMES = ["Vitrinita", "Inertinita", "Liptinita", "Fondo"]
CLASS_COLORS = np.array(
    [
        [255, 0, 0],      # Vitrinita - Red
        [255, 255, 0],    # Inertinita - Yellow
        [0, 0, 255],      # Liptinita - Blue
        [0, 0, 0],        # Fondo - Black
    ],
    dtype=np.uint8,
)

# --- Model Configuration ---
IMG_SIZE = 512
NUM_CLASSES = 4

# --- Supported Formats ---
SUPPORTED_EXTENSIONS = {"tiff", "tif", "png", "jpg", "jpeg"}

# --- Maceral Colors (hex for Plotly) ---
MACERAL_COLORS = {
    "Vitrinita": "#FF0000",
    "Inertinita": "#FFD700",
    "Liptinita": "#0000FF",
    "Fondo": "#808080",
}

# --- Industrial Classification Colors ---
CLASSIFICATION_COLORS = {
    "Coqueable Primario": "#22C55E",
    "Coqueable Secundario": "#84CC16",
    "Rico en Liptinita": "#3B82F6",
    "Térmico": "#EF4444",
    "Mixto": "#A855F7",
}

# --- Assumed Ash Percentage ---
ASSUMED_ASH_PCT = 15
