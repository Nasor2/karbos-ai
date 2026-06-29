"""Configuración central de Karbos AI.

Constantes, normalización, umbrales de confianza y colores de clase.
"""

import numpy as np
import torch

# --- Nombres y Colores de Clase ---
CLASS_NAMES = ["Vitrinita", "Inertinita", "Liptinita", "Fondo"]
CLASS_COLORS = np.array(
    [
        [255, 0, 0],      # Vitrinita — Rojo
        [255, 255, 0],    # Inertinita — Amarillo
        [0, 0, 255],      # Liptinita — Azul
        [0, 0, 0],        # Fondo — Negro
    ],
    dtype=np.uint8,
)

# --- Normalización (ImageNet) ---
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

# --- Configuración del Modelo ---
IMG_SIZE = 512
NUM_CLASSES = 4
CHECKPOINT_PATH = "best_mIoU.pth"
CHECKPOINT_URL = (
    "https://github.com/Nasor2/coal-maceral-segmentation/"
    "releases/download/v1.0.0/best_mIoU.pth"
)
DEVICE = "cpu"

# --- Pesos de Clase (para loss en entrenamiento, referencial) ---
CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 3.0, 0.5])

# --- Umbrales de Confianza ---
CONFIDENCE_THRESHOLDS = {
    "high": 0.90,
    "medium": 0.75,
    "low": 0.0,
}

CONFIDENCE_COLORS = {
    "high": "#22C55E",    # Verde
    "medium": "#EAB308",  # Amarillo
    "low": "#F97316",     # Naranja
}

# --- Formatos Soportados ---
SUPPORTED_EXTENSIONS = {"tiff", "tif", "png", "jpg", "jpeg"}

# --- Colores Macerales (hex para Plotly) ---
MACERAL_COLORS = {
    "Vitrinita": "#FF0000",
    "Inertinita": "#FFD700",
    "Liptinita": "#0000FF",
    "Fondo": "#808080",
}

# --- Clasificación Industrial ---
CLASSIFICATION_COLORS = {
    "Coqueable Primario": "#22C55E",
    "Coqueable Secundario": "#84CC16",
    "Rico en Liptinita": "#3B82F6",
    "Térmico": "#EF4444",
    "Mixto": "#A855F7",
}

# --- Parámetros Proximate ---
ASSUMED_ASH_PCT = 15  # Cenizas asumidas para estimación proximate
