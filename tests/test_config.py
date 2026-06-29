from config import (
    ASSUMED_ASH_PCT,
    CHECKPOINT_PATH,
    CHECKPOINT_URL,
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    CLASSIFICATION_COLORS,
    CONFIDENCE_COLORS,
    CONFIDENCE_THRESHOLDS,
    DEVICE,
    IMG_SIZE,
    MACERAL_COLORS,
    MEAN,
    NUM_CLASSES,
    STD,
    SUPPORTED_EXTENSIONS,
)


def test_class_names_count():
    assert len(CLASS_NAMES) == 4


def test_class_colors_shape():
    assert CLASS_COLORS.shape == (4, 3)


def test_class_weights_count():
    assert len(CLASS_WEIGHTS) == 4


def test_normalization_shapes():
    assert MEAN.shape == (3, 1, 1)
    assert STD.shape == (3, 1, 1)


def test_img_size():
    assert IMG_SIZE == 512


def test_num_classes():
    assert NUM_CLASSES == 4


def test_checkpoint_path():
    assert CHECKPOINT_PATH.endswith(".pth")


def test_device():
    assert DEVICE == "cpu"


def test_confidence_thresholds():
    assert CONFIDENCE_THRESHOLDS["high"] == 0.90
    assert CONFIDENCE_THRESHOLDS["medium"] == 0.75
    assert CONFIDENCE_THRESHOLDS["low"] == 0.0


def test_confidence_colors():
    assert "high" in CONFIDENCE_COLORS
    assert "medium" in CONFIDENCE_COLORS
    assert "low" in CONFIDENCE_COLORS


def test_supported_extensions():
    assert "tiff" in SUPPORTED_EXTENSIONS
    assert "png" in SUPPORTED_EXTENSIONS
    assert "jpeg" in SUPPORTED_EXTENSIONS


def test_maceral_colors():
    assert "Vitrinita" in MACERAL_COLORS
    assert "Inertinita" in MACERAL_COLORS
    assert "Liptinita" in MACERAL_COLORS
    assert MACERAL_COLORS["Vitrinita"].startswith("#")


def test_maceral_colors_background():
    assert "Fondo" in MACERAL_COLORS
    assert MACERAL_COLORS["Fondo"].startswith("#")


def test_checkpoint_url():
    assert isinstance(CHECKPOINT_URL, str)
    assert CHECKPOINT_URL.startswith("https://")
    assert CHECKPOINT_URL.endswith(".pth")


def test_supported_extensions_all():
    assert "tiff" in SUPPORTED_EXTENSIONS
    assert "tif" in SUPPORTED_EXTENSIONS
    assert "png" in SUPPORTED_EXTENSIONS
    assert "jpg" in SUPPORTED_EXTENSIONS
    assert "jpeg" in SUPPORTED_EXTENSIONS


def test_classification_colors():
    expected = ["Coqueable Primario", "Coqueable Secundario", "Rico en Liptinita", "Térmico", "Mixto"]
    for name in expected:
        assert name in CLASSIFICATION_COLORS
        assert CLASSIFICATION_COLORS[name].startswith("#")


def test_assumed_ash_pct():
    assert ASSUMED_ASH_PCT == 15
    assert isinstance(ASSUMED_ASH_PCT, int)
