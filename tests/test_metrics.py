"""Tests para métricas de calidad del carbón."""

from metrics import (
    aggregate_compositions,
    aggregate_metrics,
    classify_coal,
    compute_quality_metrics,
    estimate_proximate,
)


def test_compute_quality_metrics_coal_riqueable():
    """ Carbón coqueable: alta vitrinita, bajo inertinita."""
    comp = {"Vitrinita": 65, "Inertinita": 15, "Liptinita": 10, "Fondo": 10}
    m = compute_quality_metrics(comp)
    assert m["TRI"] == 70.0  # 65 + 0.5*10
    assert m["V/I"] == 4.33
    assert m["R/I"] == 3.0  # (65+10)/(15+10)
    assert m["%Reactivos"] == 75.0
    assert m["%Inertes"] == 25.0


def test_compute_quality_metrics_thermal():
    """ Carbón térmico: alta inertinita."""
    comp = {"Vitrinita": 20, "Inertinita": 60, "Liptinita": 5, "Fondo": 15}
    m = compute_quality_metrics(comp)
    assert m["TRI"] == 22.5  # 20 + 0.5*5
    assert m["V/I"] == 0.33
    assert m["%Inertes"] == 75.0


def test_compute_quality_metrics_zero_inertinite():
    """División por cero en V/I cuando Inertinita=0."""
    comp = {"Vitrinita": 80, "Inertinita": 0, "Liptinita": 10, "Fondo": 10}
    m = compute_quality_metrics(comp)
    assert m["V/I"] == float("inf")


def test_classify_coal_primary():
    """Clasificación coqueable primario (spec matrix 4.2: V>60 y V/I>1.5)."""
    comp = {"Vitrinita": 65, "Inertinita": 15, "Liptinita": 10, "Fondo": 10}
    result = classify_coal(comp)
    assert result == "Coqueable Primario"


def test_classify_coal_secondary():
    """Clasificación coqueable secundario (V>50 y V/I>1.5)."""
    comp = {"Vitrinita": 55, "Inertinita": 25, "Liptinita": 10, "Fondo": 10}
    result = classify_coal(comp)
    assert result == "Coqueable Secundario"


def test_classify_coal_liptinite_rich():
    """Rico en Liptinita (L>20%)."""
    comp = {"Vitrinita": 30, "Inertinita": 20, "Liptinita": 35, "Fondo": 15}
    result = classify_coal(comp)
    assert result == "Rico en Liptinita"


def test_classify_coal_thermal():
    """Clasificación térmico (I>50%)."""
    comp = {"Vitrinita": 20, "Inertinita": 55, "Liptinita": 5, "Fondo": 20}
    result = classify_coal(comp)
    assert result == "Térmico"


def test_classify_coal_mixed():
    """Clasificación mixto."""
    comp = {"Vitrinita": 40, "Inertinita": 30, "Liptinita": 10, "Fondo": 20}
    result = classify_coal(comp)
    assert result == "Mixto"


def test_estimate_proximate():
    """Estimaciones proximate razonables."""
    comp = {"Vitrinita": 60, "Inertinita": 20, "Liptinita": 10, "Fondo": 10}
    p = estimate_proximate(comp)
    assert 30 <= p["VM%"] <= 70
    assert p["FC%"] > 0
    assert p["Cenizas%"] == 15
    assert 7000 < p["CV (kcal/kg)"] < 12000


def test_estimate_proximate_high_liptinite():
    """Liptinita alta incrementa VM."""
    comp_high_l = {"Vitrinita": 40, "Inertinita": 20, "Liptinita": 30, "Fondo": 10}
    comp_low_l = {"Vitrinita": 40, "Inertinita": 20, "Liptinita": 5, "Fondo": 35}
    p_high = estimate_proximate(comp_high_l)
    p_low = estimate_proximate(comp_low_l)
    assert p_high["VM%"] > p_low["VM%"]


# --- Tests de agregación multi-imagen ---


def test_aggregate_compositions_single():
    """Agregación con una sola imagen retorna el mismo valor."""
    comp = [{"Vitrinita": 60, "Inertinita": 20, "Liptinita": 10, "Fondo": 10}]
    result = aggregate_compositions(comp)
    assert result["Vitrinita"]["mean"] == 60.0
    assert result["Vitrinita"]["std"] == 0.0


def test_aggregate_compositions_multiple():
    """Agregación con múltiples imágenes calcula promedio y desviación."""
    comps = [
        {"Vitrinita": 60, "Inertinita": 20, "Liptinita": 10, "Fondo": 10},
        {"Vitrinita": 50, "Inertinita": 30, "Liptinita": 10, "Fondo": 10},
    ]
    result = aggregate_compositions(comps)
    assert result["Vitrinita"]["mean"] == 55.0
    assert result["Vitrinita"]["std"] > 0
    assert result["Inertinita"]["mean"] == 25.0


def test_aggregate_compositions_three_images():
    """Agregación con tres imágenes."""
    comps = [
        {"Vitrinita": 60, "Inertinita": 20, "Liptinita": 10, "Fondo": 10},
        {"Vitrinita": 50, "Inertinita": 30, "Liptinita": 10, "Fondo": 10},
        {"Vitrinita": 70, "Inertinita": 10, "Liptinita": 10, "Fondo": 10},
    ]
    result = aggregate_compositions(comps)
    assert result["Vitrinita"]["mean"] == 60.0
    assert result["Inertinita"]["mean"] == 20.0


def test_aggregate_metrics_single():
    """Agregación de métricas con una sola imagen."""
    metrics = [
        {"TRI": 65.0, "V/I": 3.0, "R/I": 2.5, "%Reactivos": 70.0, "%Inertes": 30.0},
    ]
    result = aggregate_metrics(metrics)
    assert result["TRI"]["mean"] == 65.0
    assert result["TRI"]["std"] == 0.0
    assert result["V/I"]["mean"] == 3.0


def test_aggregate_metrics_multiple():
    """Agregación de métricas con múltiples imágenes."""
    metrics = [
        {"TRI": 65.0, "V/I": 3.0, "R/I": 2.5, "%Reactivos": 70.0, "%Inertes": 30.0},
        {"TRI": 55.0, "V/I": 2.0, "R/I": 1.8, "%Reactivos": 60.0, "%Inertes": 40.0},
    ]
    result = aggregate_metrics(metrics)
    assert result["TRI"]["mean"] == 60.0
    assert result["V/I"]["mean"] == 2.5
    assert result["R/I"]["mean"] == 2.15
    assert result["TRI"]["std"] > 0


def test_aggregate_metrics_with_inf():
    """Agregación maneja V/I = inf correctamente."""
    metrics = [
        {"TRI": 65.0, "V/I": float("inf"), "R/I": 2.5, "%Reactivos": 70.0, "%Inertes": 30.0},
        {"TRI": 55.0, "V/I": 2.0, "R/I": 1.8, "%Reactivos": 60.0, "%Inertes": 40.0},
    ]
    result = aggregate_metrics(metrics)
    # Solo incluye valores finitos en el promedio
    assert result["V/I"]["mean"] == 2.0
    assert result["TRI"]["mean"] == 60.0


# --- Tests de edge cases ---


def test_aggregate_compositions_empty():
    """Agregación con lista vacía retorna dict vacío."""
    result = aggregate_compositions([])
    assert result == {}


def test_aggregate_metrics_empty():
    """Agregación de métricas con lista vacía retorna dict vacío."""
    result = aggregate_metrics([])
    assert result == {}


def test_estimate_proximate_high_liptinite_capped():
    """VM% se trunca en 100% para liptinita extrema."""
    comp = {"Vitrinita": 10, "Inertinita": 5, "Liptinita": 80, "Fondo": 5}
    p = estimate_proximate(comp)
    assert p["VM%"] <= 100
    assert p["FC%"] >= 0


def test_classify_coal_primary_v_gt_60_vi_gt_1_5():
    """Spec matrix 4.2: V>60% y V/I>1.5 = Primario."""
    comp = {"Vitrinita": 65, "Inertinita": 40, "Liptinita": 5, "Fondo": 0}
    # V/I = 65/40 = 1.625 > 1.5, V > 60
    result = classify_coal(comp)
    assert result == "Coqueable Primario"


def test_classify_coal_secondary_v_gt_60_vi_lte_1_5():
    """Spec matrix 4.2: V>60% pero V/I<=1.5 = Mixto (no cumple Primario ni Secundario)."""
    comp = {"Vitrinita": 65, "Inertinita": 50, "Liptinita": 5, "Fondo": 0}
    # V/I = 65/50 = 1.3 <= 1.5, no cumple umbral para Primario/Secundario
    result = classify_coal(comp)
    assert result == "Mixto"
