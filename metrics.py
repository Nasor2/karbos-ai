"""Simplified metrics module - public interface only.

This module provides metrics calculation without exposing
the internal formulas and business logic.
"""

from config import ASSUMED_ASH_PCT


def compute_quality_metrics(composition: dict) -> dict:
    """Calculate coal quality metrics from maceral composition.

    Args:
        composition: Dictionary with Vitrinita, Inertinita, Liptinita, Fondo percentages.

    Returns:
        Dictionary with TRI, V/I, R/I, %Reactivos, %Inertes.
    """
    v = composition.get("Vitrinita", 0)
    i = composition.get("Inertinita", 0)
    liptinite = composition.get("Liptinita", 0)
    bg = composition.get("Fondo", 0)

    inertes = i + bg
    reactivos = v + liptinite

    tri = round(v + 0.5 * liptinite, 1)
    vi_ratio = round(v / i, 2) if i > 0 else float("inf")
    ri_ratio = round(reactivos / inertes, 2) if inertes > 0 else float("inf")

    return {
        "TRI": tri,
        "V/I": vi_ratio,
        "R/I": ri_ratio,
        "%Reactivos": round(reactivos, 1),
        "%Inertes": round(inertes, 1),
    }


def classify_coal(composition: dict) -> str:
    """Classify coal based on maceral composition.

    Args:
        composition: Dictionary with maceral percentages.

    Returns:
        Classification name.
    """
    v = composition.get("Vitrinita", 0)
    i = composition.get("Inertinita", 0)
    liptinite = composition.get("Liptinita", 0)

    vi_ratio = v / i if i > 0 else float("inf")

    if v > 60 and vi_ratio > 1.5:
        return "Coqueable Primario"
    elif v > 50 and vi_ratio > 1.5:
        return "Coqueable Secundario"
    elif liptinite > 20:
        return "Rico en Liptinita"
    elif i > 50:
        return "Térmico"
    else:
        return "Mixto"


def estimate_proximate(composition: dict) -> dict:
    """Estimate proximate analysis from maceral composition.

    Args:
        composition: Dictionary with maceral percentages.

    Returns:
        Dictionary with VM%, FC%, Cenizas%, CV (kcal/kg).
    """
    v = composition.get("Vitrinita", 0)
    liptinite = composition.get("Liptinita", 0)
    i = composition.get("Inertinita", 0)

    vm = round(0.8 * v + 1.2 * liptinite + 0.5 * i, 1)
    vm = min(vm, 100)
    fc = round(100 - vm - ASSUMED_ASH_PCT, 1)
    cv = round(8000 + 40 * v + 60 * liptinite, 0)

    return {
        "VM%": vm,
        "FC%": max(fc, 0),
        "Cenizas%": ASSUMED_ASH_PCT,
        "CV (kcal/kg)": cv,
    }


def aggregate_compositions(compositions: list) -> dict:
    """Aggregate compositions from multiple images (mean + std).

    Args:
        compositions: List of composition dictionaries.

    Returns:
        Dict with 'mean' and 'std' for each maceral.
    """
    if not compositions:
        return {}
    keys = compositions[0].keys()
    result = {}
    for key in keys:
        values = [c[key] for c in compositions]
        mean = round(sum(values) / len(values), 1)
        std = round((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5, 1)
        result[key] = {"mean": mean, "std": std}
    return result


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate quality metrics from multiple images.

    Args:
        metrics_list: List of metrics dictionaries.

    Returns:
        Dict with 'mean' and 'std' for each metric.
    """
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if m[key] != float("inf")]
        if not values:
            result[key] = {"mean": float("inf"), "std": 0}
            continue
        mean = round(sum(values) / len(values), 2)
        std = round((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5, 2)
        result[key] = {"mean": mean, "std": std}
    return result
