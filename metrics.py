"""Métricas de calidad del carbón derivada de composición maceral.

TRI, V/I, R/I, estimaciones proximate, clasificación industrial y agregación multi-imagen.
Basado en: 04-CALIDAD-CARBON.md
"""

from config import ASSUMED_ASH_PCT


def compute_quality_metrics(composition: dict[str, float]) -> dict[str, float]:
    """Calcula métricas de calidad del carbón desde composición maceral.

    Args:
        composition: Diccionario {Vitrinita: %, Inertinita: %, Liptinita: %, Fondo: %}.

    Returns:
        Diccionario con TRI, V/I, R/I, %Reactivos, %Inertes.
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


def classify_coal(composition: dict[str, float]) -> str:
    """Clasifica el carbón según composición maceral (spec matrix 4.2).

    Matriz de decisión:
                    %V > 60    %V 50-60    %V < 50
    V/I > 1.5       Primario    Secundario  (ver abajo)
    V/I ≤ 1.5       Mixto       Mixto       Térmico (si I>50)

    Args:
        composition: Diccionario con porcentajes de macerales.

    Returns:
        Nombre de la clase de clasificación.
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


def estimate_proximate(composition: dict[str, float]) -> dict[str, float]:
    """Estima propiedades proximate desde composición maceral.

    Nota: Son estimaciones, no sustituyen análisis de laboratorio.

    Args:
        composition: Diccionario con porcentajes de macerales.

    Returns:
        Diccionario con VM%, FC%, Cenizas%, CV (kcal/kg).
    """
    v = composition.get("Vitrinita", 0)
    liptinite = composition.get("Liptinita", 0)
    i = composition.get("Inertinita", 0)

    vm = round(0.8 * v + 1.2 * liptinite + 0.5 * i, 1)
    vm = min(vm, 100)  # Cap at 100% to prevent physical impossibility
    fc = round(100 - vm - ASSUMED_ASH_PCT, 1)
    cv = round(8000 + 40 * v + 60 * liptinite, 0)

    return {
        "VM%": vm,
        "FC%": max(fc, 0),
        "Cenizas%": ASSUMED_ASH_PCT,
        "CV (kcal/kg)": cv,
    }


# --- Agregación multi-imagen ---


def aggregate_compositions(
    compositions: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Agrega composiciones de múltiples imágenes (promedio + desviación).

    Args:
        compositions: Lista de diccionarios de composición.

    Returns:
        Dict con 'mean' y 'std' para cada maceral.
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


def aggregate_metrics(
    metrics_list: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Agrega métricas de calidad de múltiples imágenes.

    Args:
        metrics_list: Lista de diccionarios de métricas.

    Returns:
        Dict con 'mean' y 'std' para cada métrica.
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
