"""Karbos AI — Copiloto de Análisis Petrográfico de Carbón.

Interfaz Streamlit para segmentación de macerales con DA-VIT.
Soporte multi-imagen con gráficos Plotly y vista agregada.
"""

import os
import urllib.request

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from config import (
    ASSUMED_ASH_PCT,
    CHECKPOINT_PATH,
    CHECKPOINT_URL,
    CLASS_NAMES,
    CLASSIFICATION_COLORS,
    CONFIDENCE_THRESHOLDS,
    DEVICE,
    MACERAL_COLORS,
    SUPPORTED_EXTENSIONS,
)
from inference import (
    compute_composition,
    confidence_statistics,
    decode_mask,
    load_model,
    predict,
    preprocess,
)
from metrics import (
    aggregate_compositions,
    aggregate_metrics,
    classify_coal,
    compute_quality_metrics,
    estimate_proximate,
)

# --- Configuración de Página ---
st.set_page_config(
    page_title="Karbos AI — Segmentación de Carbón",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Header (full width) ---
st.markdown(
    """
# 🔬 KARBOS AI
### Copiloto de Análisis Petrográfico — Segmentación de Macerales con IA
---
"""
)


# --- Descargar modelo si no existe ---
def _ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    try:
        with st.spinner("Descargando modelo DA-VIT (~50 MB)..."):
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    except Exception as e:
        st.error(f"Error descargando modelo: {e}")
        st.info("Descargue manualmente el checkpoint y colóquelo en el directorio del proyecto.")


_ensure_checkpoint()


# --- Cargar Modelo (cacheado como recurso global) ---
@st.cache_resource(show_spinner="Cargando modelo DA-VIT...")
def _load_model():
    return load_model(CHECKPOINT_PATH, DEVICE)


try:
    model = _load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error al cargar el modelo: {e}")
    st.info("Asegúrese de que el archivo `best_mIoU.pth` esté en el directorio del proyecto.")


# --- Inicializar session_state ---
if "results" not in st.session_state:
    st.session_state.results = []
if "show_all_images" not in st.session_state:
    st.session_state.show_all_images = False


# --- Limpiar resultados si se cambian los archivos (ANTES de renderizar) ---
uploaded_files = None  # Will be set in col_left


# ═══════════════════════════════════════════════════════════════
# LAYOUT DOS COLUMNAS
# ═══════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 3])

# --- COLUMNA IZQUIERDA: Upload + Slider + Info ---
with col_left:
    uploaded_files = st.file_uploader(
        "Subir imágenes de carbón (múltiples permitidas)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        help="Formatos: TIFF, PNG, JPEG. Suba múltiples imágenes del mismo briquete "
        "para estadísticas más representativas.",
    )

    transparency = st.slider("Transparencia de máscara", 0, 100, 50)

    with st.expander("ℹ️ Sobre el método"):
        st.markdown(
            "**Karbos AI** utiliza un modelo DA-VIT (Transformador de Visión "
            "con Doble Atención) para segmentar macerales en imágenes de carbón "
            "polares bajo microscopio de luz reflejada.\n\n"
            "**Macerales detectados:**\n"
            "- 🟤 Vitrinita — Componente principal del carbón\n"
            "- ⚫ Inertinita — Material inertificado\n"
            "- 🟡 Liptinita — Rico en hidrocarburos\n\n"
            "**Métricas calculadas:**\n"
            "- TRI (Termo-Reactividad-Inerte)\n"
            "- V/I y R/I (ratios de calidad)\n"
            "- Estimaciones de análisis proximate (VM%, FC%, CV)\n\n"
            "**Nota:** Los resultados son estimaciones. Para decisiones "
            "industriales, se requiere análisis en laboratorio "
            "(ASTM D2799 / ISO 7404-3)."
        )

# --- Limpiar resultados después de leer uploaded_files ---
if uploaded_files:
    current_names = {f.name for f in uploaded_files}
    st.session_state.results = [
        r for r in st.session_state.results if r["name"] in current_names
    ]
else:
    st.session_state.results = []
    st.session_state.show_all_images = False

# --- COLUMNA DERECHA: Resultados ---
with col_right:
    if uploaded_files and model_loaded:
        # --- Procesar imágenes nuevas ---
        existing_names = {r["name"] for r in st.session_state.results}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        if new_files:
            progress = st.progress(0, text="Procesando imágenes...")
            for i, file in enumerate(new_files):
                progress.progress(
                    (i + 1) / len(new_files),
                    text=f"Procesando {file.name} ({i + 1}/{len(new_files)})...",
                )
                try:
                    tensor, orig_size, orig_img = preprocess(file)
                    mask, confidence = predict(model, tensor)
                    composition = compute_composition(mask)
                    metrics = compute_quality_metrics(composition)
                    stats = confidence_statistics(confidence)
                    mask_pil = decode_mask(mask).resize(orig_size, Image.NEAREST)

                    # Thumbnail combinado (original + máscara) — preservar aspect ratio
                    thumb = orig_img.copy()
                    thumb.thumbnail((200, 200))
                    mask_thumb = mask_pil.copy()
                    mask_thumb.thumbnail((200, 200))
                    combined = Image.new("RGB", (400, 200))
                    combined.paste(thumb, (0, 0))
                    combined.paste(mask_thumb, (200, 0))

                    # Overlay
                    overlay = Image.blend(orig_img, mask_pil, transparency / 100)

                    st.session_state.results.append(
                        {
                            "name": file.name,
                            "composition": composition,
                            "metrics": metrics,
                            "confidence_stats": stats,
                            "orig_size": orig_size,
                            "orig_img": orig_img,
                            "mask_pil": mask_pil,
                            "overlay": overlay,
                            "combined_thumb": combined,
                            "_last_transparency": transparency,
                        }
                    )
                except Exception as e:
                    st.error(f"Error procesando {file.name}: {e}")
            progress.empty()
            st.success(f"✅ {len(new_files)} imagen(es) procesada(s) correctamente.")

        # Actualizar overlays si cambia la transparencia
        for r in st.session_state.results:
            if r.get("_last_transparency") != transparency:
                r["overlay"] = Image.blend(r["orig_img"], r["mask_pil"], transparency / 100)
                r["_last_transparency"] = transparency

        results = st.session_state.results
        n_images = len(results)

        # Guard contra procesamiento fallido (C1)
        if n_images == 0:
            st.warning("No se pudieron procesar las imágenes. Verifique el formato y el modelo.")
            st.stop()

        st.markdown(f"### Análisis de Muestra — {n_images} imagen(es)")

        # ═══════════════════════════════════════════════════════════
        # CALIDAD DE SEGMENTACIÓN — Siempre visible, métricas agregadas
        # ═══════════════════════════════════════════════════════════
        all_conf_stats = [r["confidence_stats"] for r in results]
        agg_conf_mean = round(sum(s["mean"] for s in all_conf_stats) / n_images * 100, 1)
        agg_conf_high = round(sum(s["high_pct"] for s in all_conf_stats) / n_images, 1)
        agg_conf_low = round(sum(s["low_pct"] for s in all_conf_stats) / n_images, 1)

        high_threshold = CONFIDENCE_THRESHOLDS["high"]
        low_threshold = CONFIDENCE_THRESHOLDS["medium"]

        st.markdown("#### Calidad de Segmentación (promedio)")
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Calidad Media", f"{agg_conf_mean}%", help="Promedio de certeza del modelo en todos los píxeles")
        qc2.metric(f"Alta Calidad (≥{int(high_threshold * 100)}%)", f"{agg_conf_high}%", help="Píxeles donde el modelo está muy seguro")
        qc3.metric(f"Baja Calidad (<{int(low_threshold * 100)}%)", f"{agg_conf_low}%", help="Zonas donde el modelo duda — humano debe revisar")

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # GALERÍA — Grid 3×2 + Modal
        # ═══════════════════════════════════════════════════════════
        VISIBLE_COLS = 3
        VISIBLE_ROWS = 2
        VISIBLE_COUNT = VISIBLE_COLS * VISIBLE_ROWS

        st.markdown(f"#### Galería ({n_images} imágenes)")

        # Grid principal: máximo 6 thumbnails
        visible_results = results[:VISIBLE_COUNT]
        cols = st.columns(VISIBLE_COLS)
        for idx, item in enumerate(visible_results):
            with cols[idx % VISIBLE_COLS]:
                st.image(item["combined_thumb"], use_container_width=True)
                st.caption(f"{item['name'][:20]}")

        # "Ver más" toggle (H4)
        if n_images > VISIBLE_COUNT:
            st.session_state.show_all_images = True
        elif n_images <= VISIBLE_COUNT:
            st.session_state.show_all_images = False

        if st.session_state.show_all_images:
            with st.expander(f"Todas las imágenes ({n_images})", expanded=False):
                cols_modal = st.columns(VISIBLE_COLS)
                for idx, item in enumerate(results):
                    with cols_modal[idx % VISIBLE_COLS]:
                        st.image(item["combined_thumb"], use_container_width=True)
                        st.caption(f"{item['name'][:20]}")

        # ═══════════════════════════════════════════════════════════
        # DETALLE POR IMAGEN (expander, usa caché)
        # ═══════════════════════════════════════════════════════════
        with st.expander(f"Detalle por imagen ({n_images} disponibles)"):
            selected_idx = st.selectbox(
                "Seleccionar imagen",
                range(n_images),
                format_func=lambda i: results[i]["name"],
                key="detail_select",
            )
            detail = results[selected_idx]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original**")
                st.image(detail["orig_img"], use_container_width=True)
            with c2:
                st.markdown("**Máscara IA**")
                st.image(detail["mask_pil"], use_container_width=True)
            with c3:
                st.markdown("**Superposición**")
                st.image(detail["overlay"], use_container_width=True)

            st.markdown("**Composición:**")
            comp = detail["composition"]
            maceral_items = [(name, pct) for name, pct in comp.items() if name != "Fondo"]
            comp_cols = st.columns(len(maceral_items))
            for col_idx, (name, pct) in enumerate(maceral_items):
                color = MACERAL_COLORS.get(name, "#808080")
                comp_cols[col_idx].markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background:{color}22; border-radius:6px; "
                    f"border-left:3px solid {color};'>"
                    f"<b style='color:{color};'>{name}</b><br>"
                    f"<span style='font-size:1.2em;'>{pct}%</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # AGREGACIONES
        # ═══════════════════════════════════════════════════════════
        all_compositions = [r["composition"] for r in results]
        all_metrics = [r["metrics"] for r in results]
        agg_comp = aggregate_compositions(all_compositions)
        agg_met = aggregate_metrics(all_metrics)

        # ═══════════════════════════════════════════════════════════
        # COMPOSICIÓN MACERAL — Donut Chart
        # ═══════════════════════════════════════════════════════════
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### Composición Maceral Promedio")
            maceral_names = [m for m in CLASS_NAMES if m != "Fondo"]
            means = [agg_comp[m]["mean"] for m in maceral_names]
            stds = [agg_comp[m]["std"] for m in maceral_names]
            colors = [MACERAL_COLORS[m] for m in maceral_names]
            text = [f"{m:.1f}% ± {s:.1f}%" for m, s in zip(means, stds)]

            fig = px.pie(
                names=maceral_names,
                values=means,
                color=maceral_names,
                color_discrete_map=dict(zip(maceral_names, colors)),
                hole=0.4,
            )
            fig.update_traces(
                textinfo="label+percent",
                text=text,
                textfont_size=13,
            )
            fig.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("¿Qué son los macerales?"):
                st.markdown(
                    "- **Vitrinita** — Principal componente del carbón. "
                    "Responsable de la capacidad coqueable.\n"
                    "- **Inertinita** — Material inertificado por oxidación. "
                    "Aporta rigidez al coque.\n"
                    "- **Liptinita** — Rico en hidrocarburos. "
                    "Aporta volátiles y poder calorífico.\n"
                    "- **Fondo** — Mineral de ganga o partes no identificadas."
                )

        # ═══════════════════════════════════════════════════════════
        # MÉTRICAS DE CALIDAD — Bar Chart Horizontal
        # ═══════════════════════════════════════════════════════════
        with c_right:
            st.markdown("#### Métricas de Calidad Promedio")

            metric_names = ["TRI", "V/I", "R/I"]
            metric_means = [agg_met[k]["mean"] for k in metric_names]
            metric_stds = [agg_met[k]["std"] for k in metric_names]

            finite_mask = [m != float("inf") for m in metric_means]
            finite_names = [n for n, f in zip(metric_names, finite_mask) if f]
            finite_means = [m for m, f in zip(metric_means, finite_mask) if f]
            finite_stds = [s for s, f in zip(metric_stds, finite_mask) if f]

            if finite_names:
                bar_colors = ["#FF6B35", "#22C55E", "#3B82F6"][:len(finite_names)]
                error_minus = [min(m, s) for m, s in zip(finite_means, finite_stds)]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=finite_names,
                    x=finite_means,
                    error_x=dict(
                        type="data",
                        array=finite_stds,
                        arrayminus=error_minus,
                        thickness=2,
                    ),
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{m:.1f}" for m in finite_means],
                    textposition="auto",
                    textfont=dict(size=14, color="white"),
                    hovertext=[f"{n}: {m:.1f} ± {s:.1f}" for n, m, s in zip(finite_names, finite_means, finite_stds)],
                    hoverinfo="text",
                ))
                fig.update_layout(
                    margin=dict(t=20, b=20, l=80, r=40),
                    height=max(140, len(finite_names) * 60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.1)",
                        title="Valor",
                    ),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Todas las métricas son indefinidas (Inertinita = 0).")

            # Reactivos / Inertes
            r_met = agg_met["%Reactivos"]
            i_met = agg_met["%Inertes"]
            rc1, rc2 = st.columns(2)
            rc1.metric(
                "Reactivos",
                f"{r_met['mean']}%",
                delta=f"± {r_met['std']}%" if r_met["std"] > 0 else None,
            )
            rc2.metric(
                "Inertes",
                f"{i_met['mean']}%",
                delta=f"± {i_met['std']}%" if i_met["std"] > 0 else None,
            )

            with st.expander("¿Cómo se calculan?"):
                st.markdown("**TRI — Termo-Reactividad-Inerte**")
                st.latex(r"TRI = V + 0.5 \times L")
                st.markdown(
                    "Fracción reactiva para coqueificación. "
                    "Mayor valor = mejor calidad para coque."
                )
                st.markdown("**V/I — Vitrinita / Inertinita**")
                st.latex(r"V/I = \frac{V}{I}")
                st.markdown("> 1.5 = coqueable primario, > 1.5 = secundario.")
                st.markdown("**R/I — Reactivos / Inertes**")
                st.latex(r"R/I = \frac{V + L}{I + BG}")
                st.markdown("Capacidad general de reactividad.")

        # ═══════════════════════════════════════════════════════════
        # CLASIFICACIÓN — Pie Chart o Badge
        # ═══════════════════════════════════════════════════════════
        c_left2, c_right2 = st.columns(2)

        with c_left2:
            st.markdown("#### Clasificación por Consensus")
            classifications = [classify_coal(c) for c in all_compositions]
            class_counts = {}
            for cls_name in classifications:
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            if len(class_counts) == 1:
                class_name = list(class_counts.keys())[0]
                class_color = CLASSIFICATION_COLORS.get(class_name, "#808080")
                st.markdown(
                    f"<div style='text-align:center; padding:24px; "
                    f"background:{class_color}22; border-radius:12px; "
                    f"border:2px solid {class_color}; margin:10px 0;'>"
                    f"<span style='font-size:1.6em; color:{class_color};'>"
                    f"**{class_name}**</span><br>"
                    f"<small style='color:#aaa;'>100% — {n_images}/{n_images} imágenes</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                fig = px.pie(
                    names=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    color=list(class_counts.keys()),
                    color_discrete_map=CLASSIFICATION_COLORS,
                    hole=0.3,
                )
                fig.update_traces(
                    textinfo="label+percent",
                    textfont_size=13,
                    pull=[0.05 if v == max(class_counts.values()) else 0
                          for v in class_counts.values()],
                )
                fig.update_layout(
                    showlegend=True,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5,
                        font=dict(size=11),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver criterios de clasificación"):
                st.markdown(
                    "| Tipo | Condición | Uso |\n"
                    "|------|-----------|-----|\n"
                    "| **Coqueable Primario** | V > 60% y V/I > 1.5 | Coque de alto horno |\n"
                    "| **Coqueable Secundario** | V > 50% y V/I > 1.5 | Coque con mezcla |\n"
                    "| **Rico en Liptinita** | L > 20% | Carbón energético |\n"
                    "| **Térmico** | I > 50% | Generación eléctrica |\n"
                    "| **Mixto** | Resto | Uso combinado |"
                )

        # ═══════════════════════════════════════════════════════════
        # ESTIMACIONES PROXIMATE — Bar Chart Vertical
        # ═══════════════════════════════════════════════════════════
        with c_right2:
            st.markdown("#### Estimaciones de Análisis Proximate Promedio")
            all_proximate = [estimate_proximate(c) for c in all_compositions]
            agg_vm = round(sum(p["VM%"] for p in all_proximate) / n_images, 1)
            agg_fc = round(sum(p["FC%"] for p in all_proximate) / n_images, 1)
            agg_cv = round(sum(p["CV (kcal/kg)"] for p in all_proximate) / n_images, 0)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Materia Volátil", "Carbono Fijo", "Cenizas*"],
                y=[agg_vm, agg_fc, ASSUMED_ASH_PCT],
                marker_color=["#FF6B35", "#22C55E", "#808080"],
                text=[f"{agg_vm}%", f"{agg_fc}%", f"{ASSUMED_ASH_PCT}%*"],
                textposition="auto",
                textfont=dict(size=13, color="white"),
            ))
            fig.update_layout(
                margin=dict(t=10, b=10, l=30, r=30),
                height=250,
                yaxis_title="%",
                yaxis=dict(range=[0, 100]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Calorías (promedio)", f"{agg_cv:.0f} kcal/kg")

            st.warning(
                f"⚠️ Estimaciones basadas en correlaciones de literatura "
                f"(error ±15-20%). Cenizas asumidas {ASSUMED_ASH_PCT}%. Para decisiones "
                f"industriales, se requiere análisis en laboratorio."
            )

            with st.expander("¿Cómo se estiman?"):
                st.markdown("**VM% — Materia Volátil**")
                st.latex(r"VM = 0.8V + 1.2L + 0.5I")
                st.markdown("Se evapora al calentar. Indica inflamabilidad.")
                st.markdown("**FC% — Carbono Fijo**")
                st.latex(r"FC = 100 - VM - Cenizas")
                st.markdown("Combustible sólido permanente.")
                st.markdown("**CV — Calorías (kcal/kg)**")
                st.latex(r"CV = 8000 + 40V + 60L")
                st.markdown("Energía liberada al quemar.")

        # ═══════════════════════════════════════════════════════════
        # DISTRIBUCIÓN POR IMAGEN — Line Chart con Markers
        # ═══════════════════════════════════════════════════════════
        if n_images > 1:
            st.markdown("#### Distribución por Imagen")
            rows = []
            for idx, item in enumerate(results):
                for maceral in CLASS_NAMES:
                    if maceral != "Fondo":
                        rows.append({
                            "Imagen": idx + 1,
                            "Maceral": maceral,
                            "Porcentaje": item["composition"][maceral],
                            "Nombre": item["name"][:20],
                        })
            df = pd.DataFrame(rows)

            fig = px.line(
                df,
                x="Imagen",
                y="Porcentaje",
                color="Maceral",
                color_discrete_map=MACERAL_COLORS,
                markers=True,
                hover_data={"Nombre": True, "Porcentaje": ":.1f", "Imagen": False},
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=40, r=30),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title=None,
                    dtick=1,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    title="% Maceral",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.1)",
                    range=[0, 100],
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.35,
                    xanchor="center", x=0.5,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════
        # DISCLAIMER + MÉTODO
        # ═══════════════════════════════════════════════════════════
        st.warning(
            f"📋 **Nota:** Resultados de **{n_images} imagen(es)** — "
            "un campo de visión por imagen. El método de referencia "
            "(ASTM D2799 / ISO 7404-3) requiere **500-1,000 puntos por "
            "muestra** en múltiples campos de visión para validez estadística."
        )

        with st.expander("¿Cómo funciona el método de referencia?"):
            st.markdown(
                "**ASTM D2799 / ISO 7404-3 — Análisis petrográfico por punto**\n\n"
                "1. Se prepara una sección pulida de carbón (briquete con resina).\n"
                "2. Se examina bajo microscopio de luz reflejada a **500x**.\n"
                "3. Se cuenta **1 punto cada 0.5 mm** moviendo la etapa.\n"
                "4. Se registran **500-1,000 puntos** por muestra.\n"
                "5. Se calculan los % de macerales y ratios de calidad.\n\n"
                "**Tiempo real:** 4-8 horas por muestra (manual).\n\n"
            )

    elif uploaded_files and not model_loaded:
        st.error("No se puede procesar: el modelo no se cargó correctamente.")

    else:
        st.info(
            "👈 Sube una o más imágenes de carbón para iniciar el análisis. "
            "Subir múltiples imágenes del mismo briquete mejora la "
            "representatividad estadística."
        )

# --- Footer (full width) ---
st.markdown("---")
st.markdown("Desarrollado para Karbos AI © 2026")
