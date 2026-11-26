import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from escudos_helper import (
    agregar_columna_escudo,
    config_columna_escudo,
    img_escudo_html,
)
from modelo_predictivo import (
    adaptar_df_enciclo_a_motor,
    calcular_elo_dinamico,
    calcular_tabla,
    estimar_rho_mle,
    obtener_plantel_y_xi_default,
    proyectar_jugadores_partido,
    simular_partido_completo,
)

RED = "#ED1A3B"
BLUE = "#3b82f6"
GOLD = "#cfb45e"
GRAY = "#4a4a52"
AXIS_TXT = "#8b8b95"

PLOT_LAYOUT = dict(
    font=dict(family="Manrope, sans-serif", size=12, color="#a0a0a8"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=25, b=10),
    hoverlabel=dict(bgcolor="#17171c", bordercolor=RED, font=dict(family="Manrope", size=12, color="#ffffff")),
)


@st.cache_data(show_spinner=False)
def _preparar_motor(df_eq: pd.DataFrame, df_jug: pd.DataFrame):
    df_largo = adaptar_df_enciclo_a_motor(df_eq, df_jug)
    elos = calcular_elo_dinamico(df_largo)
    tabla = calcular_tabla(df_largo, elos)
    rho = estimar_rho_mle(df_largo)
    return df_largo, elos, tabla, rho


def _resumen_tactico_xi(titulares: list[str] | None, plantel: pd.DataFrame) -> tuple[str, list[str]]:
    """Valida que la formación tenga equilibrio táctico real sin imponer un esquema fijo."""
    if not titulares or plantel.empty:
        return "", []

    pos_map = plantel.set_index("jugador")["Posicion"].to_dict() if "Posicion" in plantel.columns else {}
    conteo = {"ARQ": 0, "DEF": 0, "MED": 0, "DEL": 0}
    for j in titulares:
        p = str(pos_map.get(j, "")).lower()
        if any(k in p for k in ("arquero", "goalkeeper", "gk")):
            conteo["ARQ"] += 1
        elif any(k in p for k in ("defensor", "defender", "def")):
            conteo["DEF"] += 1
        elif any(k in p for k in ("mediocampista", "midfielder", "med")):
            conteo["MED"] += 1
        elif any(k in p for k in ("delantero", "forward", "del")):
            conteo["DEL"] += 1

    alertas = []
    if conteo["ARQ"] == 0:
        alertas.append("sin arquero")
    elif conteo["ARQ"] > 1:
        alertas.append(f"{conteo['ARQ']} arqueros")

    n = len(titulares)
    if n == 11 and conteo["ARQ"] == 1:
        if conteo["DEF"] < 3 or conteo["DEF"] > 5:
            alertas.append(f"{conteo['DEF']} defensores (permitido: 3 a 5)")
        if conteo["MED"] < 2 or conteo["MED"] > 6:
            alertas.append(f"{conteo['MED']} mediocampistas (permitido: 2 a 6)")
        if conteo["DEL"] < 1 or conteo["DEL"] > 4:
            alertas.append(f"{conteo['DEL']} delanteros (permitido: 1 a 4)")

        esquema = f"{conteo['DEF']}-{conteo['MED']}-{conteo['DEL']}"
        return f"Esquema {esquema}", alertas

    detalle = f"{conteo['ARQ']} ARQ · {conteo['DEF']} DEF · {conteo['MED']} MED · {conteo['DEL']} DEL"
    return detalle, alertas


def _badge_impacto_xi(mod_xi: dict, titulares: list[str] | None = None, plantel: pd.DataFrame | None = None) -> str:
    detalle_tac, alertas_tac = _resumen_tactico_xi(titulares, plantel if plantel is not None else pd.DataFrame())
    tac_txt = f" · <span style='color:#ececf0;'>{detalle_tac}</span>" if detalle_tac else ""
    alerta_txt = (
        f" <span style='color:#f59e0b; font-weight:700;'>(⚠️ {', '.join(alertas_tac)})</span>"
        if alertas_tac
        else ""
    )

    if not mod_xi.get("activo"):
        n = mod_xi.get("n_sel", 0)
        return (
            f"<span style='color:#f59e0b; font-size:0.78rem;'>"
            f"⚠️ {n}/11 elegidos (completá 11 para aplicar ajuste){tac_txt}{alerta_txt}</span>"
        )

    d_atk = (mod_xi["mult_atk"] - 1.0) * 100
    d_def = (mod_xi["mult_def"] - 1.0) * 100
    c_atk = "#10b981" if d_atk >= 0 else "#ef4444"
    c_def = "#10b981" if d_def >= 0 else "#ef4444"
    return (
        f"<span style='color:#a0a0a8; font-size:0.78rem;'>"
        f"✅ <b>11/11</b>{tac_txt}{alerta_txt} · Ataque: <b style='color:{c_atk}'>{d_atk:+.1f}%</b> · "
        f"Defensa: <b style='color:{c_def}'>{d_def:+.1f}%</b></span>"
    )


def _generar_resumen_simulacion(
    eq_local: str,
    eq_vis: str,
    sim: dict,
    df_proy_jug: pd.DataFrame,
    neutral: bool = False,
) -> str:
    top_m = sim["top_marcadores"].iloc[0]
    marc_txt = f"{top_m['Marcador']} ({float(top_m['Probabilidad']):.1f}%)"
    cond_txt = " (Cancha neutral)" if neutral else ""

    lineas = [
        f"🔮 Simulación LPF: {eq_local} vs {eq_vis}{cond_txt}",
        "",
        f"🏠 {eq_local}: {sim['p_loc']:.1f}%",
        f"🤝 Empate: {sim['p_emp']:.1f}%",
        f"✈️ {eq_vis}: {sim['p_vis']:.1f}%",
        "",
        f"📊 Goles esperados (λ): {sim['lam_loc']:.2f} - {sim['lam_vis']:.2f}",
        f"🎯 Marcador más probable: {marc_txt}",
        f"🧤 Valla invicta: {eq_local} {sim['valla_loc']:.0f}% · {eq_vis} {sim['valla_vis']:.0f}%",
        f"⚽ Ambos marcan: {sim['ambos_marcan']:.1f}% · +2.5 goles: {sim['over_25']:.1f}%",
    ]

    if not df_proy_jug.empty:
        top_g = df_proy_jug.sort_values("Prob_Gol", ascending=False).head(3)
        figs = [f"{r['Jugador']} ({r['Prob_Gol']:.0f}%)" for _, r in top_g.iterrows()]
        if figs:
            lineas.append(f"🔥 Mayor prob. de gol: {' · '.join(figs)}")

    lineas.extend(["", "📲 Simulado en Enciclo LPF · enciclolpf.streamlit.app"])
    return "\n".join(lineas)


def render_simulador(df_eq: pd.DataFrame, df_jug: pd.DataFrame, periodo: str = "Temporada Completa"):
    if df_eq.empty:
        st.warning("No hay datos suficientes para ejecutar el modelo predictivo.")
        return

    equipos = sorted(df_eq["equipo"].dropna().unique())
    if len(equipos) < 2:
        st.warning("Se necesitan al menos 2 equipos en el período seleccionado.")
        return

    with st.expander("📖 Guía de uso y aclaraciones del simulador"):
        st.markdown("""
        * **Es un modelo probabilístico, no una bola de cristal:** El simulador estima escenarios posibles a partir del rendimiento histórico (`xG`, goles, fuerza del rival, localía y Elo dinámico). Un 65% de victoria significa que, jugando ese partido 100 veces en condiciones similares, ese equipo ganaría ~65 veces, pero el fútbol siempre conserva margen para la sorpresa.
        * **Cómo leer `λ` (Goles Esperados del cruce):** Representa el promedio de goles proyectado para cada equipo en este enfrentamiento específico. A partir de `λ` se calcula toda la matriz de marcadores exactos (`0-0`, `1-0`, `2-1`, etc.) y las probabilidades `1X2`.
        * **11 Titular:** Por defecto carga la última formación utilizada por cada club. Si cambiás nombres (por lesiones, suspensiones o rotación), el motor recalcula automáticamente el peso ofensivo y defensivo del equipo.
        * **Penalización extra (%):** Es un **ajuste manual de contexto**. Usalo cuando un equipo llega disminuido por factores que los datos previos no ven (rotación masiva por copa internacional, bajas sensibles de último momento, desgaste de viaje o contexto adverso). Subir la barra reduce su expectativa de gol y aumenta el del rival.
        * **Peso forma reciente:** Controla cuánto pesan las últimas fechas frente al rendimiento acumulado del año (*Agresivo* reacciona rápido a rachas; *Conservador* prioriza la muestra larga).
        """)

    # ── Selectores del Cruce 1 vs 1 con Escudos ──
    c1, c_mid, c2 = st.columns([2.2, 0.45, 2.2])
    idx_riv = equipos.index("River Plate") if "River Plate" in equipos else 0
    idx_vis = 1 if idx_riv == 0 else 0

    with c1:
        col_e1, col_s1 = st.columns([0.32, 2.2])
        with col_s1:
            eq_local = st.selectbox("🏠 Equipo Local", equipos, index=idx_riv, key="sim_eq_loc")
        with col_e1:
            st.markdown(f"<div style='padding-top:26px; text-align:center;'>{img_escudo_html(eq_local, 42)}</div>", unsafe_allow_html=True)

    with c_mid:
        st.markdown("<div style='text-align:center; padding-top:32px; font-family:Oswald,sans-serif; font-size:1.3rem; color:#8b8b95;'>VS</div>", unsafe_allow_html=True)

    with c2:
        col_s2, col_e2 = st.columns([2.2, 0.32])
        with col_s2:
            eq_vis = st.selectbox("✈️ Equipo Visitante", equipos, index=idx_vis, key="sim_eq_vis")
        with col_e2:
            st.markdown(f"<div style='padding-top:26px; text-align:center;'>{img_escudo_html(eq_vis, 42)}</div>", unsafe_allow_html=True)

    if eq_local == eq_vis:
        st.info("Seleccioná dos equipos distintos para simular el enfrentamiento.")
        return

    # ── Selector de 11 Titular por Equipo ──
    plantel_loc, xi_def_loc = obtener_plantel_y_xi_default(df_jug, eq_local)
    plantel_vis, xi_def_vis = obtener_plantel_y_xi_default(df_jug, eq_vis)

    titulares_loc = None
    titulares_vis = None

    with st.expander("📋 Armar 11 Titular de cada equipo (recalcula λ, probabilidades y figuras)", expanded=True):
        col_xi1, col_xi2 = st.columns(2)

        with col_xi1:
            if not plantel_loc.empty:
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'>"
                    f"{img_escudo_html(eq_local, 24)}<b>11 Titular · {eq_local}</b></div>",
                    unsafe_allow_html=True,
                )
                opciones_loc = plantel_loc["jugador"].tolist()
                etiquetas_loc = {
                    r["jugador"]: f"{r['jugador']} ({r['Posicion'][:3].upper()} · {r['Nota']:.2f})"
                    for _, r in plantel_loc.iterrows()
                }
                key_loc = f"xi_sel_loc_{eq_local}_{periodo}"
                if key_loc not in st.session_state:
                    st.session_state[key_loc] = xi_def_loc

                if st.button("🔄 Último 11", key=f"rst_loc_{eq_local}_{periodo}"):
                    st.session_state[key_loc] = xi_def_loc

                titulares_loc = st.multiselect(
                    f"Titulares {eq_local}",
                    options=opciones_loc,
                    format_func=lambda j: etiquetas_loc.get(j, j),
                    max_selections=11,
                    key=key_loc,
                    label_visibility="collapsed",
                )
            else:
                st.info(f"Sin plantel individual registrado para {eq_local}.")

        with col_xi2:
            if not plantel_vis.empty:
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'>"
                    f"{img_escudo_html(eq_vis, 24)}<b>11 Titular · {eq_vis}</b></div>",
                    unsafe_allow_html=True,
                )
                opciones_vis = plantel_vis["jugador"].tolist()
                etiquetas_vis = {
                    r["jugador"]: f"{r['jugador']} ({r['Posicion'][:3].upper()} · {r['Nota']:.2f})"
                    for _, r in plantel_vis.iterrows()
                }
                key_vis = f"xi_sel_vis_{eq_vis}_{periodo}"
                if key_vis not in st.session_state:
                    st.session_state[key_vis] = xi_def_vis

                if st.button("🔄 Último 11", key=f"rst_vis_{eq_vis}_{periodo}"):
                    st.session_state[key_vis] = xi_def_vis

                titulares_vis = st.multiselect(
                    f"Titulares {eq_vis}",
                    options=opciones_vis,
                    format_func=lambda j: etiquetas_vis.get(j, j),
                    max_selections=11,
                    key=key_vis,
                    label_visibility="collapsed",
                )
            else:
                st.info(f"Sin plantel individual registrado para {eq_vis}.")

    # ── Ajustes avanzados de contexto ──
    with st.expander("⚙️ Ajustes adicionales de contexto (cancha neutral, penalización extra, half-life)"):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            neutral = st.checkbox(
                "Cancha neutral (Playoffs)",
                value=False,
                help="Elimina la ventaja de localía en el cálculo de goles esperados.",
            )
        with a2:
            pen_loc = st.slider(
                f"Penalización extra {eq_local[:14]} (%)",
                0, 45, 0, step=5,
                help="Castigo contextual manual (bajas sensibles, rotación por copa, desgaste o contexto adverso). Reduce el λ propio y aumenta el del rival.",
            )
        with a3:
            pen_vis = st.slider(
                f"Penalización extra {eq_vis[:14]} (%)",
                0, 45, 0, step=5,
                help="Castigo contextual manual (bajas sensibles, rotación por copa, desgaste o contexto adverso). Reduce el λ propio y aumenta el del rival.",
            )
        with a4:
            peso_forma = st.select_slider(
                "Peso forma reciente",
                options=["Conservador", "Calibrado (4 fechas)", "Agresivo"],
                value="Calibrado (4 fechas)",
                help="Define qué tan rápido reacciona el modelo a las últimas fechas respecto al historial del torneo.",
            )
            hl_map = {"Conservador": 8.0, "Calibrado (4 fechas)": 4.0, "Agresivo": 2.2}

    # ── Validación de equilibrio táctico por líneas ──
    errores_tacticos = []
    if not plantel_loc.empty and titulares_loc is not None:
        _, al_loc = _resumen_tactico_xi(titulares_loc, plantel_loc)
        if al_loc:
            errores_tacticos.append(f"**{eq_local}**: {', '.join(al_loc)}")
    if not plantel_vis.empty and titulares_vis is not None:
        _, al_vis = _resumen_tactico_xi(titulares_vis, plantel_vis)
        if al_vis:
            errores_tacticos.append(f"**{eq_vis}**: {', '.join(al_vis)}")

    if errores_tacticos:
        st.error(f"📋 Formación inválida para simular ({' · '.join(errores_tacticos)}).")
        return

    df_largo, elos, tabla, rho = _preparar_motor(df_eq, df_jug)
    sim = simular_partido_completo(
        df_largo=df_largo,
        elos=elos,
        tabla=tabla,
        rho_dinamico=rho,
        eq_local=eq_local,
        eq_vis=eq_vis,
        aplicar_localia=(not neutral),
        penalizacion_loc=pen_loc,
        penalizacion_vis=pen_vis,
        half_life=hl_map[peso_forma],
        df_jug=df_jug,
        titulares_loc=titulares_loc,
        titulares_vis=titulares_vis,
    )

    # Mostrar impacto del 11 elegido + estructura táctica en tiempo real
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(_badge_impacto_xi(sim["mod_xi_loc"], titulares_loc, plantel_loc), unsafe_allow_html=True)
    with ic2:
        st.markdown(_badge_impacto_xi(sim["mod_xi_vis"], titulares_vis, plantel_vis), unsafe_allow_html=True)

    esc_loc_html = img_escudo_html(eq_local, 34)
    esc_vis_html = img_escudo_html(eq_vis, 34)

    # ── 1. TARJETAS KPI PRINCIPALES ──
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label" style="display:flex; align-items:center; justify-content:center; gap:8px;">
                {esc_loc_html}<span>Victoria {eq_local}</span>
            </div>
            <div class="kpi-value" style="color:{RED}">{sim['p_loc']:.1f}%</div>
            <div class="kpi-sub">λ: <b>{sim['lam_loc']:.2f}</b> · Elo L: {sim['elo_loc']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Empate</div>
            <div class="kpi-value" style="color:{GOLD}">{sim['p_emp']:.1f}%</div>
            <div class="kpi-sub">Total goles esp.: <b>{(sim['lam_loc'] + sim['lam_vis']):.2f}</b></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label" style="display:flex; align-items:center; justify-content:center; gap:8px;">
                {esc_vis_html}<span>Victoria {eq_vis}</span>
            </div>
            <div class="kpi-value" style="color:{BLUE}">{sim['p_vis']:.1f}%</div>
            <div class="kpi-sub">λ: <b>{sim['lam_vis']:.2f}</b> · Elo V: {sim['elo_vis']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Vallas Invictas</div>
            <div class="kpi-value" style="font-size:1.65rem">{sim['valla_loc']:.0f}% <span style="color:#8b8b95; font-size:1.1rem">vs</span> {sim['valla_vis']:.0f}%</div>
            <div class="kpi-sub">ρ Dixon-Coles: <b>{sim['rho']:+.3f}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Barra visual de probabilidad 1X2 ──
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=["1X2"], x=[sim["p_loc"]], name=f"{eq_local} ({sim['p_loc']}%)",
        orientation="h", marker=dict(color=RED), text=f"{eq_local}: {sim['p_loc']}%", textposition="inside",
        textfont=dict(color="#ffffff", size=13, family="Oswald"),
    ))
    fig_bar.add_trace(go.Bar(
        y=["1X2"], x=[sim["p_emp"]], name=f"Empate ({sim['p_emp']}%)",
        orientation="h", marker=dict(color="#2c2c35"), text=f"Empate: {sim['p_emp']}%", textposition="inside",
        textfont=dict(color="#ececf0", size=13, family="Oswald"),
    ))
    fig_bar.add_trace(go.Bar(
        y=["1X2"], x=[sim["p_vis"]], name=f"{eq_vis} ({sim['p_vis']}%)",
        orientation="h", marker=dict(color=BLUE), text=f"{eq_vis}: {sim['p_vis']}%", textposition="inside",
        textfont=dict(color="#ffffff", size=13, family="Oswald"),
    ))
    fig_bar.update_layout(
        barmode="stack",
        height=72,
        margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width="stretch")

    # ── 2. TENDENCIAS, MARCADORES EXACTOS Y MATRIZ ──
    col_m1, col_m2, col_m3 = st.columns([1.15, 1.1, 1.35])

    with col_m1:
        st.markdown('<div class="section-header">Tendencias del Partido</div>', unsafe_allow_html=True)
        df_tend = pd.DataFrame([
            {"Tendencia": f"Valla invicta · {eq_local}", "Probabilidad": f"{sim['valla_loc']:.1f}%"},
            {"Tendencia": f"Valla invicta · {eq_vis}", "Probabilidad": f"{sim['valla_vis']:.1f}%"},
            {"Tendencia": "Ambos equipos marcan (Sí)", "Probabilidad": f"{sim['ambos_marcan']:.1f}%"},
            {"Tendencia": "Más de 1.5 goles", "Probabilidad": f"{sim['over_15']:.1f}%"},
            {"Tendencia": "Más de 2.5 goles", "Probabilidad": f"{sim['over_25']:.1f}%"},
            {"Tendencia": "Menos de 2.5 goles", "Probabilidad": f"{(100 - sim['over_25']):.1f}%"},
        ])
        st.dataframe(
            df_tend.style.set_properties(subset=["Probabilidad"], **{"color": GOLD, "font-weight": "bold"}),
            hide_index=True,
            width="stretch",
            height=252,
        )

    with col_m2:
        st.markdown('<div class="section-header">Top 5 Marcadores Exactos</div>', unsafe_allow_html=True)
        top_m = sim["top_marcadores"][["Marcador", "Probabilidad"]].copy()
        top_m["Probabilidad"] = top_m["Probabilidad"].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            top_m.style.set_properties(subset=["Marcador"], **{"font-weight": "bold", "color": "#ffffff"})
            .set_properties(subset=["Probabilidad"], **{"color": RED, "font-weight": "bold"}),
            hide_index=True,
            width="stretch",
            height=220,
        )

    with col_m3:
        st.markdown('<div class="section-header">Matriz de Resultados (%)</div>', unsafe_allow_html=True)
        z = sim["matriz"]
        fig_hm = go.Figure(go.Heatmap(
            z=z,
            x=[f"{eq_vis[:10]} {j}" for j in range(6)],
            y=[f"{eq_local[:10]} {i}" for i in range(6)],
            text=np.round(z, 1),
            texttemplate="%{text}%",
            textfont=dict(size=10.5, color="#ffffff"),
            colorscale=[[0, "#121216"], [0.5, "#7f1d1d"], [1, RED]],
            showscale=False,
            hovertemplate="Local %{y} - Visitante %{x}<br>Probabilidad: %{z:.1f}%<extra></extra>",
        ))
        fig_hm.update_layout(
            **PLOT_LAYOUT,
            height=265,
            yaxis=dict(autorange="reversed", tickfont=dict(size=10, color=AXIS_TXT)),
            xaxis=dict(tickfont=dict(size=10, color=AXIS_TXT)),
        )
        st.plotly_chart(fig_hm, width="stretch")

    # ── 3. PROYECCIÓN DE MÉTRICAS DEL PARTIDO Y JUGADORES ──
    df_proy_jug = proyectar_jugadores_partido(
        df_jug=df_jug,
        eq_local=eq_local,
        eq_visitante=eq_vis,
        lam_loc=sim["lam_loc"],
        lam_vis=sim["lam_vis"],
        titulares_loc=titulares_loc,
        titulares_vis=titulares_vis,
    )

    if not df_proy_jug.empty:
        convocados = set()
        if titulares_loc:
            convocados.update((eq_local, j) for j in titulares_loc)
        if titulares_vis:
            convocados.update((eq_vis, j) for j in titulares_vis)

        if convocados:
            mascara = df_proy_jug.apply(lambda r: (r["Equipo"], r["Jugador"]) in convocados, axis=1)
            df_proy_jug = df_proy_jug[mascara].copy()

    with st.expander("📲 Copiar pronóstico de la simulación para WhatsApp / X"):
        st.code(
            _generar_resumen_simulacion(eq_local, eq_vis, sim, df_proy_jug, neutral=neutral),
            language=None,
        )

    st.markdown("---")
    tab_jug_proy, tab_met_proy = st.tabs([
        "🌟 Jugadores Proyectados y Posibles Figuras",
        "📊 Proyección de Métricas del Partido",
    ])

    with tab_met_proy:
        st.dataframe(
            sim["metricas_partido"].style.set_properties(subset=["Métrica"], **{"font-weight": "bold"}),
            hide_index=True,
            width="stretch",
            height=430,
        )

    with tab_jug_proy:
        st.caption("Estimación ajustada al 11 titular confirmado y a los goles esperados (λ) de cada equipo en este cruce.")

        if df_proy_jug.empty:
            st.info("No hay suficientes registros individuales recientes para proyectar jugadores en este cruce.")
        else:
            sub_dest, sub_loc, sub_vis = st.tabs([
                "🔥 Posibles Figuras (Ambos Equipos)",
                f"🏠 Plantel {eq_local}",
                f"✈️ Plantel {eq_vis}",
            ])

            def _render_tabla_proy(df_sub: pd.DataFrame, alto: int = 410):
                show = df_sub[[
                    "Jugador", "Equipo", "Rol", "Posición", "Nota_Base",
                    "xG_Proy", "Prob_Gol", "xA_Proy", "Prob_Asist", "Prob_GA", "Indice_Figura"
                ]].copy()
                show.columns = [
                    "Jugador", "Club", "Rol", "Pos", "Nota Base",
                    "xG Proy.", "% Gol", "xA Proy.", "% Asist.", "% Gol o Asist.", "Índice Figura"
                ]
                show = agregar_columna_escudo(show, col_equipo="Club")
                st.dataframe(
                    show.style.format({
                        "Nota Base": "{:.2f}",
                        "xG Proy.": "{:.2f}",
                        "% Gol": "{:.1f}%",
                        "xA Proy.": "{:.2f}",
                        "% Asist.": "{:.1f}%",
                        "% Gol o Asist.": "{:.1f}%",
                        "Índice Figura": "{:.2f}",
                    })
                    .set_properties(subset=["Jugador"], **{"font-weight": "bold"})
                    .set_properties(subset=["% Gol"], **{"color": RED, "font-weight": "bold"})
                    .set_properties(subset=["Índice Figura"], **{"color": GOLD, "font-weight": "bold"}),
                    column_config={"Escudo": config_columna_escudo("")},
                    hide_index=True,
                    width="stretch",
                    height=min(alto, (len(show) * 35) + 42),
                )

            with sub_dest:
                c_g1, c_g2 = st.columns(2)
                with c_g1:
                    st.markdown("**⚽ Mayor Probabilidad de Gol**")
                    top_g = df_proy_jug.sort_values("Prob_Gol", ascending=False).head(6)[
                        ["Jugador", "Equipo", "Rol", "Posición", "xG_Proy", "Prob_Gol"]
                    ].rename(columns={"Equipo": "Club", "Posición": "Pos", "xG_Proy": "xG Proy.", "Prob_Gol": "% Gol"})
                    top_g = agregar_columna_escudo(top_g, col_equipo="Club")
                    st.dataframe(
                        top_g.style.format({"xG Proy.": "{:.2f}", "% Gol": "{:.1f}%"})
                        .set_properties(subset=["Jugador"], **{"font-weight": "bold"})
                        .set_properties(subset=["% Gol"], **{"color": RED, "font-weight": "bold"}),
                        column_config={"Escudo": config_columna_escudo("")},
                        hide_index=True,
                        width="stretch",
                    )
                with c_g2:
                    st.markdown("**🎯 Mayor Probabilidad de Gol o Asistencia**")
                    top_ga = df_proy_jug.sort_values("Prob_GA", ascending=False).head(6)[
                        ["Jugador", "Equipo", "Rol", "Posición", "Prob_Gol", "Prob_Asist", "Prob_GA"]
                    ].rename(columns={"Equipo": "Club", "Posición": "Pos", "Prob_Gol": "% Gol", "Prob_Asist": "% Asist.", "Prob_GA": "% G+A"})
                    top_ga = agregar_columna_escudo(top_ga, col_equipo="Club")
                    st.dataframe(
                        top_ga.style.format({"% Gol": "{:.1f}%", "% Asist.": "{:.1f}%", "% G+A": "{:.1f}%"})
                        .set_properties(subset=["Jugador"], **{"font-weight": "bold"})
                        .set_properties(subset=["% G+A"], **{"color": GOLD, "font-weight": "bold"}),
                        column_config={"Escudo": config_columna_escudo("")},
                        hide_index=True,
                        width="stretch",
                    )

                st.markdown("**🏆 Ranking Global de Figuras Proyectadas del Partido**")
                _render_tabla_proy(df_proy_jug.head(12), alto=460)

            with sub_loc:
                st.markdown(f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>{img_escudo_html(eq_local, 28)}<b>{eq_local}</b></div>", unsafe_allow_html=True)
                _render_tabla_proy(df_proy_jug[df_proy_jug["Equipo"] == eq_local], alto=520)

            with sub_vis:
                st.markdown(f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>{img_escudo_html(eq_vis, 28)}<b>{eq_vis}</b></div>", unsafe_allow_html=True)
                _render_tabla_proy(df_proy_jug[df_proy_jug["Equipo"] == eq_vis], alto=520)