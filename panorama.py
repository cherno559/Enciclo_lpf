import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RED = "#ED1A3B"
GRAY = "#4a4a52"
GRID = "rgba(255,255,255,0.06)"
AXIS_TXT = "#8b8b95"
AXIS_LINE = "#26262d"

PLOT_LAYOUT = dict(
    font=dict(family="Manrope, sans-serif", size=12, color="#a0a0a8"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=20, t=30, b=10),
    hoverlabel=dict(bgcolor="#17171c", bordercolor=RED, font=dict(family="Manrope", size=12, color="#ffffff")),
)


def _estilizar_ejes(fig, grid_x=False, grid_y=True):
    base = dict(zeroline=False, showline=True, linecolor=AXIS_LINE, tickfont=dict(size=11, color=AXIS_TXT))
    fig.update_xaxes(showgrid=grid_x, gridcolor=GRID, **base)
    fig.update_yaxes(showgrid=grid_y, gridcolor=GRID, **base)
    return fig


def _enriquecer_con_rival(df_eq: pd.DataFrame) -> pd.DataFrame:
    """Cruza cada fila con la fila del rival en el mismo match_id para tener métricas 'En Contra'."""
    cols_metricas = [
        "expected_goals", "ball_possession", "total_shots_on_goal", "shots_off_goal",
        "blocked_scoring_attempt", "big_chance_created", "corner_kicks", "passes",
        "accurate_passes", "total_tackle", "interception_won", "total_clearance",
        "goalkeeper_saves", "goals_prevented", "total_shots_inside_box", "fouls",
    ]
    cols_disp = [c for c in cols_metricas if c in df_eq.columns]
    rival_df = df_eq[["match_id", "equipo"] + cols_disp].rename(
        columns={"equipo": "rival", **{c: f"{c}_contra" for c in cols_disp}}
    )
    merged = df_eq.merge(rival_df, on=["match_id", "rival"], how="left")

    # Tiros totales propios y en contra
    merged["tiros_totales"] = (
        merged["total_shots_on_goal"].fillna(0)
        + merged["shots_off_goal"].fillna(0)
        + merged["blocked_scoring_attempt"].fillna(0)
    )
    merged["tiros_totales_contra"] = (
        merged["total_shots_on_goal_contra"].fillna(0)
        + merged["shots_off_goal_contra"].fillna(0)
        + merged["blocked_scoring_attempt_contra"].fillna(0)
    )
    return merged


def render_panorama(df_eq: pd.DataFrame):
    if df_eq.empty:
        st.warning("No hay partidos para el filtro seleccionado.")
        return

    df = _enriquecer_con_rival(df_eq)

    # ── 1. TARJETAS KPI DEL TORNEO ───────────────────────────────────
    partidos_unicos = df[df["condicion"].str.lower() == "local"]
    total_partidos = len(partidos_unicos)
    prom_goles = (partidos_unicos["goles_favor"] + partidos_unicos["goles_contra"]).mean()
    prom_xg = (partidos_unicos["expected_goals"] + partidos_unicos["expected_goals_contra"]).mean()
    pct_local = (partidos_unicos["goles_favor"] > partidos_unicos["goles_contra"]).mean() * 100
    pct_valla = df["valla_invicta"].mean() * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Partidos Disputados</div>
            <div class="kpi-value">{total_partidos}</div>
            <div class="kpi-sub">Encuentros registrados</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Goles / Partido</div>
            <div class="kpi-value accent">{prom_goles:.2f}</div>
            <div class="kpi-sub">Promedio total por duelo</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">xG / Partido</div>
            <div class="kpi-value">{prom_xg:.2f}</div>
            <div class="kpi-sub">Goles esperados combinados</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Triunfos Locales</div>
            <div class="kpi-value">{pct_local:.1f}%</div>
            <div class="kpi-sub">Factor localía en el torneo</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Vallas Invictas</div>
            <div class="kpi-value">{pct_valla:.1f}%</div>
            <div class="kpi-sub">Arcos en cero por equipo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. TABLA DE POSICIONES ───────────────────────────────────────
    st.markdown('<div class="section-header">Tabla de Posiciones y Rendimiento Esperado</div>', unsafe_allow_html=True)

    c_filt1, c_filt2 = st.columns([2, 2])
    with c_filt1:
        cond_tabla = st.radio("Escenario", ["General", "Local", "Visitante"], horizontal=True, key="pan_cond")
    with c_filt2:
        solo_reg = st.toggle("Solo Fase Regular (excluir Playoffs de la tabla)", value=True)

    d_tab = df.copy()
    if solo_reg and "es_playoff" in d_tab.columns:
        d_tab = d_tab[~d_tab["es_playoff"].fillna(False)]
    if cond_tabla != "General":
        d_tab = d_tab[d_tab["condicion"].str.capitalize() == cond_tabla]

    if not d_tab.empty:
        d_tab["V"] = (d_tab["goles_favor"] > d_tab["goles_contra"]).astype(int)
        d_tab["E"] = (d_tab["goles_favor"] == d_tab["goles_contra"]).astype(int)
        d_tab["D"] = (d_tab["goles_favor"] < d_tab["goles_contra"]).astype(int)

        tabla = (
            d_tab.groupby("equipo", as_index=False)
            .agg(
                PJ=("match_id", "count"),
                V=("V", "sum"),
                E=("E", "sum"),
                D=("D", "sum"),
                GF=("goles_favor", "sum"),
                GC=("goles_contra", "sum"),
                DG=("diferencia_gol", "sum"),
                PTS=("puntos", "sum"),
                xGF=("expected_goals", "sum"),
                xGC=("expected_goals_contra", "sum"),
            )
            .sort_values(["PTS", "DG", "GF"], ascending=[False, False, False])
            .reset_index(drop=True)
        )
        tabla.insert(0, "#", tabla.index + 1)
        tabla["Efec %"] = (tabla["PTS"] / (tabla["PJ"] * 3) * 100).round(1)
        tabla["Dif xG"] = (tabla["xGF"] - tabla["xGC"]).round(2)
        tabla = tabla.rename(columns={"equipo": "Equipo"})

        cols_orden = ["#", "Equipo", "PJ", "V", "E", "D", "GF", "GC", "DG", "PTS", "Efec %", "xGF", "xGC", "Dif xG"]
        tabla = tabla[cols_orden]

        st.dataframe(
            tabla.style.format({
                "Efec %": "{:.1f}%",
                "xGF": "{:.2f}",
                "xGC": "{:.2f}",
                "Dif xG": "{:+.2f}",
            })
            .set_properties(subset=["#"], **{"color": RED, "font-weight": "bold"})
            .set_properties(subset=["PTS"], **{"background-color": "#2a1318", "color": "#ffffff", "font-weight": "bold"}),
            hide_index=True,
            use_container_width=True,
            height=(len(tabla) * 35) + 42,
        )

    # ── 3. MAPA DE EFICIENCIA (xG A FAVOR VS xG EN CONTRA) ───────────
    st.markdown('<div class="section-header">Mapa de Eficiencia Colectiva (xG por Partido)</div>', unsafe_allow_html=True)

    df_scatter = (
        df.groupby("equipo", as_index=False)
        .agg(
            xG_Favor=("expected_goals", "mean"),
            xG_Contra=("expected_goals_contra", "mean"),
            Posesion=("ball_possession", "mean"),
        )
    )

    fig_sc = go.Figure(
        go.Scatter(
            x=df_scatter["xG_Favor"],
            y=df_scatter["xG_Contra"],
            mode="markers+text",
            text=df_scatter["equipo"],
            textposition="top center",
            marker=dict(size=13, color=RED, opacity=0.92, line=dict(width=2, color="#0a0a0c")),
            textfont=dict(family="Manrope", size=11, color="#d4d4da"),
            hovertemplate="<b>%{text}</b><br>xG a favor/pj: %{x:.2f}<br>xG en contra/pj: %{y:.2f}<extra></extra>",
        )
    )
    fig_sc.add_vline(x=df_scatter["xG_Favor"].mean(), line=dict(color=GRAY, dash="dash", width=1))
    fig_sc.add_hline(y=df_scatter["xG_Contra"].mean(), line=dict(color=GRAY, dash="dash", width=1))
    fig_sc.update_layout(
        **PLOT_LAYOUT,
        height=520,
        xaxis_title="xG Generado por Partido (Más a la derecha = Mejor ataque)",
        yaxis_title="xG Concedido por Partido (Más abajo = Mejor defensa)",
    )
    _estilizar_ejes(fig_sc, grid_x=True, grid_y=True)
    fig_sc.update_yaxes(autorange="reversed")  # Abajo = menos xG recibido (mejor defensa)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── 4. RANKING DE MÉTRICAS GLOBALES ──────────────────────────────
    st.markdown('<div class="section-header">Rankings por Métrica (Promedio por Partido)</div>', unsafe_allow_html=True)

    metricas_opciones = {
        "Goles esperados (xG)": "expected_goals",
        "Posesión de balón (%)": "ball_possession",
        "Tiros totales": "tiros_totales",
        "Tiros al arco": "total_shots_on_goal",
        "Ocasiones claras": "big_chance_created",
        "Tiros dentro del área": "total_shots_inside_box",
        "Córners": "corner_kicks",
        "Quites": "total_tackle",
        "Intercepciones": "interception_won",
        "Despejes": "total_clearance",
        "Atajadas del arquero": "goalkeeper_saves",
        "Goles evitados (arquero)": "goals_prevented",
    }
    disponibles = {k: v for k, v in metricas_opciones.items() if v in df.columns}

    cm1, cm2 = st.columns([2, 1.5])
    with cm1:
        label_m = st.selectbox("Elegir estadística", list(disponibles.keys()))
    with cm2:
        enfoque = st.radio("Enfoque", ["A Favor", "En Contra"], horizontal=True, key="pan_enf")

    col_base = disponibles[label_m]
    col_usar = col_base if enfoque == "A Favor" else f"{col_base}_contra"

    if col_usar in df.columns:
        ranking = df.groupby("equipo", as_index=False)[col_usar].mean().sort_values(col_usar, ascending=False)
        color_bar = RED if enfoque == "A Favor" else GRAY
        fig_bar = px.bar(ranking, x="equipo", y=col_usar, color_discrete_sequence=[color_bar])
        fig_bar.update_traces(
            marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{label_m} ({enfoque.lower()}): %{{y:.2f}}<extra></extra>",
        )
        fig_bar.update_layout(**PLOT_LAYOUT, height=420, bargap=0.28, xaxis_title="", yaxis_title="Promedio / PJ")
        _estilizar_ejes(fig_bar)
        fig_bar.update_xaxes(tickangle=-40)
        st.plotly_chart(fig_bar, use_container_width=True)