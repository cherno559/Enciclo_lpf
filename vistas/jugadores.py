from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from escudos_helper import (
    agregar_columna_escudo,
    config_columna_escudo,
    escudo_base64,
    img_escudo_html,
)

RED = "#ED1A3B"
PALETTE = ["#ED1A3B", "#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#ec4899"]
GRAY = "#4a4a52"
GRID = "rgba(255,255,255,0.06)"
AXIS_TXT = "#8b8b95"
AXIS_LINE = "#26262d"

PLOT_LAYOUT = dict(
    font=dict(family="Manrope, sans-serif", size=12, color="#a0a0a8"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=20, t=25, b=10),
    hoverlabel=dict(bgcolor="#17171c", bordercolor=RED, font=dict(family="Manrope", size=12, color="#ffffff")),
)

TRADUCCION_CUERPO = {
    "right-foot": "Pie derecho",
    "left-foot": "Pie izquierdo",
    "head": "Cabeza",
    "other": "Otro",
}

TRADUCCION_SITUACION = {
    "regular": "Jugada abierta",
    "assisted": "Asistido",
    "set-piece": "Pelota parada",
    "corner": "Córner",
    "free-kick": "Tiro libre",
    "penalty": "Penal",
    "fast-break": "Contraataque",
    "throw-in-set-piece": "Lateral",
}

TRADUCCION_ZONA = {
    "low-centre": "Abajo al centro",
    "low-left": "Abajo a la izquierda",
    "low-right": "Abajo a la derecha",
    "bottom-left": "Abajo a la izquierda",
    "bottom-right": "Abajo a la derecha",
    "bottom-centre": "Abajo al centro",
    "top-left": "Ángulo izquierdo",
    "top-right": "Ángulo derecho",
    "top-centre": "Arriba al centro",
    "high": "Por encima del travesaño",
    "high-left": "Afuera arriba izquierda",
    "high-right": "Afuera arriba derecha",
    "high-centre": "Por encima del travesaño",
    "close-high": "Cerca por arriba",
    "close-left": "Cerca palo izquierdo",
    "close-right": "Cerca palo derecho",
    "left": "Desviado a la izquierda",
    "right": "Desviado a la derecha",
    "wide-left": "Desviado a la izquierda",
    "wide-right": "Desviado a la derecha",
    "blocked": "Bloqueado",
    "post-left": "Palo izquierdo",
    "post-right": "Palo derecho",
    "bar": "Travesaño",
}


def _estilizar_ejes(fig, grid_x=False, grid_y=True):
    base = dict(zeroline=False, showline=True, linecolor=AXIS_LINE, tickfont=dict(size=11, color=AXIS_TXT))
    fig.update_xaxes(showgrid=grid_x, gridcolor=GRID, **base)
    fig.update_yaxes(showgrid=grid_y, gridcolor=GRID, **base)
    return fig


def _club_actual(equipo_str: str) -> str:
    """Devuelve el club más reciente (el último en 'Club Anterior / Club Actual')."""
    partes = [p.strip() for p in str(equipo_str).split(" / ") if p.strip()]
    return partes[-1] if partes else ""


@st.cache_data(show_spinner=False)
def _cargar_tiros_fallback() -> pd.DataFrame:
    ruta = Path(__file__).resolve().parents[1] / "datos" / "tiros.parquet"
    if not ruta.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ruta)
    if "fecha_dt" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _agregar_jugadores(df_jug: pd.DataFrame) -> pd.DataFrame:
    """Agrupa estadísticas por jugador consolidando partidos en distintas posiciones."""
    d = df_jug[df_jug["minutesPlayed"].fillna(0) > 0].copy()
    if d.empty:
        return pd.DataFrame()

    if "fecha_dt" in d.columns:
        d = d.sort_values("fecha_dt")

    d["titular"] = (~d["suplente"].fillna(True)).astype(int)
    d["tiros_calc"] = np.where(
        d["totalShots"].notna(),
        d["totalShots"],
        d["onTargetScoringAttempt"].fillna(0) + d["shotOffTarget"].fillna(0) + d["blockedScoringAttempt"].fillna(0),
    )

    if "jugador_id" in d.columns:
        d["_uid"] = d["jugador_id"].fillna(d["jugador"]).astype(str)
    else:
        d["_uid"] = d["jugador"].astype(str)
    id_col = "_uid"

    pos_principal = (
        d.groupby([id_col, "posicion_desc"], as_index=False)["minutesPlayed"]
        .sum()
        .sort_values("minutesPlayed", ascending=False)
        .drop_duplicates(id_col)
        .set_index(id_col)["posicion_desc"]
    )

    agg_dict = dict(
        Jugador=("jugador", "first"),
        Equipo=("equipo", lambda s: " / ".join(dict.fromkeys(s.dropna()))),
        PJ=("match_id", "count"),
        Titular=("titular", "sum"),
        Minutos=("minutesPlayed", "sum"),
        Nota=("rating", "mean"),
        Goles=("goals", "sum"),
        Asist=("goalAssist", "sum"),
        xG=("expectedGoals", "sum"),
        xGOT=("expectedGoalsOnTarget", "sum"),
        xA=("expectedAssists", "sum"),
        Tiros=("tiros_calc", "sum"),
        Al_Arco=("onTargetScoringAttempt", "sum"),
        Oc_Falladas=("bigChanceMissed", "sum"),
        Oc_Creadas=("bigChanceCreated", "sum"),
        Pases_Clave=("keyPass", "sum"),
        Pases_Ok=("accuratePass", "sum"),
        Pases_Tot=("totalPass", "sum"),
        Regates_Ok=("wonContest", "sum"),
        Regates_Tot=("totalContest", "sum"),
        Dist_Prog=("totalProgressiveBallCarriesDistance", "sum"),
        Cond_Prog=("progressiveBallCarriesCount", "sum"),
        Quites_Ok=("wonTackle", "sum"),
        Intercep=("interceptionWon", "sum"),
        Recup=("ballRecovery", "sum"),
        Despejes=("totalClearance", "sum"),
        Duelos_Ok=("duelWon", "sum"),
        Duelos_Perd=("duelLost", "sum"),
        Aereos_Ok=("aerialWon", "sum"),
        Aereos_Perd=("aerialLost", "sum"),
        Perdidas=("possessionLostCtrl", "sum"),
        Atajadas=("saves", "sum"),
        Goles_Rec=("goalsConceded", "sum"),
        Goles_Evit=("goalsPrevented", "sum"),
        Ataj_Area=("savedShotsFromInsideTheBox", "sum"),
        Pen_Ataj=("penaltySave", "sum"),
        Salidas_Arq=("accurateKeeperSweeper", "sum"),
        Km_Tot=("kilometersCovered", "sum"),
        Vel_Max=("topSpeed", "max"),
        Sprints=("numberOfSprints", "sum"),
    )

    cols_df = set(d.columns) | {"titular", "tiros_calc"}
    agg_valid = {k: v for k, v in agg_dict.items() if v[0] in cols_df}
    res = d.groupby(id_col, as_index=False).agg(**agg_valid)

    res["Posicion"] = res[id_col].map(pos_principal).fillna("Sin dato")
    res = res.drop(columns=[id_col], errors="ignore")

    res["Minutos"] = res["Minutos"].fillna(0).round(0).astype(int)
    res["PJ"] = res["PJ"].astype(int)
    res["Titular"] = res["Titular"].astype(int)

    res["Pases_Pct"] = np.where(res["Pases_Tot"] > 0, (res["Pases_Ok"] / res["Pases_Tot"]) * 100, 0.0)
    res["Regates_Pct"] = np.where(res["Regates_Tot"] > 0, (res["Regates_Ok"] / res["Regates_Tot"]) * 100, 0.0)
    duelos_tot = res["Duelos_Ok"] + res["Duelos_Perd"]
    res["Duelos_Pct"] = np.where(duelos_tot > 0, (res["Duelos_Ok"] / duelos_tot) * 100, 0.0)
    aereos_tot = res["Aereos_Ok"] + res["Aereos_Perd"]
    res["Aereos_Pct"] = np.where(aereos_tot > 0, (res["Aereos_Ok"] / aereos_tot) * 100, 0.0)

    res["Factor_90"] = np.where(res["Minutos"] > 0, 90.0 / res["Minutos"], 0.0)
    return res


def _dibujar_media_cancha(fig: go.Figure):
    line_color = "rgba(255, 255, 255, 0.25)"
    shapes = [
        # Media cancha (x: 50 a 100, y: 0 a 100)
        dict(type="rect", x0=50, y0=0, x1=100, y1=100, line=dict(color=line_color, width=1.5), fillcolor="#141419", layer="below"),
        # Área grande
        dict(type="rect", x0=83, y0=21.1, x1=100, y1=78.9, line=dict(color=line_color, width=1.5), layer="below"),
        # Área chica
        dict(type="rect", x0=94.2, y0=36.8, x1=100, y1=63.2, line=dict(color=line_color, width=1.5), layer="below"),
        # Arco
        dict(type="rect", x0=100, y0=45.2, x1=101.6, y1=54.8, line=dict(color="#ffffff", width=2), fillcolor="rgba(255,255,255,0.18)", layer="below"),
        # Punto penal
        dict(type="circle", x0=88.2, y0=49.5, x1=88.8, y1=50.5, line=dict(color=line_color, width=1.5), fillcolor=line_color, layer="below"),
        # Semicírculo central
        dict(type="path", path="M 50 38 A 9.15 12 0 0 1 50 62", line=dict(color=line_color, width=1.5), layer="below"),
    ]
    fig.update_layout(shapes=shapes)


def _render_rendimiento_partido(df_jug: pd.DataFrame, df_tiros: pd.DataFrame, tiene_avanzadas: bool):
    d_act = df_jug[df_jug["minutesPlayed"].fillna(0) > 0].copy()
    if d_act.empty:
        st.info("No hay partidos con minutos registrados para el período seleccionado.")
        return

    if df_tiros is None or df_tiros.empty:
        df_tiros = _cargar_tiros_fallback()

    c1, c2, c3 = st.columns([1.3, 1.5, 2.2])
    equipos = sorted(d_act["equipo"].dropna().unique().tolist())
    with c1:
        eq_sel = st.selectbox("Club", equipos, key="rend_part_eq")

    sub_eq = d_act[d_act["equipo"] == eq_sel].copy()
    jugadores = (
        sub_eq.groupby("jugador", as_index=False)["minutesPlayed"]
        .sum()
        .sort_values("minutesPlayed", ascending=False)["jugador"]
        .tolist()
    )
    with c2:
        jug_sel = st.selectbox("Jugador", jugadores, key="rend_part_jug")

    sub_jug = sub_eq[sub_eq["jugador"] == jug_sel].sort_values("fecha_dt", ascending=False).copy()

    opciones_partido = ["Todos los partidos del período"]
    mapa_mid = {}
    for _, r in sub_jug.iterrows():
        fecha_str = pd.to_datetime(r["fecha_dt"]).strftime("%d/%m") if pd.notna(r.get("fecha_dt")) else ""
        inst = f"F{int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) and pd.notna(r.get("ronda_num")) else str(r.get("fase", ""))
        pref = f"{inst} · " if inst else ""
        rival = r.get("rival", "Rival")
        mins = int(round(r.get("minutesPlayed", 0)))
        nota = f" · Nota {r['rating']:.1f}" if pd.notna(r.get("rating")) else ""
        etiqueta = f"{pref}{fecha_str} vs {rival} ({mins}'{nota})"
        opciones_partido.append(etiqueta)
        mapa_mid[etiqueta] = r["match_id"]

    with c3:
        partido_sel = st.selectbox("Partido", opciones_partido, key="rend_part_match")

    mid_sel = mapa_mid.get(partido_sel)
    df_p = sub_jug[sub_jug["match_id"] == mid_sel].copy() if mid_sel is not None else sub_jug.copy()

    pos_jugador = (
        df_p.groupby("posicion_desc")["minutesPlayed"].sum().idxmax()
        if "posicion_desc" in df_p.columns and not df_p["posicion_desc"].dropna().empty
        else "Jugador"
    )
    subtitulo_partido = (
        f"vs {df_p.iloc[0]['rival']} ({df_p.iloc[0]['fecha']})"
        if mid_sel is not None and len(df_p) > 0
        else f"{len(df_p)} partidos disputados"
    )

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; margin:6px 0 14px 0;">
        {img_escudo_html(eq_sel, 46)}
        <div>
            <div style="font-family:'Oswald',sans-serif; font-size:1.65rem; color:#ffffff; line-height:1.05; letter-spacing:0.8px;">
                {jug_sel.upper()}
            </div>
            <div style="font-size:0.82rem; color:#8b8b95;">{eq_sel} · {pos_jugador} · <b style="color:#ececf0;">{subtitulo_partido}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs de Rendimiento del Partido ──
    mins_tot = int(round(df_p["minutesPlayed"].sum()))
    nota_prom = df_p["rating"].mean() if "rating" in df_p.columns and df_p["rating"].notna().any() else np.nan
    nota_txt = f"{nota_prom:.2f}" if pd.notna(nota_prom) else "-"
    pases_ok = int(df_p["accuratePass"].fillna(0).sum()) if "accuratePass" in df_p.columns else 0
    pases_tot = int(df_p["totalPass"].fillna(0).sum()) if "totalPass" in df_p.columns else 0
    pct_pase = (pases_ok / pases_tot * 100) if pases_tot > 0 else 0.0

    if "arquero" in pos_jugador.lower():
        ataj = int(df_p["saves"].fillna(0).sum()) if "saves" in df_p.columns else 0
        g_rec = int(df_p["goalsConceded"].fillna(0).sum()) if "goalsConceded" in df_p.columns else 0
        g_evit = float(df_p["goalsPrevented"].fillna(0).sum()) if "goalsPrevented" in df_p.columns else 0.0
        evit_sub = f"Goles evitados: {g_evit:+.2f}" if tiene_avanzadas else "Valla del equipo"
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Nota SofaScore</div><div class="kpi-value" style="color:#cfb45e">{nota_txt}</div><div class="kpi-sub">{mins_tot}' jugados</div></div>
            <div class="kpi-card"><div class="kpi-label">Atajadas</div><div class="kpi-value">{ataj}</div><div class="kpi-sub">Remates contenidos</div></div>
            <div class="kpi-card"><div class="kpi-label">Goles Recibidos</div><div class="kpi-value">{g_rec}</div><div class="kpi-sub">{evit_sub}</div></div>
            <div class="kpi-card"><div class="kpi-label">Precisión de Pases</div><div class="kpi-value">{pct_pase:.0f}%</div><div class="kpi-sub">{pases_ok}/{pases_tot} completados</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        goles = int(df_p["goals"].fillna(0).sum()) if "goals" in df_p.columns else 0
        asist = int(df_p["goalAssist"].fillna(0).sum()) if "goalAssist" in df_p.columns else 0
        xg = float(df_p["expectedGoals"].fillna(0).sum()) if "expectedGoals" in df_p.columns else 0.0
        xa = float(df_p["expectedAssists"].fillna(0).sum()) if "expectedAssists" in df_p.columns else 0.0
        p_clave = int(df_p["keyPass"].fillna(0).sum()) if "keyPass" in df_p.columns else 0
        quites = int(df_p["wonTackle"].fillna(0).sum()) if "wonTackle" in df_p.columns else 0
        interc = int(df_p["interceptionWon"].fillna(0).sum()) if "interceptionWon" in df_p.columns else 0
        recup = int(df_p["ballRecovery"].fillna(0).sum()) if "ballRecovery" in df_p.columns else 0
        duelos_g = int(df_p["duelWon"].fillna(0).sum()) if "duelWon" in df_p.columns else 0

        xg_txt = f"xG generado: {xg:.2f}" if tiene_avanzadas else f"Tiros al arco: {int(df_p['onTargetScoringAttempt'].fillna(0).sum())}"
        xa_txt = f"xA: {xa:.2f} · Pases clave: {p_clave}" if tiene_avanzadas else f"Pases clave: {p_clave}"

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Nota SofaScore</div><div class="kpi-value" style="color:#cfb45e">{nota_txt}</div><div class="kpi-sub">{mins_tot}' en cancha</div></div>
            <div class="kpi-card"><div class="kpi-label">Goles / Asistencias</div><div class="kpi-value">{goles} <span style="color:#8b8b95; font-size:1.4rem;">/</span> {asist}</div><div class="kpi-sub">{xg_txt} · {xa_txt}</div></div>
            <div class="kpi-card"><div class="kpi-label">Distribución</div><div class="kpi-value">{pct_pase:.0f}%</div><div class="kpi-sub">{pases_ok}/{pases_tot} pases · {int(df_p['wonContest'].fillna(0).sum())} regates</div></div>
            <div class="kpi-card"><div class="kpi-label">Defensa y Duelos</div><div class="kpi-value">{recup} <span style="font-size:1.1rem; color:#8b8b95;">recup.</span></div><div class="kpi-sub">{quites} quites · {interc} interc. · {duelos_g} duelos</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Preparación común de remates del jugador/partido ──
    col_jug_t = next((c for c in ["jugador", "player", "player_name"] if c in df_tiros.columns), None) if not df_tiros.empty else None
    t_sub = df_tiros[df_tiros[col_jug_t] == jug_sel].copy() if col_jug_t else pd.DataFrame()
    if not t_sub.empty and "match_id" in t_sub.columns:
        if mid_sel is not None:
            t_sub = t_sub[t_sub["match_id"] == mid_sel].copy()
        else:
            t_sub = t_sub[t_sub["match_id"].isin(sub_jug["match_id"].unique())].copy()

    if not t_sub.empty:
        res_col = "resultado" if "resultado" in t_sub.columns else None
        xg_col = next((c for c in ["xg", "expectedGoals", "xG"] if c in t_sub.columns), None)
        t_sub["_xg_val"] = pd.to_numeric(t_sub[xg_col], errors="coerce").fillna(0.08) if xg_col else 0.10

        def _cat_tiro(r):
            val = str(r.get(res_col, "")).lower() if res_col else ""
            if val == "goal" or r.get("es_gol", 0) == 1:
                return "Gol"
            if val in ("save", "post", "on_target"):
                return "Al arco / Palo"
            return "Afuera / Bloqueado"

        t_sub["Desenlace"] = t_sub.apply(_cat_tiro, axis=1)
        t_sub["_zona_es"] = (
            t_sub["zona_arco"].astype(str).str.lower().map(TRADUCCION_ZONA).fillna("-")
            if "zona_arco" in t_sub.columns
            else "-"
        )
        t_sub["_cuerpo_es"] = (
            t_sub["parte_cuerpo"].astype(str).str.lower().map(TRADUCCION_CUERPO).fillna("-")
            if "parte_cuerpo" in t_sub.columns
            else "-"
        )
        t_sub["_sit_es"] = (
            t_sub["situacion"].astype(str).str.lower().map(TRADUCCION_SITUACION).fillna("-")
            if "situacion" in t_sub.columns
            else "-"
        )

    # ── Mapa de Remates + Detalle de Tiros ──
    col_map, col_det = st.columns([1.3, 1.1])

    with col_map:
        st.markdown('<div class="section-header">🎯 Mapa de Remates</div>', unsafe_allow_html=True)
        col_x = next((c for c in ["x", "coord_x", "pos_x", "shot_x", "X"] if c in t_sub.columns), None) if not t_sub.empty else None
        col_y = next((c for c in ["y", "coord_y", "pos_y", "shot_y", "Y"] if c in t_sub.columns), None) if not t_sub.empty else None

        if t_sub.empty:
            st.info("Sin remates registrados para esta selección.")
        elif not col_x or not col_y:
            st.info("La base de tiros actual no incluye coordenadas (x, y) en cancha; podés ver el detalle completo de cada remate en la tabla de la derecha.")
        else:
            tx = pd.to_numeric(t_sub[col_x], errors="coerce")
            ty = pd.to_numeric(t_sub[col_y], errors="coerce")
            if tx.median() < 50:
                tx = 100.0 - tx
            t_sub["_px"] = tx.clip(50, 100)
            t_sub["_py"] = ty.clip(0, 100)

            colores = {"Gol": "#cfb45e", "Al arco / Palo": RED, "Afuera / Bloqueado": "#8b8b95"}
            fig_shot = go.Figure()
            _dibujar_media_cancha(fig_shot)

            for cat in ["Afuera / Bloqueado", "Al arco / Palo", "Gol"]:
                grp = t_sub[t_sub["Desenlace"] == cat]
                if grp.empty:
                    continue
                fig_shot.add_trace(
                    go.Scatter(
                        x=grp["_px"],
                        y=grp["_py"],
                        mode="markers",
                        name=f"{cat} ({len(grp)})",
                        marker=dict(
                            size=np.clip(grp["_xg_val"] * 40 + 11, 11, 32),
                            color=colores.get(cat, "#ffffff"),
                            line=dict(color="#0a0a0c", width=1.4),
                            symbol="star" if cat == "Gol" else "circle",
                        ),
                        customdata=np.stack([grp["_xg_val"].round(2), grp["_zona_es"], grp["_cuerpo_es"], grp["_sit_es"]], axis=-1),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "xG: %{customdata[0]}<br>"
                            "Destino: %{customdata[1]}<br>"
                            "Perfil: %{customdata[2]} · %{customdata[3]}<extra></extra>"
                        ),
                    )
                )

            fig_shot.update_layout(
                **PLOT_LAYOUT,
                height=390,
                xaxis=dict(range=[49, 102], visible=False, fixedrange=True),
                yaxis=dict(range=[-2, 102], visible=False, scaleanchor="x", scaleratio=0.68, fixedrange=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_shot, width="stretch")

    with col_det:
        st.markdown('<div class="section-header">📋 Detalle de Remates</div>', unsafe_allow_html=True)
        if t_sub.empty:
            st.caption("No hay remates registrados en este recorte.")
        else:
            df_t_show = pd.DataFrame({
                "Min": pd.to_numeric(t_sub["minuto"], errors="coerce").fillna(0).astype(int) if "minuto" in t_sub.columns else "-",
                "Resultado": t_sub["Desenlace"],
                "xG": t_sub["_xg_val"],
                "xGOT": pd.to_numeric(t_sub["xgot"], errors="coerce").fillna(0.0) if "xgot" in t_sub.columns else 0.0,
                "Zona": t_sub["_zona_es"],
                "Perfil": t_sub["_cuerpo_es"],
                "Situación": t_sub["_sit_es"],
            })
            st.dataframe(
                df_t_show.style.format({"xG": "{:.2f}", "xGOT": "{:.2f}"}),
                hide_index=True,
                width="stretch",
                height=390,
            )


def render_jugadores(df_jug: pd.DataFrame, df_tiros: pd.DataFrame, periodo: str = "Temporada Completa"):
    if df_jug.empty:
        st.warning("No hay datos de jugadores para el período seleccionado.")
        return

    tiene_avanzadas = (periodo == "Torneo Clausura") and (df_jug["expectedGoals"].fillna(0).sum() > 0)
    tiene_fisico = ("kilometersCovered" in df_jug.columns) and (df_jug["kilometersCovered"].fillna(0).sum() > 0)

    # Sub-navegación con carga diferida (solo ejecuta la vista activa)
    sub_vista = st.radio(
        "Vista de jugadores",
        options=[
            "📋 Tablas y Rankings",
            "🔎 Ficha de Jugador",
            "🎯 Rendimiento por Partido",
            "⚔️ Comparador",
            "🏆 Líderes por Fecha",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="jug_subvista_nav",
    )

    # =========================================================================
    # VISTA 1: TABLAS Y RANKINGS
    # =========================================================================
    if sub_vista == "📋 Tablas y Rankings":
        f1, f2, f3, f4 = st.columns([1.4, 1.3, 1.3, 1.2])
        equipos_disp = ["Todos los equipos"] + sorted(df_jug["equipo"].dropna().unique())
        posiciones_disp = ["Todas las posiciones"] + sorted(df_jug["posicion_desc"].dropna().unique())

        with f1:
            club_sel = st.selectbox("Club", equipos_disp, key="jug_club_filt")
        with f2:
            pos_sel = st.selectbox("Posición", posiciones_disp, key="jug_pos_filt")
        with f3:
            categorias = ["General (Todas)", "Ataque", "Creación y Pases", "Defensa y Duelos", "Arqueros"]
            if tiene_fisico:
                categorias.append("Físico")
            cat_sel = st.selectbox("Bloque de métricas", categorias, key="jug_cat_filt")
        with f4:
            modo_val = st.radio("Valores", ["Totales", "Cada 90'"], horizontal=True, key="jug_modo_val")

        d_sub = df_jug if club_sel == "Todos los equipos" else df_jug[df_jug["equipo"] == club_sel]

        df_agg = _agregar_jugadores(d_sub)
        if not df_agg.empty and pos_sel != "Todas las posiciones":
            df_agg = df_agg[df_agg["Posicion"] == pos_sel].copy()

        if df_agg.empty:
            st.info("No hay jugadores con minutos registrados para ese filtro.")
        else:
            max_min = int(df_agg["Minutos"].max())
            default_min = 0 if club_sel != "Todos los equipos" else min(270, max_min // 3)
            min_minutos = st.slider("Mínimo de minutos jugados", 0, max(10, max_min), default_min, step=30)

            df_f = df_agg[df_agg["Minutos"] >= min_minutos].copy()
            if df_f.empty:
                st.warning("Ningún jugador supera ese mínimo de minutos. Bajá el deslizador.")
            else:
                es_p90 = modo_val == "Cada 90'"
                if es_p90:
                    cols_escalar = [
                        "Goles", "Asist", "xG", "xGOT", "xA", "Tiros", "Al_Arco",
                        "Oc_Falladas", "Oc_Creadas", "Pases_Clave", "Pases_Ok",
                        "Regates_Ok", "Dist_Prog", "Cond_Prog", "Quites_Ok",
                        "Intercep", "Recup", "Despejes", "Perdidas", "Atajadas",
                        "Goles_Rec", "Goles_Evit", "Ataj_Area", "Km_Tot", "Sprints",
                    ]
                    for c in cols_escalar:
                        if c in df_f.columns:
                            df_f[c] = df_f[c] * df_f["Factor_90"]

                base_cols = ["Jugador", "Equipo", "Posicion", "PJ", "Titular", "Minutos", "Nota"]
                if cat_sel == "General (Todas)":
                    if tiene_avanzadas:
                        extra = [
                            "Goles", "xG", "xGOT", "Asist", "xA", "Tiros", "Al_Arco", "Oc_Creadas", "Oc_Falladas",
                            "Pases_Clave", "Pases_Pct", "Cond_Prog", "Regates_Ok", "Perdidas",
                            "Quites_Ok", "Intercep", "Recup", "Despejes", "Duelos_Pct", "Aereos_Pct",
                            "Atajadas", "Goles_Rec", "Goles_Evit",
                        ]
                    else:
                        extra = [
                            "Goles", "Asist", "Tiros", "Al_Arco", "Oc_Creadas", "Oc_Falladas",
                            "Pases_Clave", "Pases_Pct", "Regates_Ok", "Perdidas",
                            "Quites_Ok", "Intercep", "Recup", "Despejes", "Duelos_Pct", "Aereos_Pct",
                            "Atajadas", "Goles_Rec",
                        ]
                    if tiene_fisico:
                        extra += ["Km_Tot", "Vel_Max", "Sprints"]
                    orden_col = "Nota"
                elif cat_sel == "Ataque":
                    extra = ["Goles", "xG", "xGOT", "Tiros", "Al_Arco", "Oc_Falladas"] if tiene_avanzadas else ["Goles", "Tiros", "Al_Arco", "Oc_Falladas"]
                    orden_col = "Goles"
                elif cat_sel == "Creación y Pases":
                    extra = ["Asist", "xA", "Pases_Clave", "Oc_Creadas", "Pases_Pct", "Cond_Prog", "Regates_Ok", "Perdidas"] if tiene_avanzadas else ["Asist", "Pases_Clave", "Oc_Creadas", "Pases_Pct", "Regates_Ok", "Perdidas"]
                    orden_col = "Pases_Clave"
                elif cat_sel == "Defensa y Duelos":
                    extra = ["Quites_Ok", "Intercep", "Recup", "Despejes", "Duelos_Pct", "Aereos_Pct", "Perdidas"]
                    orden_col = "Recup"
                elif cat_sel == "Arqueros":
                    df_f = df_f[df_f["Atajadas"] > 0] if pos_sel == "Todas las posiciones" else df_f
                    extra = ["Atajadas", "Goles_Rec", "Goles_Evit", "Pen_Ataj"] if tiene_avanzadas else ["Atajadas", "Goles_Rec", "Pen_Ataj"]
                    orden_col = "Atajadas"
                else:
                    extra = ["Km_Tot", "Vel_Max", "Sprints"]
                    orden_col = "Km_Tot"

                cols_finales = [c for c in (base_cols + extra) if c in df_f.columns]
                df_show = df_f[cols_finales].sort_values([orden_col, "Minutos"], ascending=[False, False]).reset_index(drop=True)
                df_show.insert(0, "#", df_show.index + 1)

                renombres = {
                    "Posicion": "Pos", "Al_Arco": "Al Arco", "Oc_Falladas": "Oc. Falladas",
                    "Oc_Creadas": "Oc. Creadas", "Pases_Clave": "Pases Clave", "Pases_Pct": "Pases %",
                    "Cond_Prog": "Cond. Prog.", "Regates_Ok": "Regates", "Perdidas": "Pérdidas",
                    "Quites_Ok": "Quites", "Intercep": "Intercep.", "Recup": "Recuperac.",
                    "Duelos_Pct": "Duelos %", "Aereos_Pct": "Aéreos %", "Goles_Rec": "Goles Rec.",
                    "Goles_Evit": "Goles Evit.", "Pen_Ataj": "Pen. Ataj.",
                    "Km_Tot": "Km", "Vel_Max": "Vel. Máx (km/h)",
                }
                df_show = df_show.rename(columns=renombres)
                df_show.insert(
                    df_show.columns.get_loc("Equipo"),
                    "Escudo",
                    df_show["Equipo"].map(lambda x: escudo_base64(_club_actual(x))),
                )

                fmt = {"Nota": "{:.2f}", "Pases %": "{:.1f}%", "Duelos %": "{:.1f}%", "Aéreos %": "{:.1f}%", "Vel. Máx (km/h)": "{:.1f}"}
                cols_dec = ["xG", "xGOT", "xA", "Goles Evit.", "Km"]
                if es_p90:
                    cols_dec += ["Goles", "Asist", "Tiros", "Al Arco", "Oc. Falladas", "Oc. Creadas", "Pases Clave", "Cond. Prog.", "Regates", "Pérdidas", "Quites", "Intercep.", "Recuperac.", "Despejes", "Atajadas", "Goles Rec.", "Sprints"]
                else:
                    for ci in ["Goles", "Asist", "Tiros", "Al Arco", "Oc. Falladas", "Oc. Creadas", "Pases Clave", "Cond. Prog.", "Regates", "Pérdidas", "Quites", "Intercep.", "Recuperac.", "Despejes", "Atajadas", "Goles Rec.", "Pen. Ataj.", "Sprints"]:
                        if ci in df_show.columns:
                            df_show[ci] = df_show[ci].fillna(0).round(0).astype(int)

                for cd in cols_dec:
                    if cd in df_show.columns:
                        fmt[cd] = "{:.2f}"

                st.dataframe(
                    df_show.style.format(fmt)
                    .set_properties(subset=["#"], **{"color": RED, "font-weight": "bold"})
                    .set_properties(subset=["Jugador"], **{"font-weight": "bold"})
                    .set_properties(subset=["Nota"], **{"color": "#cfb45e", "font-weight": "bold"}),
                    column_config={"Escudo": config_columna_escudo("")},
                    hide_index=True,
                    width="stretch",
                    height=min(680, (len(df_show) * 35) + 42),
                )

    # =========================================================================
    # VISTA 2: FICHA DE JUGADOR
    # =========================================================================
    elif sub_vista == "🔎 Ficha de Jugador":
        cf1, cf2 = st.columns([1.5, 2])
        with cf1:
            eq_ficha = st.selectbox("Filtrar por club", ["Todos"] + sorted(df_jug["equipo"].dropna().unique()), key="ficha_jug_eq")
        d_pool = df_jug if eq_ficha == "Todos" else df_jug[df_jug["equipo"] == eq_ficha]
        d_pool = d_pool[d_pool["minutesPlayed"].fillna(0) > 0]

        jugadores_lista = sorted(d_pool["jugador"].dropna().unique())
        with cf2:
            jug_sel = st.selectbox("Seleccioná un jugador", jugadores_lista, key="ficha_jug_nombre")

        dj = d_pool[d_pool["jugador"] == jug_sel].sort_values("fecha_dt").reset_index(drop=True)
        if not dj.empty:
            pj = len(dj)
            tit = int((~dj["suplente"].fillna(True)).sum())
            mins = int(round(dj["minutesPlayed"].sum()))
            nota_m = dj["rating"].mean()
            pos_jugador = (
                dj.groupby("posicion_desc")["minutesPlayed"].sum().idxmax()
                if "posicion_desc" in dj.columns and not dj["posicion_desc"].dropna().empty
                else "Jugador"
            )
            equipos_j = list(dict.fromkeys(dj["equipo"].dropna()))
            eq_actual_j = equipos_j[-1] if equipos_j else ""
            eq_txt_j = " / ".join(equipos_j)

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:14px; margin:6px 0 14px 0;">
                {img_escudo_html(eq_actual_j, 48)}
                <div>
                    <div style="font-family:'Oswald',sans-serif; font-size:1.75rem; color:#ffffff; line-height:1.05; letter-spacing:0.8px;">
                        {jug_sel.upper()}
                    </div>
                    <div style="font-size:0.82rem; color:#8b8b95;">{eq_txt_j} · {pos_jugador}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "arquero" in pos_jugador.lower():
                ataj_t = int(dj["saves"].fillna(0).sum())
                grec_t = int(dj["goalsConceded"].fillna(0).sum())
                vallas_j = int((dj["goalsConceded"].fillna(99) == 0).sum())
                gevit_txt = f" · Evitados: {dj['goalsPrevented'].fillna(0).sum():+.2f}" if tiene_avanzadas else ""
                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card"><div class="kpi-label">Presencias · {pos_jugador}</div><div class="kpi-value">{pj} PJ · <span style="color:#ED1A3B">{tit} TIT</span></div><div class="kpi-sub">{mins} minutos jugados</div></div>
                    <div class="kpi-card"><div class="kpi-label">Nota Promedio</div><div class="kpi-value" style="color:#cfb45e">{nota_m:.2f}</div><div class="kpi-sub">Calificación SofaScore</div></div>
                    <div class="kpi-card"><div class="kpi-label">Atajadas Totales</div><div class="kpi-value">{ataj_t}</div><div class="kpi-sub">Promedio: {(ataj_t/pj if pj else 0):.2f} por partido</div></div>
                    <div class="kpi-card"><div class="kpi-label">Goles Recibidos</div><div class="kpi-value">{grec_t}</div><div class="kpi-sub">Vallas invictas: {vallas_j}{gevit_txt}</div></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                goles_t = int(dj["goals"].fillna(0).sum())
                asist_t = int(dj["goalAssist"].fillna(0).sum())
                recup_t = int(dj["ballRecovery"].fillna(0).sum())
                quites_t = int(dj["wonTackle"].fillna(0).sum())

                if "defensor" in pos_jugador.lower():
                    st.markdown(f"""
                    <div class="kpi-grid">
                        <div class="kpi-card"><div class="kpi-label">Presencias · {pos_jugador}</div><div class="kpi-value">{pj} PJ · <span style="color:#ED1A3B">{tit} TIT</span></div><div class="kpi-sub">{mins} minutos jugados</div></div>
                        <div class="kpi-card"><div class="kpi-label">Nota Promedio</div><div class="kpi-value" style="color:#cfb45e">{nota_m:.2f}</div><div class="kpi-sub">Calificación SofaScore</div></div>
                        <div class="kpi-card"><div class="kpi-label">Recuperaciones / Quites</div><div class="kpi-value">{recup_t} · <span style="font-size:1.5rem; color:#8b8b95">({quites_t} quites)</span></div><div class="kpi-sub">Intercepciones: {int(dj['interceptionWon'].fillna(0).sum())}</div></div>
                        <div class="kpi-card"><div class="kpi-label">Duelos Aéreos / Despejes</div><div class="kpi-value">{int(dj['aerialWon'].fillna(0).sum())} · <span style="font-size:1.5rem; color:#8b8b95">({int(dj['totalClearance'].fillna(0).sum())} desp.)</span></div><div class="kpi-sub">Goles: {goles_t} · Asist: {asist_t}</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    xg_sub = f" · xG: {dj['expectedGoals'].fillna(0).sum():.2f}" if tiene_avanzadas else ""
                    xa_sub = f" · xA: {dj['expectedAssists'].fillna(0).sum():.2f}" if tiene_avanzadas else ""
                    st.markdown(f"""
                    <div class="kpi-grid">
                        <div class="kpi-card"><div class="kpi-label">Presencias · {pos_jugador}</div><div class="kpi-value">{pj} PJ · <span style="color:#ED1A3B">{tit} TIT</span></div><div class="kpi-sub">{mins} minutos jugados</div></div>
                        <div class="kpi-card"><div class="kpi-label">Nota Promedio</div><div class="kpi-value" style="color:#cfb45e">{nota_m:.2f}</div><div class="kpi-sub">Calificación SofaScore</div></div>
                        <div class="kpi-card"><div class="kpi-label">Goles</div><div class="kpi-value">{goles_t}</div><div class="kpi-sub">Cada 90': {(goles_t*90/mins if mins else 0):.2f}{xg_sub}</div></div>
                        <div class="kpi-card"><div class="kpi-label">Asistencias</div><div class="kpi-value">{asist_t}</div><div class="kpi-sub">Pases clave: {int(dj['keyPass'].fillna(0).sum())}{xa_sub}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

            # Evolución de nota
            st.markdown('<div class="section-header">Evolución de Rendimiento (Nota por Partido)</div>', unsafe_allow_html=True)
            multi_torneo_j = dj["torneo"].nunique() > 1 if "torneo" in dj.columns else False

            def _etiqueta_nota(r):
                pref = ""
                if multi_torneo_j:
                    t_str = str(r.get("torneo", "")).lower()
                    pref = "AP · " if "apertura" in t_str else ("CL · " if "clausura" in t_str else f"{str(r.get('torneo', ''))[:3].upper()} · ")
                inst = f"F{int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) else str(r.get("fase", "PO"))
                return f"{pref}{inst} vs {str(r['rival'])[:12]}"

            dj["Etiqueta"] = dj.apply(_etiqueta_nota, axis=1)
            fig_nota = go.Figure(go.Scatter(
                x=dj["Etiqueta"],
                y=dj["rating"].round(2),
                mode="lines+markers+text",
                text=dj["rating"].round(1),
                textposition="top center",
                line=dict(color=RED, width=3),
                marker=dict(size=9, color="#cfb45e", line=dict(width=1.5, color="#0a0a0c")),
                textfont=dict(size=10.5, color="#ececf0"),
                customdata=dj["minutesPlayed"].round(0).astype(int),
                hovertemplate="<b>%{x}</b><br>Nota: %{y:.2f}<br>Minutos: %{customdata}'<extra></extra>",
            ))
            fig_nota.add_hline(y=dj["rating"].mean(), line=dict(color=GRAY, dash="dash", width=1))
            fig_nota.update_layout(**PLOT_LAYOUT, height=320, yaxis=dict(range=[5.0, 10.2]))
            _estilizar_ejes(fig_nota)
            fig_nota.update_xaxes(
                tickangle=-35,
                type="category",
                categoryorder="array",
                categoryarray=dj["Etiqueta"].tolist(),
            )
            st.plotly_chart(fig_nota, width="stretch")

            # Perfil de Remates en español
            tiros_j = df_tiros[df_tiros["jugador"] == jug_sel].copy() if not df_tiros.empty and "jugador" in df_tiros.columns else pd.DataFrame()
            if not tiros_j.empty and tiene_avanzadas:
                st.markdown('<div class="section-header">Perfil de Definición y Remates</div>', unsafe_allow_html=True)
                ct1, ct2 = st.columns(2)
                with ct1:
                    tiros_j["Zona"] = (
                        tiros_j["zona_arco"].astype(str).str.lower()
                        .map(TRADUCCION_ZONA)
                        .fillna(tiros_j["zona_arco"].astype(str).str.capitalize())
                    )
                    df_z = tiros_j.groupby("Zona", as_index=False).agg(Remates=("xg", "count"), Goles=("es_gol", "sum"), xG=("xg", "sum"), xGOT=("xgot", "sum")).sort_values("Remates", ascending=False)
                    df_z["Goles"] = df_z["Goles"].astype(int)
                    st.dataframe(df_z.style.format({"xG": "{:.2f}", "xGOT": "{:.2f}"}), hide_index=True, width="stretch")
                with ct2:
                    tiros_j["Perfil"] = (
                        tiros_j["parte_cuerpo"].astype(str).str.lower()
                        .map(TRADUCCION_CUERPO)
                        .fillna(tiros_j["parte_cuerpo"].astype(str).str.capitalize())
                    )
                    df_c = tiros_j.groupby("Perfil", as_index=False).agg(Remates=("xg", "count"), Goles=("es_gol", "sum"), xG=("xg", "sum")).sort_values("Remates", ascending=False)
                    df_c["Goles"] = df_c["Goles"].astype(int)
                    st.dataframe(df_c.style.format({"xG": "{:.2f}"}), hide_index=True, width="stretch")

            # Registro partido a partido adaptado por posición
            st.markdown('<div class="section-header">Registro Partido a Partido</div>', unsafe_allow_html=True)
            dj["Instancia"] = dj.apply(lambda r: f"Fecha {int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) else str(r["fase"]), axis=1)

            base_log = ["fecha", "torneo", "Instancia", "equipo", "rival", "minutesPlayed", "rating"]
            if "arquero" in pos_jugador.lower():
                pos_cols = ["saves", "goalsConceded", "accuratePass"]
                if tiene_avanzadas:
                    pos_cols.insert(2, "goalsPrevented")
            elif "defensor" in pos_jugador.lower():
                pos_cols = ["wonTackle", "interceptionWon", "ballRecovery", "totalClearance", "duelWon", "aerialWon", "possessionLostCtrl", "goals", "goalAssist"]
            elif "mediocampista" in pos_jugador.lower():
                pos_cols = ["accuratePass", "keyPass", "goalAssist", "ballRecovery", "wonTackle", "wonContest", "possessionLostCtrl", "goals"]
                if tiene_avanzadas:
                    pos_cols.insert(3, "expectedAssists")
            else:
                pos_cols = ["goals", "onTargetScoringAttempt", "goalAssist", "keyPass", "wonContest"]
                if tiene_avanzadas:
                    pos_cols.insert(1, "expectedGoals")
                    pos_cols.insert(2, "expectedGoalsOnTarget")
                    pos_cols.insert(5, "expectedAssists")

            cols_log = [c for c in (base_log + pos_cols) if c in dj.columns]
            log_show = dj[cols_log].copy()

            cols_enteras_log = [
                "minutesPlayed", "saves", "goalsConceded", "accuratePass",
                "wonTackle", "interceptionWon", "ballRecovery", "totalClearance",
                "duelWon", "aerialWon", "possessionLostCtrl", "goals", "goalAssist",
                "keyPass", "wonContest", "onTargetScoringAttempt",
            ]
            for ce in cols_enteras_log:
                if ce in log_show.columns:
                    log_show[ce] = log_show[ce].fillna(0).round(0).astype(int)

            ren_log = {
                "fecha": "Fecha", "torneo": "Torneo", "equipo": "Club", "rival": "Rival",
                "minutesPlayed": "Min'", "rating": "Nota", "goals": "Goles",
                "expectedGoals": "xG", "expectedGoalsOnTarget": "xGOT",
                "goalAssist": "Asist.", "expectedAssists": "xA",
                "keyPass": "Pases Clave", "ballRecovery": "Recup.", "wonTackle": "Quites",
                "interceptionWon": "Intercep.", "totalClearance": "Despejes",
                "duelWon": "Duelos Gan.", "aerialWon": "Aéreos Gan.", "accuratePass": "Pases Ok",
                "wonContest": "Regates", "possessionLostCtrl": "Pérdidas",
                "onTargetScoringAttempt": "Al Arco", "saves": "Atajadas",
                "goalsConceded": "Goles Rec.", "goalsPrevented": "Goles Evit.",
            }
            log_show = log_show.rename(columns=ren_log)
            fmt_log = {"Nota": "{:.2f}"}
            for dec_c in ["xG", "xGOT", "xA", "Goles Evit."]:
                if dec_c in log_show.columns:
                    fmt_log[dec_c] = "{:.2f}"

            log_show_tbl = agregar_columna_escudo(log_show.iloc[::-1].reset_index(drop=True), col_equipo="Rival")
            st.dataframe(
                log_show_tbl.style.format(fmt_log),
                column_config={"Escudo": config_columna_escudo("")},
                hide_index=True,
                width="stretch",
            )

    # =========================================================================
    # VISTA 3: RENDIMIENTO POR PARTIDO Y MAPA DE TIROS
    # =========================================================================
    elif sub_vista == "🎯 Rendimiento por Partido":
        _render_rendimiento_partido(df_jug, df_tiros, tiene_avanzadas)

    # =========================================================================
    # VISTA 4: COMPARADOR
    # =========================================================================
    elif sub_vista == "⚔️ Comparador":
        df_base_comp = _agregar_jugadores(df_jug)
        df_base_comp = df_base_comp[df_base_comp["Minutos"] >= 90].copy()

        pos_comp = st.selectbox(
            "Posición a comparar (adapta los ejes del radar y métricas clave)",
            sorted(df_base_comp["Posicion"].unique()),
            key="comp_jug_pos",
        )
        pool_comp = df_base_comp[df_base_comp["Posicion"] == pos_comp].copy()
        pool_comp["Label"] = pool_comp["Jugador"] + " (" + pool_comp["Equipo"] + ")"
        opciones_j = sorted(pool_comp["Label"].tolist())

        sel_jugs = st.multiselect("Elegí 2 o más jugadores", opciones_j, default=opciones_j[:2] if len(opciones_j) >= 2 else opciones_j)

        if len(sel_jugs) < 2:
            st.info("Seleccioná al menos 2 jugadores para activar la comparativa.")
        else:
            badges_j = []
            for l in sel_jugs:
                rj_b = pool_comp[pool_comp["Label"] == l].iloc[0]
                eq_b = _club_actual(rj_b["Equipo"])
                badges_j.append(
                    f'<div style="display:inline-flex; align-items:center; gap:8px; background:#17171c; border:1px solid #26262d; padding:6px 14px; border-radius:999px;">'
                    f'{img_escudo_html(eq_b, 24)}<span style="font-weight:700; color:#ffffff; font-size:0.86rem;">{rj_b["Jugador"]}</span></div>'
                )
            st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px 0;">{"".join(badges_j)}</div>', unsafe_allow_html=True)

            pos_low = pos_comp.lower()
            if "arquero" in pos_low:
                metricas_radar = [
                    ("Atajadas / 90'", "Atajadas"),
                    ("Salidas / 90'", "Salidas_Arq"),
                    ("Pases Ok / 90'", "Pases_Ok"),
                    ("Aéreos / 90'", "Aereos_Ok"),
                ]
                if tiene_avanzadas:
                    metricas_radar.insert(1, ("Goles Evitados / 90'", "Goles_Evit"))
            elif "defensor" in pos_low:
                metricas_radar = [
                    ("Quites / 90'", "Quites_Ok"),
                    ("Intercepciones / 90'", "Intercep"),
                    ("Recuperaciones / 90'", "Recup"),
                    ("Despejes / 90'", "Despejes"),
                    ("Duelos Gan. / 90'", "Duelos_Ok"),
                    ("Aéreos Gan. / 90'", "Aereos_Ok"),
                ]
            elif "mediocampista" in pos_low:
                metricas_radar = [
                    ("Pases Clave / 90'", "Pases_Clave"),
                    ("Asistencias / 90'", "Asist"),
                    ("Regates / 90'", "Regates_Ok"),
                    ("Recuperaciones / 90'", "Recup"),
                    ("Quites / 90'", "Quites_Ok"),
                    ("Duelos Gan. / 90'", "Duelos_Ok"),
                ]
                if tiene_avanzadas:
                    metricas_radar.insert(2, ("xA / 90'", "xA"))
                    metricas_radar.insert(3, ("Cond. Prog. / 90'", "Cond_Prog"))
            else:
                metricas_radar = [
                    ("Goles / 90'", "Goles"),
                    ("Tiros al Arco / 90'", "Al_Arco"),
                    ("Asistencias / 90'", "Asist"),
                    ("Pases Clave / 90'", "Pases_Clave"),
                    ("Regates / 90'", "Regates_Ok"),
                    ("Aéreos Gan. / 90'", "Aereos_Ok"),
                ]
                if tiene_avanzadas:
                    metricas_radar.insert(1, ("xG / 90'", "xG"))
                    metricas_radar.insert(2, ("xGOT / 90'", "xGOT"))
                    metricas_radar.insert(5, ("xA / 90'", "xA"))

            pool_p90 = pool_comp.copy()
            for _, col in metricas_radar:
                if col in pool_p90.columns:
                    pool_p90[f"{col}_p90"] = pool_p90[col] * pool_p90["Factor_90"]

            col_rad, col_tbl = st.columns([1.2, 1])

            with col_rad:
                st.markdown('<div class="section-header">Radar de Percentiles (Cada 90\')</div>', unsafe_allow_html=True)
                categorias_rad = [m[0] for m in metricas_radar if m[1] in pool_p90.columns]
                fig_rad = go.Figure()

                for idx, lbl in enumerate(sel_jugs):
                    row_j = pool_p90[pool_p90["Label"] == lbl].iloc[0]
                    vals_pct = []
                    for _, col in metricas_radar:
                        if col not in pool_p90.columns:
                            continue
                        serie = pool_p90[f"{col}_p90"]
                        val_j = row_j[f"{col}_p90"]
                        pct = float((serie <= val_j).mean() * 100) if len(serie) > 1 else 50.0
                        vals_pct.append(round(pct, 1))

                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals_pct + [vals_pct[0]],
                        theta=categorias_rad + [categorias_rad[0]],
                        fill="toself",
                        name=row_j["Jugador"],
                        line=dict(color=PALETTE[idx % len(PALETTE)], width=2.5),
                    ))

                fig_rad.update_layout(
                    **PLOT_LAYOUT,
                    height=430,
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID, tickfont=dict(size=9, color=AXIS_TXT)),
                        angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10.5, color="#ececf0")),
                    ),
                    legend=dict(orientation="h", y=1.15, x=0),
                )
                st.plotly_chart(fig_rad, width="stretch")

            with col_tbl:
                st.markdown('<div class="section-header">Valores Reales (Cada 90\')</div>', unsafe_allow_html=True)
                filas_c = [
                    {"Métrica": "Minutos jugados", **{pool_p90[pool_p90["Label"] == l].iloc[0]["Jugador"]: int(pool_p90[pool_p90["Label"] == l].iloc[0]["Minutos"]) for l in sel_jugs}},
                    {"Métrica": "Nota promedio", **{pool_p90[pool_p90["Label"] == l].iloc[0]["Jugador"]: round(float(pool_p90[pool_p90["Label"] == l].iloc[0]["Nota"]), 2) for l in sel_jugs}},
                ]
                for label_m, col in metricas_radar:
                    if f"{col}_p90" not in pool_p90.columns:
                        continue
                    fila = {"Métrica": label_m}
                    for l in sel_jugs:
                        rj = pool_p90[pool_p90["Label"] == l].iloc[0]
                        fila[rj["Jugador"]] = round(float(rj[f"{col}_p90"]), 2)
                    filas_c.append(fila)

                df_c = pd.DataFrame(filas_c)
                st.dataframe(df_c, hide_index=True, width="stretch", height=(len(df_c) * 36) + 42)

    # =========================================================================
    # VISTA 5: LÍDERES POR FECHA
    # =========================================================================
    else:
        df_vf = df_jug[df_jug["minutesPlayed"].fillna(0) > 0].copy()
        df_vf["Instancia_Label"] = df_vf.apply(
            lambda r: f"{r['torneo']} · Fecha {int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) else f"{r['torneo']} · {r['fase']}",
            axis=1,
        )
        orden_inst = (
            df_vf.groupby("Instancia_Label", as_index=False)["fecha_dt"]
            .max()
            .sort_values("fecha_dt", ascending=False)["Instancia_Label"]
            .tolist()
        )

        inst_sel = st.selectbox("Seleccioná la Fecha / Instancia", orden_inst, key="lideres_fecha_sel")
        df_f_act = df_vf[df_vf["Instancia_Label"] == inst_sel].copy()
        df_f_act["Min"] = df_f_act["minutesPlayed"].fillna(0).round(0).astype(int)
        df_f_act["Tiros"] = np.where(
            df_f_act["totalShots"].notna(),
            df_f_act["totalShots"],
            df_f_act["onTargetScoringAttempt"].fillna(0) + df_f_act["shotOffTarget"].fillna(0) + df_f_act["blockedScoringAttempt"].fillna(0),
        )

        pases_tot_f = df_f_act["totalPass"].fillna(0)
        df_f_act["Pases_Pct"] = np.where(pases_tot_f > 0, (df_f_act["accuratePass"].fillna(0) / pases_tot_f) * 100, 0.0)
        duelos_tot_f = df_f_act["duelWon"].fillna(0) + df_f_act["duelLost"].fillna(0)
        df_f_act["Duelos_Pct"] = np.where(duelos_tot_f > 0, (df_f_act["duelWon"].fillna(0) / duelos_tot_f) * 100, 0.0)

        tiene_xg_fecha = df_f_act["expectedGoals"].fillna(0).sum() > 0

        def _tabla_top_posicion(sub_pos: pd.DataFrame, cols_extra: list, renombres_extra: dict, cols_decimales: list, cols_pct: list):
            if sub_pos.empty:
                return None
            top = sub_pos.sort_values(by=["rating", "Min"], ascending=[False, False]).reset_index(drop=True)
            top.insert(0, "#", top.index + 1)
            cols_usar = ["#", "jugador", "equipo", "Min", "rating"] + [c for c in cols_extra if c in top.columns]
            out = top[cols_usar].copy()
            out = out.rename(columns={"jugador": "Jugador", "equipo": "Club", "Min": "Min'", "rating": "Nota", **renombres_extra})
            out = agregar_columna_escudo(out, col_equipo="Club")

            fmt_pos = {"Nota": "{:.2f}"}
            for col_name in out.columns:
                if col_name in ["#", "Jugador", "Escudo", "Club", "Min'", "Nota"]:
                    continue
                if col_name in cols_pct:
                    fmt_pos[col_name] = "{:.1f}%"
                elif col_name in cols_decimales:
                    fmt_pos[col_name] = "{:+.2f}" if col_name == "Goles Evit." else "{:.2f}"
                else:
                    out[col_name] = out[col_name].fillna(0).round(0).astype(int)

            alto_visible = min(392, (len(out) * 35) + 42)
            st.dataframe(
                out.style.format(fmt_pos)
                .set_properties(subset=["#"], **{"color": RED, "font-weight": "bold"})
                .set_properties(subset=["Jugador"], **{"font-weight": "bold"})
                .set_properties(subset=["Nota"], **{"color": "#cfb45e", "font-weight": "bold"}),
                column_config={"Escudo": config_columna_escudo("")},
                hide_index=True,
                width="stretch",
                height=alto_visible,
            )

        st.markdown('<div class="section-header">🧤 Arqueros de la Fecha</div>', unsafe_allow_html=True)
        df_arq = df_f_act[df_f_act["posicion_desc"].str.contains("arquero", case=False, na=False)]
        cols_arq = ["saves", "goalsConceded"]
        if tiene_xg_fecha:
            cols_arq.append("goalsPrevented")
        _tabla_top_posicion(
            df_arq,
            cols_arq,
            {"saves": "Atajadas", "goalsConceded": "Goles Rec.", "goalsPrevented": "Goles Evit."},
            ["Goles Evit."],
            [],
        )

        st.markdown('<div class="section-header">🛡️ Defensores de la Fecha</div>', unsafe_allow_html=True)
        df_def = df_f_act[df_f_act["posicion_desc"].str.contains("defensor", case=False, na=False)]
        _tabla_top_posicion(
            df_def,
            ["goals", "goalAssist", "wonTackle", "interceptionWon", "ballRecovery", "totalClearance", "aerialWon", "Duelos_Pct", "Pases_Pct", "possessionLostCtrl"],
            {
                "goals": "Goles", "goalAssist": "Asist.", "wonTackle": "Quites",
                "interceptionWon": "Intercep.", "ballRecovery": "Recup.",
                "totalClearance": "Despejes", "aerialWon": "Aéreos",
                "Duelos_Pct": "Duelos %", "Pases_Pct": "Pases %", "possessionLostCtrl": "Pérdidas",
            },
            [],
            ["Duelos %", "Pases %"],
        )

        st.markdown('<div class="section-header">🎯 Mediocampistas de la Fecha</div>', unsafe_allow_html=True)
        df_med = df_f_act[df_f_act["posicion_desc"].str.contains("mediocampista", case=False, na=False)]
        cols_med = ["goals", "goalAssist", "keyPass", "Pases_Pct", "ballRecovery", "wonTackle", "wonContest", "possessionLostCtrl"]
        if tiene_xg_fecha:
            cols_med.insert(2, "expectedAssists")
        _tabla_top_posicion(
            df_med,
            cols_med,
            {
                "goals": "Goles", "goalAssist": "Asist.", "expectedAssists": "xA",
                "keyPass": "Pases Clave", "Pases_Pct": "Pases %", "ballRecovery": "Recup.",
                "wonTackle": "Quites", "wonContest": "Regates", "possessionLostCtrl": "Pérdidas",
            },
            ["xA"],
            ["Pases %"],
        )

        st.markdown('<div class="section-header">⚽ Delanteros de la Fecha</div>', unsafe_allow_html=True)
        df_del = df_f_act[df_f_act["posicion_desc"].str.contains("delantero", case=False, na=False)]
        cols_del = ["goals", "goalAssist", "Tiros", "onTargetScoringAttempt", "keyPass", "wonContest"]
        if tiene_xg_fecha:
            cols_del.insert(1, "expectedGoals")
            cols_del.insert(2, "expectedGoalsOnTarget")
            cols_del.insert(4, "expectedAssists")
        _tabla_top_posicion(
            df_del,
            cols_del,
            {
                "goals": "Goles", "expectedGoals": "xG", "expectedGoalsOnTarget": "xGOT",
                "goalAssist": "Asist.", "expectedAssists": "xA",
                "onTargetScoringAttempt": "Al Arco", "keyPass": "Pases Clave", "wonContest": "Regates",
            },
            ["xG", "xGOT", "xA"],
            [],
        )