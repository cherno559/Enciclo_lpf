from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from escudos_helper import (
    agregar_columna_escudo,
    config_columna_escudo,
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

TRADUCCION_SITUACION = {
    "assisted": "Jugada elaborada",
    "open-play": "Jugada abierta",
    "regular": "Jugada abierta",
    "set-piece": "Pelota parada",
    "corner": "Córner",
    "free-kick": "Tiro libre",
    "penalty": "Penal",
    "fast-break": "Contraataque",
    "throw-in-set-piece": "Saque lateral",
}

POS_ABR = {
    "Arquero": "ARQ",
    "Defensor": "DEF",
    "Mediocampista": "MED",
    "Delantero": "DEL",
}


def _estilizar_ejes(fig, grid_x=False, grid_y=True):
    base = dict(zeroline=False, showline=True, linecolor=AXIS_LINE, tickfont=dict(size=11, color=AXIS_TXT))
    fig.update_xaxes(showgrid=grid_x, gridcolor=GRID, **base)
    fig.update_yaxes(showgrid=grid_y, gridcolor=GRID, **base)
    return fig


@st.cache_data(show_spinner=False)
def _cargar_tiros_fallback() -> pd.DataFrame:
    ruta = Path(__file__).resolve().parents[1] / "datos" / "tiros.parquet"
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_parquet(ruta)


def _preparar_equipos(df_eq: pd.DataFrame) -> pd.DataFrame:
    """Cruza cada partido con los datos del rival para tener métricas en contra."""
    cols_rival = [
        "expected_goals", "ball_possession", "total_shots_on_goal",
        "shots_on_goal", "big_chance_created", "touches_in_opp_box",
        "pct_pases", "duel_won_percent", "corner_kicks", "fouls",
        "goalkeeper_saves", "ball_recovery",
    ]
    disp = [c for c in cols_rival if c in df_eq.columns]
    rival = df_eq[["match_id", "equipo"] + disp].rename(
        columns={"equipo": "rival", **{c: f"{c}_contra" for c in disp}}
    )
    return df_eq.merge(rival, on=["match_id", "rival"], how="left")


def _resumen_goles_asistencias(mid, equipo: str, df_jug: pd.DataFrame, df_tiros: pd.DataFrame) -> tuple[str, str]:
    """Devuelve dos strings formateados: (goleadores_txt, asistidores_txt) para un equipo en un partido."""
    jug_m = df_jug[(df_jug["match_id"] == mid) & (df_jug["equipo"] == equipo)].copy() if not df_jug.empty else pd.DataFrame()

    # ── Goleadores ──
    mins_por_jug: dict[str, list[str]] = {}
    if not df_tiros.empty and "match_id" in df_tiros.columns:
        t_m = df_tiros[df_tiros["match_id"] == mid].copy()
        if "equipo" in t_m.columns:
            t_m = t_m[t_m["equipo"] == equipo]
        col_j_t = next((c for c in ["jugador", "player", "player_name"] if c in t_m.columns), None)
        if col_j_t and not t_m.empty:
            es_gol = (
                (t_m["es_gol"].fillna(0).astype(int) == 1)
                if "es_gol" in t_m.columns
                else (t_m["resultado"].astype(str).str.lower() == "goal" if "resultado" in t_m.columns else pd.Series(False, index=t_m.index))
            )
            goles_t = t_m[es_gol].copy()
            if not goles_t.empty:
                if "minuto" in goles_t.columns:
                    goles_t["_min_num"] = pd.to_numeric(goles_t["minuto"], errors="coerce")
                    goles_t = goles_t.sort_values("_min_num")
                for _, r in goles_t.iterrows():
                    nom = str(r[col_j_t])
                    m_txt = f"{int(r['_min_num'])}'" if pd.notna(r.get("_min_num")) else ""
                    mins_por_jug.setdefault(nom, []).append(m_txt)

    goleadores_lista = []
    vistos = set()
    for nom, lista_m in mins_por_jug.items():
        vistos.add(nom)
        mins_limpios = [m for m in lista_m if m]
        if mins_limpios:
            goleadores_lista.append(f"{nom} ({', '.join(mins_limpios)})")
        elif len(lista_m) > 1:
            goleadores_lista.append(f"{nom} (x{len(lista_m)})")
        else:
            goleadores_lista.append(nom)

    if not jug_m.empty and "goals" in jug_m.columns:
        g_df = jug_m[jug_m["goals"].fillna(0) > 0].sort_values("goals", ascending=False)
        for _, r in g_df.iterrows():
            nom = str(r["jugador"])
            if nom in vistos:
                continue
            cant = int(round(r["goals"]))
            goleadores_lista.append(f"{nom} (x{cant})" if cant > 1 else nom)

    # ── Asistidores ──
    asist_lista = []
    if not jug_m.empty and "goalAssist" in jug_m.columns:
        a_df = jug_m[jug_m["goalAssist"].fillna(0) > 0].sort_values(["goalAssist", "rating"], ascending=[False, False])
        for _, r in a_df.iterrows():
            nom = str(r["jugador"])
            cant = int(round(r["goalAssist"]))
            asist_lista.append(f"{nom} (x{cant})" if cant > 1 else nom)

    goles_txt = " · ".join(goleadores_lista)
    asist_txt = " · ".join(asist_lista)
    return goles_txt, asist_txt


def _render_plantel_partido(mid, equipo: str, df_jug: pd.DataFrame, tiene_xg: bool):
    sub = df_jug[(df_jug["match_id"] == mid) & (df_jug["equipo"] == equipo)].copy()
    sub = sub[sub["minutesPlayed"].fillna(0) > 0].copy()

    form_txt = ""
    if "formacion" in sub.columns and sub["formacion"].dropna().any():
        form_txt = f" · Esquema {sub['formacion'].dropna().iloc[0]}"

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin:12px 0 8px 0;">
        {img_escudo_html(equipo, 28)}
        <div style="font-family:'Oswald',sans-serif; font-size:1.15rem; color:#ffffff; letter-spacing:0.6px;">
            {equipo.upper()}<span style="font-family:'Manrope',sans-serif; font-size:0.78rem; color:#8b8b95; font-weight:600;">{form_txt}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if sub.empty:
        st.caption("Sin datos individuales de jugadores para este partido.")
        return

    sub["_es_sup"] = sub["suplente"].fillna(True).astype(bool)
    sub = sub.sort_values(["_es_sup", "minutesPlayed", "rating"], ascending=[True, False, False]).reset_index(drop=True)

    pos_raw = sub["posicion_desc"].fillna("-") if "posicion_desc" in sub.columns else pd.Series("-", index=sub.index)
    pos_corta = pos_raw.map(POS_ABR).fillna(pos_raw.astype(str).str[:3].str.upper())
    rol_txt = np.where(sub["_es_sup"], "SUP · " + pos_corta, "TIT · " + pos_corta)

    pases_ok = sub["accuratePass"].fillna(0).round(0).astype(int) if "accuratePass" in sub.columns else 0
    pases_tot = sub["totalPass"].fillna(0).round(0).astype(int) if "totalPass" in sub.columns else 0

    data = {
        "Rol": rol_txt,
        "Jugador": sub["jugador"],
        "Min'": sub["minutesPlayed"].fillna(0).round(0).astype(int),
        "Nota": sub["rating"],
        "G": sub["goals"].fillna(0).round(0).astype(int) if "goals" in sub.columns else 0,
        "A": sub["goalAssist"].fillna(0).round(0).astype(int) if "goalAssist" in sub.columns else 0,
    }
    if tiene_xg and "expectedGoals" in sub.columns:
        data["xG"] = sub["expectedGoals"].fillna(0.0)

    data["Pases"] = [f"{ok}/{tot}" for ok, tot in zip(pases_ok, pases_tot)]
    data["Quites"] = sub["wonTackle"].fillna(0).round(0).astype(int) if "wonTackle" in sub.columns else 0
    data["Recup."] = sub["ballRecovery"].fillna(0).round(0).astype(int) if "ballRecovery" in sub.columns else 0
    data["Ataj."] = sub["saves"].fillna(0).round(0).astype(int) if "saves" in sub.columns else 0

    df_pl = pd.DataFrame(data)
    fmt = {"Nota": "{:.2f}"}
    if "xG" in df_pl.columns:
        fmt["xG"] = "{:.2f}"

    st.dataframe(
        df_pl.style.format(fmt, na_rep="-")
        .set_properties(subset=["Jugador"], **{"font-weight": "bold"})
        .set_properties(subset=["Nota"], **{"color": "#cfb45e", "font-weight": "bold"})
        .set_properties(subset=["Rol"], **{"color": "#8b8b95", "font-size": "0.8rem"}),
        hide_index=True,
        use_container_width=True,
        height=(len(df_pl) * 35) + 42,
    )


def render_equipos(df_eq: pd.DataFrame, df_jug: pd.DataFrame, df_tiros: pd.DataFrame, periodo: str = "Temporada Completa"):
    if df_eq.empty:
        st.warning("No hay datos de equipos para el período seleccionado.")
        return

    df = _preparar_equipos(df_eq)
    equipos_lista = sorted(df["equipo"].dropna().unique())
    solo_clausura = (periodo == "Torneo Clausura") and (df["expected_goals"].fillna(0).sum() > 0)

    tab_liga, tab_ficha, tab_comp, tab_partidos = st.tabs([
        "📊 Tablas de Liga",
        "🔎 Ficha de Club",
        "⚔️ Comparador",
        "🔍 Buscador de Partidos",
    ])

    # =========================================================================
    # TAB 1: TABLAS DE LIGA
    # =========================================================================
    with tab_liga:
        c1, c2 = st.columns([2, 2])
        with c1:
            cond_liga = st.radio("Condición", ["General", "Local", "Visitante"], horizontal=True, key="eq_cond_liga")
        with c2:
            modo_liga = st.radio("Valores", ["Promedio por Partido", "Totales Acumulados"], horizontal=True, key="eq_modo_liga")

        d_l = df if cond_liga == "General" else df[df["condicion"].str.capitalize() == cond_liga]
        es_prom = modo_liga == "Promedio por Partido"
        agg_fn = "mean" if es_prom else "sum"

        if solo_clausura:
            resumen = d_l.groupby("equipo", as_index=False).agg(
                PJ=("match_id", "count"),
                PTS=("puntos", "sum"),
                GF=("goles_favor", agg_fn),
                GC=("goles_contra", agg_fn),
                xG=("expected_goals", agg_fn),
                xGA=("expected_goals_contra", agg_fn),
                Pos=("ball_possession", "mean"),
                Tiros=("total_shots_on_goal", agg_fn),
                Al_Arco=("shots_on_goal", agg_fn),
                Oc_Claras=("big_chance_created", agg_fn),
                Toques_Area=("touches_in_opp_box", agg_fn),
                Pases_Pct=("pct_pases", "mean"),
                Duelos_Pct=("duel_won_percent", "mean"),
            )
            resumen = resumen.sort_values(["PTS", "GF"], ascending=[False, False]).reset_index(drop=True)
            resumen.insert(0, "#", resumen.index + 1)
            resumen = resumen.rename(columns={
                "equipo": "Equipo", "Pos": "Pos %", "Al_Arco": "Al Arco",
                "Oc_Claras": "Oc. Claras", "Toques_Area": "Toques Área",
                "Pases_Pct": "Pases %", "Duelos_Pct": "Duelos %",
            })

            fmt = {"xG": "{:.2f}", "xGA": "{:.2f}", "Pos %": "{:.1f}%", "Pases %": "{:.1f}%", "Duelos %": "{:.1f}%"}
            if es_prom:
                fmt.update({"GF": "{:.2f}", "GC": "{:.2f}", "Tiros": "{:.1f}", "Al Arco": "{:.1f}", "Oc. Claras": "{:.1f}", "Toques Área": "{:.1f}"})
            else:
                for col_int in ["GF", "GC", "Tiros", "Al Arco", "Oc. Claras", "Toques Área"]:
                    resumen[col_int] = resumen[col_int].fillna(0).round(0).astype(int)

            metricas_practicas = ["xG", "xGA", "GF", "GC", "Pos %", "Tiros", "Al Arco", "Oc. Claras", "Toques Área", "Pases %", "Duelos %"]
            def_x, def_y = "xG", "xGA"
        else:
            resumen = d_l.groupby("equipo", as_index=False).agg(
                PJ=("match_id", "count"),
                PTS=("puntos", "sum"),
                GF=("goles_favor", agg_fn),
                GC=("goles_contra", agg_fn),
                DG=("diferencia_gol", "sum"),
                Pos=("ball_possession", "mean"),
                Tiros=("total_shots_on_goal", agg_fn),
                Al_Arco=("shots_on_goal", agg_fn),
                Corners=("corner_kicks", agg_fn),
                Atajadas=("goalkeeper_saves", agg_fn),
            )
            resumen = resumen.sort_values(["PTS", "DG", "GF"], ascending=[False, False, False]).reset_index(drop=True)
            resumen.insert(0, "#", resumen.index + 1)
            resumen["Efec %"] = (resumen["PTS"] / (resumen["PJ"] * 3) * 100).round(1)
            resumen = resumen.rename(columns={
                "equipo": "Equipo", "Pos": "Pos %", "Al_Arco": "Al Arco", "Corners": "Córners",
            })
            cols_ord = ["#", "Equipo", "PJ", "PTS", "Efec %", "GF", "GC", "DG", "Pos %", "Tiros", "Al Arco", "Córners", "Atajadas"]
            resumen = resumen[cols_ord]

            fmt = {"Efec %": "{:.1f}%", "Pos %": "{:.1f}%"}
            if es_prom:
                fmt.update({"GF": "{:.2f}", "GC": "{:.2f}", "Tiros": "{:.1f}", "Al Arco": "{:.1f}", "Córners": "{:.1f}", "Atajadas": "{:.1f}"})
            else:
                for col_int in ["GF", "GC", "Tiros", "Al Arco", "Córners", "Atajadas"]:
                    resumen[col_int] = resumen[col_int].fillna(0).round(0).astype(int)

            metricas_practicas = ["GF", "GC", "Pos %", "Tiros", "Al Arco", "Córners", "Atajadas", "Efec %"]
            def_x, def_y = "GF", "GC"

        resumen_tbl = agregar_columna_escudo(resumen, col_equipo="Equipo")
        st.dataframe(
            resumen_tbl.style.format(fmt)
            .set_properties(subset=["#"], **{"color": RED, "font-weight": "bold"})
            .set_properties(subset=["Equipo"], **{"font-weight": "bold"})
            .set_properties(subset=["PTS"], **{"background-color": "#261217", "color": "#ffffff", "font-weight": "bold"}),
            column_config={"Escudo": config_columna_escudo("")},
            hide_index=True,
            use_container_width=True,
            height=(len(resumen_tbl) * 35) + 42,
        )

        # Explorador visual
        st.markdown('<div class="section-header">Explorador Visual (Eje X vs Eje Y)</div>', unsafe_allow_html=True)
        cx, cy = st.columns(2)
        with cx:
            eje_x = st.selectbox("Eje Horizontal (X)", metricas_practicas, index=metricas_practicas.index(def_x))
        with cy:
            eje_y = st.selectbox("Eje Vertical (Y)", metricas_practicas, index=metricas_practicas.index(def_y))

        fig_exp = go.Figure(go.Scatter(
            x=resumen[eje_x],
            y=resumen[eje_y],
            mode="markers+text",
            text=resumen["Equipo"],
            textposition="top center",
            marker=dict(size=12, color=RED, line=dict(width=1.5, color="#0a0a0c")),
            textfont=dict(size=10.5, color="#d4d4da"),
        ))
        fig_exp.add_vline(x=resumen[eje_x].mean(), line=dict(color=GRAY, dash="dash", width=1))
        fig_exp.add_hline(y=resumen[eje_y].mean(), line=dict(color=GRAY, dash="dash", width=1))
        fig_exp.update_layout(**PLOT_LAYOUT, height=470, xaxis_title=eje_x, yaxis_title=eje_y)
        _estilizar_ejes(fig_exp, grid_x=True, grid_y=True)
        if eje_y in ["xGA", "GC"]:
            fig_exp.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_exp, use_container_width=True)

    # =========================================================================
    # TAB 2: FICHA DE CLUB
    # =========================================================================
    with tab_ficha:
        col_sel_c, col_esc_c = st.columns([2.2, 1.8])
        with col_sel_c:
            club = st.selectbox(
                "Seleccioná un Club",
                equipos_lista,
                index=equipos_lista.index("River Plate") if "River Plate" in equipos_lista else 0,
            )
        with col_esc_c:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:14px; padding-top:14px;">
                {img_escudo_html(club, 52)}
                <div>
                    <div style="font-family:'Oswald',sans-serif; font-size:1.75rem; color:#ffffff; line-height:1.05; letter-spacing:1px;">
                        {club.upper()}
                    </div>
                    <div style="font-size:0.78rem; color:#8b8b95;">{periodo}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        d_club = df[df["equipo"] == club].sort_values("fecha_dt").reset_index(drop=True)

        pj = len(d_club)
        pts = int(d_club["puntos"].sum())
        efec = (pts / (pj * 3) * 100) if pj > 0 else 0.0
        pos_pj = d_club["ball_possession"].mean()
        vallas = int(d_club["valla_invicta"].sum())

        if solo_clausura:
            xg_pj = d_club["expected_goals"].mean()
            xga_pj = d_club["expected_goals_contra"].mean()
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Partidos / Puntos</div><div class="kpi-value">{pj} PJ · <span style="color:#ED1A3B">{pts} PTS</span></div><div class="kpi-sub">Efectividad: {efec:.1f}%</div></div>
                <div class="kpi-card"><div class="kpi-label">xG Favor / Partido</div><div class="kpi-value">{xg_pj:.2f}</div><div class="kpi-sub">Goles reales/pj: {d_club['goles_favor'].mean():.2f}</div></div>
                <div class="kpi-card"><div class="kpi-label">xG Contra / Partido</div><div class="kpi-value">{xga_pj:.2f}</div><div class="kpi-sub">Goles recibidos/pj: {d_club['goles_contra'].mean():.2f}</div></div>
                <div class="kpi-card"><div class="kpi-label">Posesión Media</div><div class="kpi-value">{pos_pj:.1f}%</div><div class="kpi-sub">Precisión pases: {d_club['pct_pases'].mean():.1f}%</div></div>
                <div class="kpi-card"><div class="kpi-label">Vallas Invictas</div><div class="kpi-value">{vallas}</div><div class="kpi-sub">{(vallas/pj*100 if pj else 0):.0f}% de sus partidos</div></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            dg_tot = int(d_club["diferencia_gol"].sum())
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Partidos / Puntos</div><div class="kpi-value">{pj} PJ · <span style="color:#ED1A3B">{pts} PTS</span></div><div class="kpi-sub">Efectividad: {efec:.1f}%</div></div>
                <div class="kpi-card"><div class="kpi-label">Goles Favor / Partido</div><div class="kpi-value">{d_club['goles_favor'].mean():.2f}</div><div class="kpi-sub">Total: {int(d_club['goles_favor'].sum())} goles</div></div>
                <div class="kpi-card"><div class="kpi-label">Goles Contra / Partido</div><div class="kpi-value">{d_club['goles_contra'].mean():.2f}</div><div class="kpi-sub">Dif. de Gol: {dg_tot:+d}</div></div>
                <div class="kpi-card"><div class="kpi-label">Posesión Media</div><div class="kpi-value">{pos_pj:.1f}%</div><div class="kpi-sub">Tiros al arco/pj: {d_club['shots_on_goal'].mean():.1f}</div></div>
                <div class="kpi-card"><div class="kpi-label">Vallas Invictas</div><div class="kpi-value">{vallas}</div><div class="kpi-sub">{(vallas/pj*100 if pj else 0):.0f}% de sus partidos</div></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Evolución Partido a Partido ──
        st.markdown('<div class="section-header">Evolución Partido a Partido</div>', unsafe_allow_html=True)
        multi_torneo = d_club["torneo"].nunique() > 1 if "torneo" in d_club.columns else False

        def _etiqueta_partido(r):
            pref = ""
            if multi_torneo:
                t_str = str(r.get("torneo", "")).lower()
                pref = "AP · " if "apertura" in t_str else ("CL · " if "clausura" in t_str else f"{str(r.get('torneo', ''))[:3].upper()} · ")
            inst = f"F{int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) else str(r.get("ronda_nombre", "PO"))
            return f"{pref}{inst} vs {str(r['rival'])[:12]}"

        d_club["Etiqueta_Partido"] = d_club.apply(_etiqueta_partido, axis=1)
        opciones_evo = (
            ["xG a Favor vs xG en Contra", "Goles a Favor vs Goles en Contra", "Posesión (%)"]
            if solo_clausura
            else ["Goles a Favor vs Goles en Contra", "Posesión (%)", "Tiros a Favor"]
        )
        modo_evo = st.radio("Métrica a seguir", opciones_evo, horizontal=True, key="evo_club_radio")

        fig_evo = go.Figure()
        if modo_evo == "xG a Favor vs xG en Contra":
            fig_evo.add_trace(go.Scatter(
                x=d_club["Etiqueta_Partido"], y=d_club["expected_goals"].round(2),
                name="xG a Favor", mode="lines+markers", line=dict(color=RED, width=3), marker=dict(size=8),
            ))
            fig_evo.add_trace(go.Scatter(
                x=d_club["Etiqueta_Partido"], y=d_club["expected_goals_contra"].round(2),
                name="xG en Contra", mode="lines+markers", line=dict(color="#6b7280", width=2, dash="dot"), marker=dict(size=7),
            ))
        elif modo_evo == "Goles a Favor vs Goles en Contra":
            fig_evo.add_trace(go.Bar(x=d_club["Etiqueta_Partido"], y=d_club["goles_favor"], name="Goles a Favor", marker_color=RED))
            fig_evo.add_trace(go.Bar(x=d_club["Etiqueta_Partido"], y=d_club["goles_contra"], name="Goles en Contra", marker_color="#4b5563"))
            fig_evo.update_layout(barmode="group")
        elif modo_evo == "Tiros a Favor":
            fig_evo.add_trace(go.Bar(
                x=d_club["Etiqueta_Partido"], y=d_club["total_shots_on_goal"],
                name="Tiros Totales", marker_color="#3f3f46",
            ))
            fig_evo.add_trace(go.Bar(
                x=d_club["Etiqueta_Partido"], y=d_club["shots_on_goal"],
                name="Tiros al Arco", marker_color=RED,
            ))
            fig_evo.update_layout(barmode="overlay")
        else:
            fig_evo.add_trace(go.Scatter(
                x=d_club["Etiqueta_Partido"], y=d_club["ball_possession"].round(1),
                name="Posesión %", mode="lines+markers", line=dict(color=RED, width=3), marker=dict(size=8),
                fill="tozeroy", fillcolor="rgba(237,26,59,0.10)",
            ))
            fig_evo.add_hline(y=50, line=dict(color=GRAY, dash="dash", width=1))

        fig_evo.update_layout(**PLOT_LAYOUT, height=340, legend=dict(orientation="h", y=1.12, x=0))
        _estilizar_ejes(fig_evo, grid_x=False, grid_y=True)
        fig_evo.update_xaxes(
            tickangle=-35,
            type="category",
            categoryorder="array",
            categoryarray=d_club["Etiqueta_Partido"].tolist(),
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        # ── Esquemas Tácticos (+ Origen de Remates solo en Clausura) ──
        jug_club = df_jug[(df_jug["equipo"] == club) & (df_jug["formacion"].notna())]
        if solo_clausura:
            col_esq, col_tiros = st.columns(2)
            with col_esq:
                st.markdown('<div class="section-header">Esquemas Tácticos Utilizados</div>', unsafe_allow_html=True)
                if not jug_club.empty:
                    form_partido = jug_club[["match_id", "formacion"]].drop_duplicates()
                    d_form = d_club.merge(form_partido, on="match_id", how="inner")
                    res_form = (
                        d_form.groupby("formacion", as_index=False)
                        .agg(PJ=("match_id", "count"), PTS=("puntos", "sum"), GF_pj=("goles_favor", "mean"), GC_pj=("goles_contra", "mean"), xG_pj=("expected_goals", "mean"), xGA_pj=("expected_goals_contra", "mean"))
                        .sort_values("PJ", ascending=False)
                    )
                    res_form["Efec %"] = (res_form["PTS"] / (res_form["PJ"] * 3) * 100).round(1)
                    res_form = res_form.rename(columns={"formacion": "Esquema", "GF_pj": "GF/pj", "GC_pj": "GC/pj", "xG_pj": "xG/pj", "xGA_pj": "xGA/pj"})
                    st.dataframe(
                        res_form.style.format({"Efec %": "{:.1f}%", "GF/pj": "{:.2f}", "GC/pj": "{:.2f}", "xG/pj": "{:.2f}", "xGA/pj": "{:.2f}"}),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.info("Sin datos de formación disponibles.")

            with col_tiros:
                st.markdown('<div class="section-header">Origen de Remates y Goles</div>', unsafe_allow_html=True)
                tiros_club = df_tiros[df_tiros["equipo"] == club].copy() if not df_tiros.empty else pd.DataFrame()
                if not tiros_club.empty and "situacion" in tiros_club.columns:
                    tiros_club["Situación"] = (
                        tiros_club["situacion"].astype(str).str.lower()
                        .map(TRADUCCION_SITUACION).fillna(tiros_club["situacion"].astype(str).str.capitalize())
                    )
                    res_sit = (
                        tiros_club.groupby("Situación", as_index=False)
                        .agg(Remates=("xg", "count"), Goles=("es_gol", "sum"), xG=("xg", "sum"), xGOT=("xgot", "sum"))
                        .sort_values("Remates", ascending=False)
                    )
                    res_sit["Goles"] = res_sit["Goles"].astype(int)
                    st.dataframe(res_sit.style.format({"xG": "{:.2f}", "xGOT": "{:.2f}"}), hide_index=True, use_container_width=True)
                else:
                    st.info("Sin datos de remates individuales.")
        else:
            st.markdown('<div class="section-header">Esquemas Tácticos Utilizados</div>', unsafe_allow_html=True)
            if not jug_club.empty:
                form_partido = jug_club[["match_id", "formacion"]].drop_duplicates()
                d_form = d_club.merge(form_partido, on="match_id", how="inner")
                res_form = (
                    d_form.groupby("formacion", as_index=False)
                    .agg(PJ=("match_id", "count"), PTS=("puntos", "sum"), GF_pj=("goles_favor", "mean"), GC_pj=("goles_contra", "mean"))
                    .sort_values("PJ", ascending=False)
                )
                res_form["Efec %"] = (res_form["PTS"] / (res_form["PJ"] * 3) * 100).round(1)
                res_form = res_form.rename(columns={"formacion": "Esquema", "GF_pj": "GF/pj", "GC_pj": "GC/pj"})
                st.dataframe(
                    res_form.style.format({"Efec %": "{:.1f}%", "GF/pj": "{:.2f}", "GC/pj": "{:.2f}"}),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("Sin datos de formación disponibles.")

        # ── Historial de Partidos ──
        st.markdown('<div class="section-header">Historial de Partidos</div>', unsafe_allow_html=True)
        hist = d_club.copy()
        hist["Instancia"] = hist.apply(
            lambda r: f"Fecha {int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) else str(r["ronda_nombre"]),
            axis=1,
        )
        hist["Resultado"] = hist["goles_favor"].astype(int).astype(str) + " - " + hist["goles_contra"].astype(int).astype(str)
        hist["Condición"] = hist["condicion"].str.capitalize()

        if solo_clausura:
            tabla_hist = hist[["fecha", "torneo", "Instancia", "Condición", "rival", "Resultado", "expected_goals", "expected_goals_contra", "ball_possession"]].copy()
            tabla_hist.columns = ["Fecha", "Torneo", "Instancia", "Condición", "Rival", "Resultado", "xG Favor", "xG Contra", "Posesión %"]
            fmt_hist = {"xG Favor": "{:.2f}", "xG Contra": "{:.2f}", "Posesión %": "{:.1f}%"}
        else:
            tabla_hist = hist[["fecha", "torneo", "Instancia", "Condición", "rival", "Resultado", "total_shots_on_goal", "shots_on_goal", "ball_possession"]].copy()
            tabla_hist.columns = ["Fecha", "Torneo", "Instancia", "Condición", "Rival", "Resultado", "Tiros", "Al Arco", "Posesión %"]
            tabla_hist["Tiros"] = tabla_hist["Tiros"].fillna(0).astype(int)
            tabla_hist["Al Arco"] = tabla_hist["Al Arco"].fillna(0).astype(int)
            fmt_hist = {"Posesión %": "{:.1f}%"}

        tabla_hist_show = agregar_columna_escudo(tabla_hist.iloc[::-1].reset_index(drop=True), col_equipo="Rival")
        st.dataframe(
            tabla_hist_show.style.format(fmt_hist),
            column_config={"Escudo": config_columna_escudo("")},
            hide_index=True,
            use_container_width=True,
        )

    # =========================================================================
    # TAB 3: COMPARADOR (2 o más equipos)
    # =========================================================================
    with tab_comp:
        default_eqs = [e for e in ["River Plate", "Boca Juniors"] if e in equipos_lista]
        if len(default_eqs) < 2 and len(equipos_lista) >= 2:
            default_eqs = equipos_lista[:2]

        seleccionados = st.multiselect("Elegí 2 o más equipos para comparar", equipos_lista, default=default_eqs)

        if len(seleccionados) < 2:
            st.info("Seleccioná al menos 2 equipos para activar la comparativa.")
        else:
            badges = "".join(
                f'<div style="display:inline-flex; align-items:center; gap:8px; background:#17171c; border:1px solid #26262d; padding:6px 14px; border-radius:999px;">'
                f'{img_escudo_html(eq, 26)}<span style="font-weight:700; color:#ffffff; font-size:0.88rem;">{eq}</span></div>'
                for eq in seleccionados
            )
            st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px 0;">{badges}</div>', unsafe_allow_html=True)

            metricas_comp = [
                ("Puntos / PJ", "puntos"),
                ("Goles a favor / PJ", "goles_favor"),
                ("Goles en contra / PJ", "goles_contra"),
                ("Posesión (%)", "ball_possession"),
                ("Tiros totales / PJ", "total_shots_on_goal"),
                ("Tiros al arco / PJ", "shots_on_goal"),
                ("Córners / PJ", "corner_kicks"),
                ("Atajadas / PJ", "goalkeeper_saves"),
            ]
            if solo_clausura:
                metricas_comp = [
                    ("Puntos / PJ", "puntos"),
                    ("Goles a favor / PJ", "goles_favor"),
                    ("xG a favor / PJ", "expected_goals"),
                    ("Goles en contra / PJ", "goles_contra"),
                    ("xG en contra / PJ", "expected_goals_contra"),
                    ("Posesión (%)", "ball_possession"),
                    ("Tiros totales / PJ", "total_shots_on_goal"),
                    ("Tiros al arco / PJ", "shots_on_goal"),
                    ("Ocasiones claras / PJ", "big_chance_created"),
                    ("Toques en área rival / PJ", "touches_in_opp_box"),
                    ("Precisión de pases (%)", "pct_pases"),
                    ("Duelos ganados (%)", "duel_won_percent"),
                    ("Recuperaciones / PJ", "ball_recovery"),
                ]

            filas = []
            for label, col in metricas_comp:
                if col not in df.columns:
                    continue
                fila = {"Métrica": label}
                for eq in seleccionados:
                    fila[eq] = round(float(df[df["equipo"] == eq][col].mean()), 2)
                filas.append(fila)

            df_comp = pd.DataFrame(filas)
            st.dataframe(
                df_comp.style.format({eq: "{:.2f}" for eq in seleccionados}).set_properties(subset=["Métrica"], **{"font-weight": "bold"}),
                hide_index=True,
                use_container_width=True,
                height=(len(df_comp) * 35) + 42,
            )

            st.markdown('<div class="section-header">Comparativa Visual (Promedios por Partido)</div>', unsafe_allow_html=True)
            claves_graf = (
                ["xG a favor / PJ", "xG en contra / PJ", "Goles a favor / PJ", "Ocasiones claras / PJ"]
                if solo_clausura
                else ["Puntos / PJ", "Goles a favor / PJ", "Goles en contra / PJ", "Tiros al arco / PJ"]
            )
            df_graf = df_comp[df_comp["Métrica"].isin(claves_graf)]
            fig_comp = go.Figure()
            for idx, eq in enumerate(seleccionados):
                fig_comp.add_trace(go.Bar(name=eq, x=df_graf["Métrica"], y=df_graf[eq], marker_color=PALETTE[idx % len(PALETTE)]))
            fig_comp.update_layout(**PLOT_LAYOUT, barmode="group", height=380, legend=dict(orientation="h", y=1.12, x=0))
            _estilizar_ejes(fig_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

    # =========================================================================
    # TAB 4: BUSCADOR DE PARTIDOS
    # =========================================================================
    with tab_partidos:
        f1, f2 = st.columns([2, 2])
        with f1:
            filtro_eq = st.selectbox("Filtrar por equipo", ["Todos los equipos"] + equipos_lista, key="busc_eq")
        with f2:
            df_loc = df[df["condicion"].str.lower() == "local"].sort_values("fecha_dt", ascending=False).copy()
            if filtro_eq != "Todos los equipos":
                df_loc = df_loc[(df_loc["local"] == filtro_eq) | (df_loc["visitante"] == filtro_eq)]

            df_loc["Instancia_Txt"] = df_loc.apply(
                lambda r: f"Fecha {int(r['ronda_num'])}" if not bool(r.get("es_playoff", False)) else str(r["ronda_nombre"]),
                axis=1,
            )
            df_loc["Label_Partido"] = (
                df_loc["torneo"].astype(str)
                + " · "
                + df_loc["Instancia_Txt"]
                + " | "
                + df_loc["local"]
                + " "
                + df_loc["goles_local"].astype(int).astype(str)
                + " - "
                + df_loc["goles_visitante"].astype(int).astype(str)
                + " "
                + df_loc["visitante"]
            )
            partido_sel = st.selectbox("Seleccioná el partido", df_loc["Label_Partido"].tolist(), key="busc_partido")

        if partido_sel:
            fila_m = df_loc[df_loc["Label_Partido"] == partido_sel].iloc[0]
            mid = fila_m["match_id"]
            nom_loc = fila_m["local"]
            nom_vis = fila_m["visitante"]

            row_l = df[(df["match_id"] == mid) & (df["equipo"] == nom_loc)].iloc[0]
            row_v = df[(df["match_id"] == mid) & (df["equipo"] == nom_vis)].iloc[0]

            tiene_xg_partido = (
                pd.notna(row_l.get("expected_goals"))
                and pd.notna(row_v.get("expected_goals"))
                and (float(row_l["expected_goals"]) + float(row_v["expected_goals"])) > 0
            )

            tiros_ref = df_tiros if (df_tiros is not None and not df_tiros.empty) else _cargar_tiros_fallback()
            goles_l_txt, asist_l_txt = _resumen_goles_asistencias(mid, nom_loc, df_jug, tiros_ref)
            goles_v_txt, asist_v_txt = _resumen_goles_asistencias(mid, nom_vis, df_jug, tiros_ref)

            bloque_inc_l = ""
            if goles_l_txt:
                bloque_inc_l += f'<div style="font-size:0.8rem; color:#ececf0; margin-top:4px;">⚽ {goles_l_txt}</div>'
            if asist_l_txt:
                bloque_inc_l += f'<div style="font-size:0.75rem; color:#8b8b95; margin-top:2px;">👟 Asist.: {asist_l_txt}</div>'

            bloque_inc_v = ""
            if goles_v_txt:
                bloque_inc_v += f'<div style="font-size:0.8rem; color:#ececf0; margin-top:4px;">⚽ {goles_v_txt}</div>'
            if asist_v_txt:
                bloque_inc_v += f'<div style="font-size:0.75rem; color:#8b8b95; margin-top:2px;">👟 Asist.: {asist_v_txt}</div>'

            fila_incidencias = ""
            if bloque_inc_l or bloque_inc_v:
                fila_incidencias = f"""
                <div style="display:flex; justify-content:center; align-items:flex-start; gap:24px; margin-top:14px; padding-top:12px; border-top:1px solid #26262d;">
                    <div style="flex:1; text-align:right;">{bloque_inc_l}</div>
                    <div style="width:110px;"></div>
                    <div style="flex:1; text-align:left;">{bloque_inc_v}</div>
                </div>
                """

            esc_l_partido = img_escudo_html(nom_loc, 52)
            esc_v_partido = img_escudo_html(nom_vis, 52)

            st.markdown(f"""
            <div style="background:linear-gradient(180deg, #17171c 0%, #121216 100%); border:1px solid #26262d; border-radius:14px; padding:22px; text-align:center; margin:12px 0 20px;">
                <div style="font-size:0.74rem; font-weight:800; color:#8b8b95; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;">
                    {fila_m['torneo']} · {fila_m['Instancia_Txt']} · {fila_m['fecha']}
                </div>
                <div style="display:flex; justify-content:center; align-items:center; gap:24px;">
                    <div style="display:flex; align-items:center; justify-content:flex-end; gap:14px; flex:1;">
                        <span style="font-family:'Bebas Neue', sans-serif; font-size:2rem; color:#fff;">{nom_loc}</span>
                        {esc_l_partido}
                    </div>
                    <div style="font-family:'Bebas Neue', sans-serif; font-size:3rem; color:#ED1A3B; background:#0a0a0c; border:1px solid #26262d; padding:4px 22px; border-radius:10px;">
                        {int(row_l['goles_favor'])} - {int(row_v['goles_favor'])}
                    </div>
                    <div style="display:flex; align-items:center; justify-content:flex-start; gap:14px; flex:1;">
                        {esc_v_partido}
                        <span style="font-family:'Bebas Neue', sans-serif; font-size:2rem; color:#fff;">{nom_vis}</span>
                    </div>
                </div>
                {fila_incidencias}
            </div>
            """, unsafe_allow_html=True)

            stats_partido = [
                ("Posesión de balón", "ball_possession", "{:.1f}%"),
                ("Tiros totales", "total_shots_on_goal", "{:.0f}"),
                ("Tiros al arco", "shots_on_goal", "{:.0f}"),
                ("Tiros dentro del área", "total_shots_inside_box", "{:.0f}"),
                ("Córners", "corner_kicks", "{:.0f}"),
                ("Faltas cometidas", "fouls", "{:.0f}"),
                ("Atajadas del arquero", "goalkeeper_saves", "{:.0f}"),
            ]
            if tiene_xg_partido:
                stats_partido = [
                    ("Goles esperados (xG)", "expected_goals", "{:.2f}"),
                    ("Posesión de balón", "ball_possession", "{:.1f}%"),
                    ("Tiros totales", "total_shots_on_goal", "{:.0f}"),
                    ("Tiros al arco", "shots_on_goal", "{:.0f}"),
                    ("Tiros dentro del área", "total_shots_inside_box", "{:.0f}"),
                    ("Ocasiones claras", "big_chance_created", "{:.0f}"),
                    ("Toques en área rival", "touches_in_opp_box", "{:.0f}"),
                    ("Precisión de pases", "pct_pases", "{:.1f}%"),
                    ("Duelos ganados", "duel_won_percent", "{:.1f}%"),
                    ("Recuperaciones", "ball_recovery", "{:.0f}"),
                    ("Córners", "corner_kicks", "{:.0f}"),
                    ("Faltas cometidas", "fouls", "{:.0f}"),
                    ("Atajadas del arquero", "goalkeeper_saves", "{:.0f}"),
                ]

            filas_p = []
            for label, col, f_str in stats_partido:
                if col in df.columns:
                    vl = float(row_l[col]) if pd.notna(row_l[col]) else 0.0
                    vv = float(row_v[col]) if pd.notna(row_v[col]) else 0.0
                    filas_p.append({nom_loc: f_str.format(vl), "Estadística": label, nom_vis: f_str.format(vv)})

            df_p = pd.DataFrame(filas_p)
            st.dataframe(
                df_p.style.set_properties(subset=["Estadística"], **{"color": "#8b8b95", "font-weight": "600"})
                .set_properties(subset=[nom_loc, nom_vis], **{"font-weight": "bold"}),
                hide_index=True,
                use_container_width=True,
                height=(len(df_p) * 35) + 42,
            )

            # ── Estadísticas Individuales de Ambos Planteles ──
            st.markdown('<div class="section-header">👥 Estadísticas de Ambos Planteles</div>', unsafe_allow_html=True)
            col_pl_loc, col_pl_vis = st.columns(2)
            with col_pl_loc:
                _render_plantel_partido(mid, nom_loc, df_jug, tiene_xg_partido)
            with col_pl_vis:
                _render_plantel_partido(mid, nom_vis, df_jug, tiene_xg_partido)