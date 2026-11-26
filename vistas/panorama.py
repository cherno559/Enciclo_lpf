import pandas as pd
import streamlit as st


def render_panorama(df_eq: pd.DataFrame):
    st.markdown('<div class="section-header">Panorama del Torneo</div>', unsafe_allow_html=True)

    if df_eq.empty:
        st.warning("No hay partidos para el filtro seleccionado.")
        return

    # KPIs rápidos (cada partido aparece 2 veces en df_eq: local y visitante)
    partidos_unicos = df_eq[df_eq["condicion"].str.lower() == "local"]
    total_partidos = len(partidos_unicos)
    prom_goles = (partidos_unicos["goles_favor"] + partidos_unicos["goles_contra"]).mean()
    pct_local = (partidos_unicos["resultado"] == "G").mean() * 100
    pct_valla = df_eq["valla_invicta"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos Jugados", f"{total_partidos}")
    c2.metric("Goles / Partido", f"{prom_goles:.2f}")
    c3.metric("Victorias Locales", f"{pct_local:.1f}%")
    c4.metric("Vallas Invictas", f"{pct_valla:.1f}%")

    # Tabla de Posiciones
    st.markdown('<div class="section-header">Tabla de Posiciones</div>', unsafe_allow_html=True)
    cond = st.radio("Condición", ["General", "Local", "Visitante"], horizontal=True)

    d = df_eq.copy()
    if cond != "General":
        d = d[d["condicion"].str.capitalize() == cond]

    tabla = (
        d.groupby("equipo", as_index=False)
        .agg(
            PJ=("match_id", "count"),
            PTS=("puntos", "sum"),
            GF=("goles_favor", "sum"),
            GC=("goles_contra", "sum"),
            DG=("diferencia_gol", "sum"),
            xGF=("expected_goals", "sum"),
            xGC=("expected_goals", lambda s: d.loc[s.index, "goles_contra"].sum()),
        )
        .sort_values(["PTS", "DG", "GF"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    tabla.insert(0, "#", tabla.index + 1)
    tabla["xGF"] = tabla["xGF"].round(2)

    st.dataframe(tabla, hide_index=True, use_container_width=True, height=(len(tabla) * 35) + 40)