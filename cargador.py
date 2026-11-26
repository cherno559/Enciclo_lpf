import pandas as pd
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos():
    df_eq = pd.read_parquet("datos/equipos.parquet")
    df_jug = pd.read_parquet("datos/jugadores.parquet")
    df_tiros = pd.read_parquet("datos/tiros.parquet")
    return df_eq, df_jug, df_tiros


def filtrar_tiempo(df_eq, df_jug, df_tiros, periodo: str, incluir_playoffs: bool):
    eq, jug, tiros = df_eq.copy(), df_jug.copy(), df_tiros.copy()

    if periodo == "Torneo Apertura":
        eq = eq[eq["torneo"].str.contains("Apertura", case=False, na=False)]
        jug = jug[jug["torneo"].str.contains("Apertura", case=False, na=False)]
        tiros = tiros[tiros["torneo"].str.contains("Apertura", case=False, na=False)]
    elif periodo == "Torneo Clausura":
        eq = eq[eq["torneo"].str.contains("Clausura", case=False, na=False)]
        jug = jug[jug["torneo"].str.contains("Clausura", case=False, na=False)]
        tiros = tiros[tiros["torneo"].str.contains("Clausura", case=False, na=False)]

    if not incluir_playoffs:
        eq = eq[~eq["es_playoff"].fillna(False)]
        jug = jug[~jug["es_playoff"].fillna(False)]
        tiros = tiros[tiros["match_id"].isin(eq["match_id"].unique())]

    return eq, jug, tiros