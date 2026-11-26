import os
import numpy as np
import pandas as pd

ARCHIVO_EXCEL = "datos/temporada_lpf_2026.xlsx"
CARPETA_SALIDA = "datos"

MAPA_FASES = {
    "Liga Profesional de Fútbol, Apertura": "Apertura",
    "Liga Profesional de Fútbol, Apertura Playoffs": "Apertura Playoffs",
    "Liga Profesional, Clausura": "Clausura",
    "Primera LPF, Clausura Playoffs": "Clausura Playoffs",
}

MAPA_POSICIONES = {
    "G": "Arquero",
    "D": "Defensor",
    "M": "Mediocampista",
    "F": "Delantero",
}


def limpiar_jugadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["match_id", "jugador_id"], keep="last").copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    df["torneo"] = df["fase"].map(MAPA_FASES).fillna(df["fase"])
    df["es_playoff"] = df["torneo"].str.contains("Playoffs", na=False)
    df["torneo_macro"] = np.where(
        df["torneo"].str.startswith("Apertura"), "Apertura 2026", "Clausura 2026"
    )
    df["posicion_desc"] = df["posicion"].map(MAPA_POSICIONES).fillna("Sin pos.")

    # Rellenar con 0 métricas acumulables (manteniendo NaN donde no hay tracking/xG)
    cols_cero = [
        "minutesPlayed", "goals", "goalAssist", "totalShots", "onTargetScoringAttempt",
        "shotOffTarget", "blockedScoringAttempt", "bigChanceCreated", "bigChanceMissed",
        "keyPass", "totalPass", "accuratePass", "totalLongBalls", "accurateLongBalls",
        "totalCross", "accurateCross", "totalContest", "wonContest", "duelWon", "duelLost",
        "aerialWon", "aerialLost", "totalTackle", "wonTackle", "interceptionWon",
        "ballRecovery", "totalClearance", "outfielderBlock", "fouls", "wasFouled",
        "touches", "possessionLostCtrl", "dispossessed", "saves", "goalsConceded",
        "savedShotsFromInsideTheBox", "goodHighClaim", "punches", "penaltySave",
    ]
    for c in cols_cero:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Identificar ciclo del jugador en cada club (detecta transferencias entre Apertura y Clausura)
    df = df.sort_values(["jugador_id", "fecha_dt", "match_id"]).reset_index(drop=True)
    cambio_club = df.groupby("jugador_id")["equipo"].transform(lambda s: s != s.shift(1))
    df["ciclo_num"] = cambio_club.groupby(df["jugador_id"]).cumsum()
    df["etiqueta_ciclo"] = df["jugador"] + " (" + df["equipo"] + ")"

    return df


def _corregir_tandas_penales(df: pd.DataFrame, df_jug: pd.DataFrame) -> pd.DataFrame:
    """Separa los goles de los 90' de los penales convertidos en definiciones de Playoffs."""
    out = df.copy()
    out["penales_favor"] = np.nan
    out["penales_contra"] = np.nan

    if df_jug.empty or "goals" not in df_jug.columns:
        return out

    # En SofaScore, 'goals' de los jugadores solo cuenta goles en el partido (excluye tanda de penales)
    goles_jug = (
        df_jug.groupby(["match_id", "equipo"], as_index=False)["goals"]
        .sum()
        .rename(columns={"goals": "gf_jug"})
    )
    out = out.merge(goles_jug, on=["match_id", "equipo"], how="left")

    # Detectar partidos de playoff donde SofaScore sumó la tanda de penales al marcador
    mask_po = out["es_playoff"] & out["gf_jug"].notna()
    hubo_tanda = mask_po & ((out["goles_favor"] - out["gf_jug"]) >= 3)

    if hubo_tanda.any():
        mids_tanda = out.loc[hubo_tanda, "match_id"].unique()
        mask_mids = out["match_id"].isin(mids_tanda)

        out.loc[mask_mids, "penales_favor"] = (
            out.loc[mask_mids, "goles_favor"] - out.loc[mask_mids, "gf_jug"]
        ).astype(int)
        out.loc[mask_mids, "goles_favor"] = out.loc[mask_mids, "gf_jug"].astype(int)

        # Reconstruir goles_contra y penales_contra cruzando contra el rival del mismo partido
        rival_info = out.loc[mask_mids, ["match_id", "equipo", "goles_favor", "penales_favor"]].rename(
            columns={
                "equipo": "rival",
                "goles_favor": "gc_90",
                "penales_favor": "pen_contra_90",
            }
        )
        out = out.merge(rival_info, on=["match_id", "rival"], how="left")
        out.loc[mask_mids, "goles_contra"] = out.loc[mask_mids, "gc_90"].astype(int)
        out.loc[mask_mids, "penales_contra"] = out.loc[mask_mids, "pen_contra_90"].astype(int)
        out = out.drop(columns=["gc_90", "pen_contra_90"], errors="ignore")

        # Actualizar también goles_local / goles_visitante si existen en el DataFrame
        if {"goles_local", "goles_visitante", "condicion"}.issubset(out.columns):
            es_loc = mask_mids & (out["condicion"].str.lower() == "local")
            es_vis = mask_mids & (out["condicion"].str.lower() == "visitante")

            out.loc[es_loc, "goles_local"] = out.loc[es_loc, "goles_favor"].astype(int)
            out.loc[es_loc, "goles_visitante"] = out.loc[es_loc, "goles_contra"].astype(int)
            out.loc[es_vis, "goles_visitante"] = out.loc[es_vis, "goles_favor"].astype(int)
            out.loc[es_vis, "goles_local"] = out.loc[es_vis, "goles_contra"].astype(int)

    return out.drop(columns=["gf_jug"], errors="ignore")


def limpiar_equipos(df: pd.DataFrame, df_jug: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["match_id", "equipo"], keep="last").copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    df["torneo"] = df["fase"].map(MAPA_FASES).fillna(df["fase"])
    df["es_playoff"] = df["torneo"].str.contains("Playoffs", na=False)
    df["torneo_macro"] = np.where(
        df["torneo"].str.startswith("Apertura"), "Apertura 2026", "Clausura 2026"
    )

    if df_jug is not None and not df_jug.empty:
        df = _corregir_tandas_penales(df, df_jug)

    df["resultado"] = np.select(
        [df["goles_favor"] > df["goles_contra"], df["goles_favor"] == df["goles_contra"]],
        ["G", "E"],
        default="P",
    )
    df["puntos"] = np.select(
        [df["resultado"] == "G", df["resultado"] == "E"],
        [3, 1],
        default=0,
    )
    df["diferencia_gol"] = df["goles_favor"] - df["goles_contra"]
    df["valla_invicta"] = (df["goles_contra"] == 0).astype(int)

    # Porcentajes de precisión calculados sobre totales
    pares_pct = [
        ("pct_pases", "accurate_passes", "passes"),
        ("pct_pases_largos", "accurate_long_balls", "accurate_long_balls_total"),
        ("pct_centros", "accurate_cross", "accurate_cross_total"),
        ("pct_duelos_suelo", "ground_duels_percentage", "ground_duels_percentage_total"),
        ("pct_duelos_aereos", "aerial_duels_percentage", "aerial_duels_percentage_total"),
        ("pct_regates", "dribbles_percentage", "dribbles_percentage_total"),
        ("pct_entradas", "won_tackle_percent", "won_tackle_percent_total"),
        ("pct_pases_ultimo_tercio", "final_third_phase_statistic", "final_third_phase_statistic_total"),
    ]
    for col_dest, col_num, col_den in pares_pct:
        if col_num in df.columns and col_den in df.columns:
            df[col_dest] = np.where(df[col_den] > 0, (df[col_num] / df[col_den]) * 100, np.nan)

    return df.sort_values(["fecha_dt", "match_id", "condicion"]).reset_index(drop=True)


def limpiar_tiros(df: pd.DataFrame, df_eq: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    mapa_torneo = df_eq.drop_duplicates("match_id").set_index("match_id")["torneo"]
    df["torneo"] = df["match_id"].map(mapa_torneo)
    df["es_gol"] = (df["resultado"] == "goal").astype(int)
    return df.reset_index(drop=True)


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    print(f"Leyendo {ARCHIVO_EXCEL}...")
    xls = pd.ExcelFile(ARCHIVO_EXCEL, engine="openpyxl")

    df_jug = limpiar_jugadores(pd.read_excel(xls, "Jugadores"))
    df_eq = limpiar_equipos(pd.read_excel(xls, "Equipos"), df_jug=df_jug)
    df_tiros = limpiar_tiros(pd.read_excel(xls, "Tiros"), df_eq)

    df_eq.to_parquet(os.path.join(CARPETA_SALIDA, "equipos.parquet"), index=False)
    df_jug.to_parquet(os.path.join(CARPETA_SALIDA, "jugadores.parquet"), index=False)
    df_tiros.to_parquet(os.path.join(CARPETA_SALIDA, "tiros.parquet"), index=False)

    print("Parquets generados correctamente en datos/:")
    print(f"  - equipos.parquet:   {len(df_eq):,} filas | {df_eq['match_id'].nunique()} partidos")
    print(f"  - jugadores.parquet: {len(df_jug):,} filas | {df_jug['jugador_id'].nunique()} jugadores")
    print(f"  - tiros.parquet:     {len(df_tiros):,} filas")


if __name__ == "__main__":
    main()