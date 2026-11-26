import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────
# PARÁMETROS DEL MOTOR CALIBRADO (LPF ANALYTICS)
# ──────────────────────────────────────────────────────────────────────
W_XG = 0.70
K_PRIOR_BASE, K_PRIOR_MIN = 12.0, 8.0
MAX_GOALS_MATRIX = 7
PESO_HISTORICO = 0.75
LAM_MIN, LAM_MAX = 0.20, 5.00
FUERZA_RIVAL_MIN, FUERZA_RIVAL_MAX = 0.65, 1.40
HGA_LAMBDA = 0.18
CALIBRAR_GOLES_TOTALES = True
ESCALA_GOLES_MIN, ESCALA_GOLES_MAX = 0.75, 1.25
RHO_PRIOR_MU, RHO_PRIOR_SD = -0.10, 0.30
VALOR_PENAL_XG = 0.79

# Priors bayesianos para proyección individual (minutos fantasma y tasa base por 90')
PRIOR_MIN_JUGADOR = 540.0  # 6 partidos completos de regularización
PRIOR_GOL_P90_POS = {
    "delantero": 0.26,
    "mediocampista": 0.10,
    "defensor": 0.04,
    "arquero": 0.002,
}
PRIOR_AST_P90_POS = {
    "delantero": 0.13,
    "mediocampista": 0.11,
    "defensor": 0.04,
    "arquero": 0.005,
}
MAX_SHARE_GOL_XI = 0.38  # Tope máximo del xG del equipo que puede concentrar 1 jugador
MAX_SHARE_AST_XI = 0.35

_JERARQUIA_BASE = {
    "River Plate": 1.250, "Boca Juniors": 1.150, "Racing Club": 1.080, "Rosario Central": 1.065,
    "Estudiantes de La Plata": 1.050, "San Lorenzo": 1.045, "CA Talleres": 1.040, "Independiente Rivadavia": 1.035,
    "CA Independiente": 1.030, "Argentinos Juniors": 1.025, "CA Lanús": 1.025, "Tigre": 1.020,
    "Club Atlético Platense": 1.000, "Newell's Old Boys": 0.995, "Gimnasia y Esgrima": 0.990,
    "Club Atlético Belgrano": 0.990, "Defensa y Justicia": 0.985, "Vélez Sarsfield": 0.970,
    "Huracán": 0.965, "Club Atlético Unión de Santa Fe": 0.960, "Barracas Central": 0.955,
    "Instituto De Córdoba": 0.950, "Gimnasia y Esgrima Mendoza": 0.945, "Sarmiento": 0.940,
    "Banfield": 0.935, "Atlético Tucumán": 0.930, "Aldosivi": 0.925, "Deportivo Riestra": 0.925,
    "Central Córdoba": 0.920, "Estudiantes de Río Cuarto": 0.915,
}
AMPLIF_JERARQUIA = 2.1
JERARQUIA_EQUIPOS = {eq: 1.0 + (val - 1.0) * AMPLIF_JERARQUIA for eq, val in _JERARQUIA_BASE.items()}

_ORDEN_POSICION = {
    "arquero": 1,
    "defensor": 2,
    "mediocampista": 3,
    "delantero": 4,
}


def _rank_pos(pos_txt: str) -> int:
    p = str(pos_txt or "").lower()
    for k, v in _ORDEN_POSICION.items():
        if k in p:
            return v
    return 5


def _prior_tasa_posicion(pos_txt: str, mapa: dict, default: float = 0.10) -> float:
    p = str(pos_txt or "").lower()
    for k, v in mapa.items():
        if k in p:
            return v
    return default


def _suavizar_tasa_p90(prod_obs: pd.Series, min_obs: pd.Series, pos_series: pd.Series, mapa_prior: dict, default_prior: float) -> pd.Series:
    """Regresión bayesiana empírica: combina producción observada con el prior de la posición."""
    prior_rates = pos_series.map(lambda p: _prior_tasa_posicion(p, mapa_prior, default_prior)).astype(float)
    prior_events = prior_rates * (PRIOR_MIN_JUGADOR / 90.0)
    denom_90 = (min_obs + PRIOR_MIN_JUGADOR) / 90.0
    return (prod_obs + prior_events) / denom_90


def _col(df: pd.DataFrame, candidatos: list, default: float = 0.0, excluir_substrings: list = None) -> pd.Series:
    excluir = [e.lower() for e in (excluir_substrings or [])]
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidatos:
        if cand.lower() in cols_lower:
            return pd.to_numeric(df[cols_lower[cand.lower()]], errors="coerce").fillna(default)

    for cand in candidatos:
        cl = cand.lower()
        for col_l, col_orig in cols_lower.items():
            if any(ex in col_l for ex in excluir):
                continue
            if cl == col_l or col_l.startswith(cl + "_") or col_l.endswith("_" + cl):
                return pd.to_numeric(df[col_orig], errors="coerce").fillna(default)

    return pd.Series(default, index=df.index, dtype=float)


def _es_local_series(df: pd.DataFrame) -> pd.Series:
    for col in ["es_local", "is_home", "home"]:
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                return s.fillna(False)
            if pd.api.types.is_numeric_dtype(s):
                return s.fillna(0) == 1
            return s.astype(str).str.lower().isin(["true", "1", "local", "home", "l", "si", "sí"])
    for col in ["condicion", "localia", "lado", "side"]:
        if col in df.columns:
            return df[col].astype(str).str.lower().str.startswith(("loc", "home", "l"))
    if "match_id" in df.columns:
        return df.groupby("match_id").cumcount() == 0
    return pd.Series(True, index=df.index, dtype=bool)


def adaptar_df_enciclo_a_motor(df_eq: pd.DataFrame, df_jug: pd.DataFrame = None) -> pd.DataFrame:
    d = df_eq.copy().sort_values("fecha_dt").reset_index(drop=True)
    d["_es_local"] = _es_local_series(d)
    d["_condicion"] = np.where(d["_es_local"], "Local", "Visitante")

    if "torneo" not in d.columns:
        d["torneo"] = "Torneo"
    d["_categoria"] = np.where(
        d["torneo"].astype(str).str.contains("apertura", case=False, na=False),
        "Histórico",
        "Actual",
    )
    if (d["_categoria"] == "Actual").sum() == 0:
        d["_categoria"] = "Actual"

    if "es_playoff" not in d.columns:
        d["es_playoff"] = False
    d["_fase"] = np.where(d["es_playoff"].fillna(False), "Playoff", "Regular")
    d["_nfecha"] = pd.to_numeric(d.get("ronda_num", pd.Series(1, index=d.index)), errors="coerce").fillna(1).astype(int)

    goles_s = _col(
        d,
        ["goles", "goals", "gf", "goles_favor", "goles_marcados", "score", "marcador"],
        default=np.nan,
        excluir_substrings=["contra", "rec", "esp", "xg", "evit", "pen", "diff", "dif"],
    )

    if (goles_s.isna().all() or goles_s.fillna(0).sum() == 0) and df_jug is not None and not df_jug.empty:
        col_g_jug = next((c for c in ["goals", "goles"] if c in df_jug.columns), None)
        if col_g_jug:
            g_map = df_jug.groupby(["match_id", "equipo"])[col_g_jug].sum().to_dict()
            goles_s = pd.Series([float(g_map.get((m, e), 0.0)) for m, e in zip(d["match_id"], d["equipo"])], index=d.index)

    d["_goles"] = goles_s.fillna(0.0)

    xg_raw = _col(
        d,
        ["expectedGoals", "xg", "xg_favor", "goles_esperados"],
        default=np.nan,
        excluir_substrings=["contra", "rec", "xgot", "target", "arco", "diff", "dif"],
    )
    if (xg_raw.isna().all() or xg_raw.fillna(0).sum() == 0) and df_jug is not None and not df_jug.empty:
        col_xg_jug = next((c for c in ["expectedGoals", "xg", "xG"] if c in df_jug.columns), None)
        if col_xg_jug:
            xg_map = df_jug.groupby(["match_id", "equipo"])[col_xg_jug].sum().to_dict()
            xg_raw = pd.Series([float(xg_map.get((m, e), np.nan)) for m, e in zip(d["match_id"], d["equipo"])], index=d.index)

    tiros_tot = _col(
        d,
        ["totalShots", "tiros_totales", "tiros", "remates"],
        0.0,
        excluir_substrings=["contra", "rec", "arco", "area", "fuera", "bloq"],
    )
    ocasiones = _col(
        d,
        ["bigChanceCreated", "ocasiones_claras", "ocasiones"],
        0.0,
        excluir_substrings=["contra", "rec", "fall", "miss"],
    )

    penales_map = {}
    if df_jug is not None and not df_jug.empty:
        col_pen = next((c for c in ["penaltyWon", "penaltyGoals", "penales"] if c in df_jug.columns), None)
        if col_pen:
            penales_map = df_jug.groupby(["match_id", "equipo"])[col_pen].sum().to_dict()

    penales_s = pd.Series(
        [float(penales_map.get((m, e), 0.0)) for m, e in zip(d["match_id"], d["equipo"])],
        index=d.index,
    )

    xg_est = (ocasiones * 0.38) + (np.maximum(0.0, tiros_tot - ocasiones) * 0.05)
    xg_est = np.where(xg_est > 0, xg_est, d["_goles"])
    xg_neto = np.maximum(0.0, xg_raw - (penales_s * VALOR_PENAL_XG))
    d["_xg_model"] = np.where(xg_raw.notna() & (xg_raw > 0), xg_neto, xg_est)
    d["_xg_raw"] = xg_raw.fillna(pd.Series(xg_est, index=d.index))

    pos_s = _col(d, ["ballPossession", "posesion", "possession"], 50.0, excluir_substrings=["contra", "rival"])
    if pos_s.max() <= 1.05 and pos_s.max() > 0:
        pos_s = pos_s * 100.0
    d["_posesion"] = pos_s

    mapa_metricas = {
        "Resultado": d["_goles"],
        "xG_Model": d["_xg_model"],
        "Goles esperados (xG)": d["_xg_raw"],
        "xG al arco (xGOT)": _col(d, ["expectedGoalsOnTarget", "xgot", "xg_arco"], 0.0, excluir_substrings=["contra", "rec"]),
        "Tiros totales": tiros_tot,
        "Tiros al arco": _col(d, ["onTargetScoringAttempt", "al_arco", "tiros_arco", "remates_arco"], 0.0, excluir_substrings=["contra", "rec"]),
        "Ocasiones claras": ocasiones,
        "Posesión de balón": d["_posesion"],
        "Córners": _col(d, ["cornerKicks", "corners", "tiros_esquina"], 0.0, excluir_substrings=["contra", "rec"]),
        "Pases totales": _col(d, ["totalPass", "pases_totales", "pases"], 0.0, excluir_substrings=["contra", "prec", "acc", "pct"]),
        "Pases precisos": _col(d, ["accuratePass", "pases_precisos"], 0.0, excluir_substrings=["contra", "pct"]),
        "Quites": _col(d, ["wonTackle", "quites", "entradas"], 0.0, excluir_substrings=["contra"]),
        "Intercepciones": _col(d, ["interceptionWon", "intercepciones"], 0.0, excluir_substrings=["contra"]),
        "Despejes": _col(d, ["totalClearance", "despejes"], 0.0, excluir_substrings=["contra"]),
        "Atajadas del arquero": _col(d, ["saves", "atajadas", "paradas"], 0.0, excluir_substrings=["contra"]),
        "Goles evitados (arquero)": _col(d, ["goalsPrevented", "goles_evitados"], 0.0, excluir_substrings=["contra"]),
        "Tiros dentro del área": _col(d, ["totalShotsInsideBox", "shotsInsideBox", "tiros_area"], 0.0, excluir_substrings=["contra"]),
    }

    base_df = d[["match_id", "equipo", "_condicion", "_nfecha", "_fase", "torneo", "_categoria"]].copy()
    for nombre_m, serie_val in mapa_metricas.items():
        base_df[nombre_m] = serie_val.values

    parejas = base_df.merge(base_df, on="match_id", suffixes=("_a", "_b"))
    parejas = parejas[parejas["equipo_a"] != parejas["equipo_b"]].copy()

    filas = []
    for nombre_m in mapa_metricas.keys():
        sub = pd.DataFrame({
            "match_id": parejas["match_id"],
            "nFecha": parejas["_nfecha_a"],
            "Fase": parejas["_fase_a"],
            "Métrica": nombre_m,
            "Torneo": parejas["torneo_a"],
            "Categoria": parejas["_categoria_a"],
            "Equipo": parejas["equipo_a"],
            "Rival": parejas["equipo_b"],
            "Condicion": parejas["_condicion_a"],
            "Propio": parejas[f"{nombre_m}_a"],
            "Concedido": parejas[f"{nombre_m}_b"],
        })
        filas.append(sub)

    return pd.concat(filas, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# GESTIÓN DE PLANTEL Y CÁLCULO DE IMPACTO DEL 11 TITULAR
# ──────────────────────────────────────────────────────────────────────
def obtener_plantel_y_xi_default(df_jug: pd.DataFrame, equipo: str) -> tuple[pd.DataFrame, list[str]]:
    """Devuelve el plantel del club ordenado por puesto y la lista sugerida de 11 titulares (último partido)."""
    if df_jug is None or df_jug.empty:
        return pd.DataFrame(), []

    sub = df_jug[df_jug["equipo"] == equipo].copy()
    if sub.empty:
        return pd.DataFrame(), []

    if "fecha_dt" in sub.columns:
        sub = sub.sort_values("fecha_dt")

    sub["_min"] = _col(sub, ["minutesPlayed", "minutos"], 0.0)
    sub["_rat"] = _col(sub, ["rating", "nota"], np.nan)
    sub["_gol"] = _col(sub, ["goals", "goles"], 0.0)
    sub["_ast"] = _col(sub, ["goalAssist", "asistencias"], 0.0)

    if "suplente" in sub.columns:
        sub["_tit"] = (~sub["suplente"].fillna(True)).astype(int)
    else:
        sub["_tit"] = (sub["_min"] >= 60).astype(int)

    # Identificar titulares del último partido disponible del club
    ult_mid = sub["match_id"].iloc[-1]
    ult_df = sub[sub["match_id"] == ult_mid]
    ult_titulares = set(ult_df[ult_df["_tit"] == 1]["jugador"].dropna().tolist())
    if len(ult_titulares) < 11:
        extra_ult = (
            ult_df.sort_values("_min", ascending=False)["jugador"]
            .dropna()
            .tolist()
        )
        for j in extra_ult:
            if len(ult_titulares) >= 11:
                break
            ult_titulares.add(j)

    pos_principal = (
        sub.groupby(["jugador", "posicion_desc"], as_index=False)["_min"]
        .sum()
        .sort_values("_min", ascending=False)
        .drop_duplicates("jugador")
        .set_index("jugador")["posicion_desc"]
    )

    plantel = sub.groupby("jugador", as_index=False).agg(
        PJ=("_min", lambda s: int((s > 0).sum())),
        Tit=("_tit", "sum"),
        Min=("_min", "sum"),
        Nota=("_rat", "mean"),
        Goles=("_gol", "sum"),
        Asist=("_ast", "sum"),
    )
    plantel["Posicion"] = plantel["jugador"].map(pos_principal).fillna("Jugador")

    plantel["Orden_Pos"] = plantel["Posicion"].map(_rank_pos)
    plantel["Nota"] = plantel["Nota"].fillna(6.65)
    plantel["Es_Ult_XI"] = plantel["jugador"].isin(ult_titulares)
    plantel = plantel.sort_values(
        ["Orden_Pos", "Es_Ult_XI", "Min", "Nota"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    # Armar lista por defecto de 11 jugadores
    xi_default = plantel[plantel["Es_Ult_XI"]]["jugador"].tolist()
    if len(xi_default) < 11:
        for j in plantel["jugador"].tolist():
            if j not in xi_default:
                xi_default.append(j)
            if len(xi_default) == 11:
                break
    xi_default = xi_default[:11]

    # Reordenar xi_default por puesto (Arquero -> Def -> Med -> Del)
    orden_map = dict(zip(plantel["jugador"], plantel.index))
    xi_default = sorted(xi_default, key=lambda j: orden_map.get(j, 999))

    return plantel, xi_default


def calcular_modificadores_xi(
    df_jug: pd.DataFrame,
    equipo: str,
    titulares: list[str] | None,
) -> dict:
    """
    Compara el aporte ofensivo y defensivo por 90' de los 11 titulares elegidos
    contra el 11 promedio habitual del club en sus partidos recientes.
    """
    neutro = {"mult_atk": 1.0, "mult_def": 1.0, "activo": False, "n_sel": len(titulares or [])}
    if not titulares or len(titulares) != 11 or df_jug is None or df_jug.empty:
        return neutro

    sub = df_jug[df_jug["equipo"] == equipo].copy()
    if sub.empty:
        return neutro

    if "fecha_dt" in sub.columns:
        sub = sub.sort_values("fecha_dt")

    ult_matches = list(dict.fromkeys(sub["match_id"].tolist()))[-16:]
    sub_rec = sub[sub["match_id"].isin(ult_matches)].copy()
    n_partidos = max(1, len(ult_matches))

    sub_rec["_min"] = _col(sub_rec, ["minutesPlayed", "minutos"], 0.0)
    sub_rec["_rat"] = _col(sub_rec, ["rating", "nota"], np.nan)
    sub_rec["_xg"] = _col(sub_rec, ["expectedGoals", "xg", "xG"], 0.0)
    sub_rec["_xa"] = _col(sub_rec, ["expectedAssists", "xa", "xA"], 0.0)
    sub_rec["_gol"] = _col(sub_rec, ["goals", "goles"], 0.0)
    sub_rec["_ast"] = _col(sub_rec, ["goalAssist", "asistencias"], 0.0)
    sub_rec["_kp"] = _col(sub_rec, ["keyPass", "pases_clave"], 0.0)
    sub_rec["_sot"] = _col(sub_rec, ["onTargetScoringAttempt", "al_arco"], 0.0)
    sub_rec["_tk"] = _col(sub_rec, ["wonTackle", "quites"], 0.0)
    sub_rec["_int"] = _col(sub_rec, ["interceptionWon", "intercepciones"], 0.0)
    sub_rec["_rec"] = _col(sub_rec, ["ballRecovery"], 0.0)
    sub_rec["_sav"] = _col(sub_rec, ["saves", "atajadas"], 0.0)
    sub_rec["_gp"] = _col(sub_rec, ["goalsPrevented", "goles_evitados"], 0.0)

    tiene_xg = sub_rec["_xg"].sum() > 0

    pos_map = (
        sub_rec.groupby(["jugador", "posicion_desc"], as_index=False)["_min"]
        .sum()
        .sort_values("_min", ascending=False)
        .drop_duplicates("jugador")
        .set_index("jugador")["posicion_desc"]
    )

    ag = sub_rec.groupby("jugador", as_index=False).agg(
        Min=("_min", "sum"),
        Nota=("_rat", "mean"),
        xG=("_xg", "sum"),
        xA=("_xa", "sum"),
        Gol=("_gol", "sum"),
        Ast=("_ast", "sum"),
        KP=("_kp", "sum"),
        SoT=("_sot", "sum"),
        TK=("_tk", "sum"),
        INT=("_int", "sum"),
        REC=("_rec", "sum"),
        SAV=("_sav", "sum"),
        GP=("_gp", "sum"),
    )
    ag["posicion_desc"] = ag["jugador"].map(pos_map).fillna("Mediocampista")

    if ag.empty or ag["Min"].sum() <= 0:
        return neutro

    if tiene_xg:
        obs_g = 0.70 * ag["xG"] + 0.30 * ag["Gol"]
        obs_a = 0.70 * ag["xA"] + 0.30 * ag["Ast"]
    else:
        obs_g = ag["Gol"] + 0.12 * ag["SoT"]
        obs_a = ag["Ast"] + 0.12 * ag["KP"]

    prod_g_90 = _suavizar_tasa_p90(obs_g, ag["Min"], ag["posicion_desc"], PRIOR_GOL_P90_POS, 0.10)
    prod_a_90 = _suavizar_tasa_p90(obs_a, ag["Min"], ag["posicion_desc"], PRIOR_AST_P90_POS, 0.09)

    f90_def = 90.0 / np.maximum(ag["Min"], 360.0)
    nota_s = ag["Nota"].fillna(6.65)
    bonus_nota = np.clip((nota_s - 6.70) * 0.14, -0.18, 0.25)

    ag["score_atk"] = np.maximum(0.05, prod_g_90 + (0.72 * prod_a_90) + (0.03 * ag["KP"] * f90_def) + (0.05 * (1.0 + bonus_nota)))

    es_arq = ag["posicion_desc"].astype(str).str.lower().str.contains("arquero")
    acc_def_90 = (ag["TK"] + ag["INT"] + 0.45 * ag["REC"]) * f90_def
    acc_arq_90 = (0.18 * ag["SAV"] + 1.25 * ag["GP"]) * f90_def
    ag["score_def"] = np.maximum(
        0.20,
        (1.0 + bonus_nota) * np.where(es_arq, 1.35 + acc_arq_90, 0.85 + 0.14 * acc_def_90),
    )

    # Línea base: equipo promedio ponderado por cuota de minutos por partido (suma ~ 11.0 jugadores)
    peso_partido = ag["Min"] / (n_partidos * 90.0)
    suma_pesos = peso_partido.sum()
    if suma_pesos <= 0:
        return neutro
    peso_norm = peso_partido * (11.0 / suma_pesos)

    base_atk = float((ag["score_atk"] * peso_norm).sum())
    base_def = float((ag["score_def"] * peso_norm).sum())

    # 11 elegido
    ag_xi = ag[ag["jugador"].isin(titulares)].copy()
    if len(ag_xi) < 11:
        faltantes = 11 - len(ag_xi)
        xi_atk = float(ag_xi["score_atk"].sum()) + faltantes * (base_atk / 11.0) * 0.75
        xi_def = float(ag_xi["score_def"].sum()) + faltantes * (base_def / 11.0) * 0.78
    else:
        xi_atk = float(ag_xi["score_atk"].sum())
        xi_def = float(ag_xi["score_def"].sum())

    raw_atk = (xi_atk / base_atk) if base_atk > 0 else 1.0
    raw_def = (xi_def / base_def) if base_def > 0 else 1.0

    mult_atk = float(np.clip(1.0 + (raw_atk - 1.0) * 0.65, 0.72, 1.26))
    mult_def = float(np.clip(1.0 + (raw_def - 1.0) * 0.65, 0.75, 1.22))

    return {
        "mult_atk": round(mult_atk, 3),
        "mult_def": round(mult_def, 3),
        "activo": True,
        "n_sel": 11,
    }


# ──────────────────────────────────────────────────────────────────────
# MOTOR MATEMÁTICO: XG-ELO BIVARIADO (LOCAL/VISITANTE) Y MLE
# ──────────────────────────────────────────────────────────────────────
def calcular_elo_dinamico(df: pd.DataFrame) -> dict:
    dx = df[df["Métrica"] == "xG_Model"].copy()
    if dx.empty:
        return {"local": {}, "visitante": {}}

    dx["Torneo_Order"] = dx["Torneo"].apply(
        lambda x: 1 if "apertura" in str(x).lower() else (2 if "clausura" in str(x).lower() else 3)
    )
    dx = dx.sort_values(["Torneo_Order", "nFecha"])
    partidos = dx[dx["Condicion"] == "Local"]

    BASE_ELO = 1500.0
    HGA_SHIFT = 65.0
    HGA_BASE = 65.0

    hga_dinamico = {}
    for eq in dx["Equipo"].unique():
        xg_l_eq = dx[(dx["Equipo"] == eq) & (dx["Condicion"] == "Local")]["Propio"].mean()
        xg_v_eq = dx[(dx["Equipo"] == eq) & (dx["Condicion"] == "Visitante")]["Propio"].mean()

        if not np.isnan(xg_l_eq) and not np.isnan(xg_v_eq):
            hga_puro = float(np.clip((xg_l_eq - xg_v_eq) * 120, 45.0, 95.0))
            hga_dinamico[eq] = (HGA_BASE * 0.65) + (hga_puro * 0.35)
        else:
            hga_dinamico[eq] = HGA_BASE

    elos_l = {
        eq: BASE_ELO + hga_dinamico.get(eq, HGA_SHIFT) + ((JERARQUIA_EQUIPOS.get(eq, 1.0) - 1.0) * 500)
        for eq in dx["Equipo"].unique()
    }
    elos_v = {
        eq: BASE_ELO - hga_dinamico.get(eq, HGA_SHIFT) + ((JERARQUIA_EQUIPOS.get(eq, 1.0) - 1.0) * 500)
        for eq in dx["Equipo"].unique()
    }

    K = 40.0

    def prob_victoria_xg(xg_a, xg_b):
        if np.isnan(xg_a) or np.isnan(xg_b):
            return 0.5
        pa = [math.exp(-xg_a) * (xg_a ** k) / math.factorial(k) for k in range(8)]
        pb = [math.exp(-xg_b) * (xg_b ** k) / math.factorial(k) for k in range(8)]
        win_a = sum(pa[i] * sum(pb[:i]) for i in range(1, 8))
        draw = sum(pa[i] * pb[i] for i in range(8))
        tot = win_a + draw + sum(pb[i] * sum(pa[:i]) for i in range(1, 8))
        return (win_a + (draw / 2)) / tot if tot > 0 else 0.5

    for _, row in partidos.iterrows():
        loc, vis = row["Equipo"], row["Rival"]
        xg_l, xg_v = row["Propio"], row["Concedido"]

        elo_l, elo_v = elos_l.get(loc, BASE_ELO), elos_v.get(vis, BASE_ELO)
        e_loc = 1 / (1 + 10 ** ((elo_v - elo_l) / 400))
        e_vis = 1 - e_loc

        s_loc = prob_victoria_xg(xg_l, xg_v)
        s_vis = 1.0 - s_loc

        elos_l[loc] = elo_l + K * (s_loc - e_loc)
        elos_v[vis] = elo_v + K * (s_vis - e_vis)

    return {"local": elos_l, "visitante": elos_v}


def _rho_limites(la: float, lb: float, eps: float = 1e-3):
    lo = max(-1.0 / max(la, 1e-9), -1.0 / max(lb, 1e-9)) + eps
    hi = min(1.0, 1.0 / max(la * lb, 1e-9)) - eps
    return lo, hi


def estimar_rho_mle(df: pd.DataFrame) -> float:
    dr = df[(df["Métrica"] == "Resultado") & (df["Condicion"] == "Local")]
    if dr.empty:
        return -0.15

    hg, ag = dr["Propio"].values, dr["Concedido"].values
    mean_hg = np.mean(hg) if np.mean(hg) > 0 else 1.17
    mean_ag = np.mean(ag) if np.mean(ag) > 0 else 0.90
    lo, hi = _rho_limites(mean_hg, mean_ag)

    mask_00 = (hg == 0) & (ag == 0)
    mask_01 = (hg == 0) & (ag == 1)
    mask_10 = (hg == 1) & (ag == 0)
    mask_11 = (hg == 1) & (ag == 1)

    def neg_log_posterior(rho_val):
        r = rho_val[0]
        corr = np.ones_like(hg, dtype=float)
        corr[mask_00] = 1 - (mean_hg * mean_ag * r)
        corr[mask_01] = 1 + (mean_hg * r)
        corr[mask_10] = 1 + (mean_ag * r)
        corr[mask_11] = 1 - r
        corr = np.clip(corr, 1e-5, None)
        prior = ((r - RHO_PRIOR_MU) ** 2) / (2 * RHO_PRIOR_SD ** 2)
        return -np.sum(np.log(corr)) + prior

    res = minimize(neg_log_posterior, [float(np.clip(-0.10, lo, hi))], bounds=[(lo, hi)])
    return float(res.x[0]) if res.success else -0.15


def calcular_tabla(df: pd.DataFrame, elos: dict) -> pd.DataFrame:
    dr = df[df["Métrica"] == "Resultado"].copy()
    if "Fase" in dr.columns:
        dr = dr[dr["Fase"] == "Regular"]

    equipos = sorted(df["Equipo"].unique())
    rows = []
    for eq in equipos:
        d = dr[dr["Equipo"] == eq]
        pj = len(d)
        v = (d["Propio"] > d["Concedido"]).sum()
        e = (d["Propio"] == d["Concedido"]).sum()
        d_ = (d["Propio"] < d["Concedido"]).sum()
        pts = int(v * 3 + e)
        gf = d["Propio"].sum()
        gc = d["Concedido"].sum()
        rows.append({
            "Equipo": eq, "PJ": pj, "V": int(v), "E": int(e), "D": int(d_),
            "GF": gf, "GC": gc, "DG": gf - gc, "PTS": pts,
        })

    tabla = pd.DataFrame(rows).sort_values(["PTS", "DG", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    tabla["prior_atk_local"], tabla["prior_def_local"] = 1.0, 1.0
    tabla["prior_atk_visitante"], tabla["prior_def_visitante"] = 1.0, 1.0

    for eq in tabla.index:
        factor_l = elos["local"].get(tabla.loc[eq, "Equipo"], 1500.0) / 1500.0
        factor_v = elos["visitante"].get(tabla.loc[eq, "Equipo"], 1500.0) / 1500.0
        tabla.loc[eq, "prior_atk_local"] = factor_l
        tabla.loc[eq, "prior_def_local"] = (1 / factor_l) if factor_l > 0 else 1.0
        tabla.loc[eq, "prior_atk_visitante"] = factor_v
        tabla.loc[eq, "prior_def_visitante"] = (1 / factor_v) if factor_v > 0 else 1.0

    return tabla.set_index("Equipo")


def _get_prior(tabla: pd.DataFrame, eq: str, condicion: str):
    if tabla is None or eq not in tabla.index:
        return 1.0, 1.0
    if condicion == "Local":
        return float(tabla.loc[eq, "prior_atk_local"]), float(tabla.loc[eq, "prior_def_local"])
    return float(tabla.loc[eq, "prior_atk_visitante"]), float(tabla.loc[eq, "prior_def_visitante"])


def _adjusted_rate(d_all, metrica, col, max_fecha_torneo, tabla, is_attack, target_cond, half_life=4.0):
    df_m = d_all[d_all["Métrica"] == metrica]
    if df_m.empty:
        return np.nan

    fechas = df_m["nFecha"].values
    categoria = df_m["Categoria"].values
    valores = df_m[col].values
    rivales = df_m["Rival"].values
    condiciones = df_m["Condicion"].values

    valores_ajustados, pesos = [], []
    for v, r, c_match, f, cat in zip(valores, rivales, condiciones, fechas, categoria):
        cond_rival = "Visitante" if c_match == "Local" else "Local"
        pa_r, pd_r = _get_prior(tabla, r, cond_rival)

        pd_r_safe = float(np.clip(pd_r, FUERZA_RIVAL_MIN, FUERZA_RIVAL_MAX))
        pa_r_safe = float(np.clip(pa_r, FUERZA_RIVAL_MIN, FUERZA_RIVAL_MAX))
        adj = v / pd_r_safe if (is_attack and pd_r_safe > 0) else v / pa_r_safe if (not is_attack and pa_r_safe > 0) else v
        valores_ajustados.append(min(adj, 3.5))

        delta_f = max(0, max_fecha_torneo - f)
        w = PESO_HISTORICO if cat == "Histórico" else float(np.exp(-math.log(2) * (delta_f / half_life)))
        if c_match == target_cond:
            w *= 1.15
        pesos.append(max(w, 0.1))

    return float(np.average(valores_ajustados, weights=pesos)) if valores_ajustados and sum(pesos) > 0 else np.nan


def _league_stats(df):
    dr = df[df["Métrica"] == "Resultado"]
    dx = df[df["Métrica"] == "xG_Model"]

    def get_avg(d, cond):
        v = d[d["Condicion"] == cond]["Propio"].mean() if not d.empty else np.nan
        return v if not np.isnan(v) else 1.0

    gh, gv = get_avg(dr, "Local"), get_avg(dr, "Visitante")
    xh, xv = get_avg(dx, "Local"), get_avg(dx, "Visitante")
    if dx.empty:
        rh, rv = gh, gv
    else:
        rh, rv = W_XG * xh + (1 - W_XG) * gh, W_XG * xv + (1 - W_XG) * gv
    return {"ref_home": rh, "ref_away": rv, "ref_all": (rh + rv) / 2, "goles_total": gh + gv}


def _strength(df_actual, eq, target_cond, league, max_fecha_torneo: int, tabla: pd.DataFrame, half_life: float):
    d_eq = df_actual[df_actual["Equipo"] == eq]

    g_atk = _adjusted_rate(d_eq, "Resultado", "Propio", max_fecha_torneo, tabla, True, target_cond, half_life)
    x_atk = _adjusted_rate(d_eq, "xG_Model", "Propio", max_fecha_torneo, tabla, True, target_cond, half_life)
    g_def = _adjusted_rate(d_eq, "Resultado", "Concedido", max_fecha_torneo, tabla, False, target_cond, half_life)
    x_def = _adjusted_rate(d_eq, "xG_Model", "Concedido", max_fecha_torneo, tabla, False, target_cond, half_life)

    n_s = len(d_eq[d_eq["Métrica"] == "Resultado"])

    def combine(g, x):
        if np.isnan(g) and np.isnan(x):
            return np.nan
        if np.isnan(x):
            return g
        if np.isnan(g):
            return x
        return W_XG * x + (1 - W_XG) * g

    atk_val, def_val = combine(g_atk, x_atk), combine(g_def, x_def)

    rh, ra = league["ref_home"], league["ref_away"]
    ref_f, ref_a = (rh, ra) if target_cond == "Local" else (ra, rh)

    atk_obs = (atk_val / ref_f) if (not np.isnan(atk_val) and ref_f > 0) else np.nan
    def_obs = (def_val / ref_a) if (not np.isnan(def_val) and ref_a > 0) else np.nan

    prior_atk, prior_def = _get_prior(tabla, eq, target_cond)

    n = n_s if n_s > 0 else 0
    n_effective = min(n, 25)

    atk_obs = atk_obs if not np.isnan(atk_obs) else prior_atk
    def_obs = def_obs if not np.isnan(def_obs) else prior_def

    k_prior = max(K_PRIOR_MIN, K_PRIOR_BASE - max_fecha_torneo)
    atk_post = (n_effective * atk_obs + k_prior * prior_atk) / (n_effective + k_prior)
    def_post = (n_effective * def_obs + k_prior * prior_def) / (n_effective + k_prior)

    return atk_post, def_post, n


def escala_goles(l: dict, es_loc: bool = True) -> float:
    if not CALIBRAR_GOLES_TOTALES:
        return 1.0
    base = l["ref_home"] + l["ref_away"] + (HGA_LAMBDA if es_loc else 0.0)
    real = l.get("goles_total", np.nan)
    if not (base > 0) or not (real > 0) or np.isnan(real):
        return 1.0
    return float(np.clip(real / base, ESCALA_GOLES_MIN, ESCALA_GOLES_MAX))


def calcular_lambdas(df, eq_a, eq_b, es_loc, tabla, half_life=4.0):
    df_actual = df[df["Categoria"] == "Actual"]
    if df_actual.empty:
        df_actual = df

    l = _league_stats(df_actual)
    max_fecha_torneo = int(df_actual["nFecha"].max()) if not df_actual.empty else 1

    ca, cb = ("Local", "Visitante") if es_loc else ("Visitante", "Local")

    aa, da, _ = _strength(df_actual, eq_a, ca, l, max_fecha_torneo, tabla, half_life)
    ab, db, _ = _strength(df_actual, eq_b, cb, l, max_fecha_torneo, tabla, half_life)

    la = (l["ref_home"] if ca == "Local" else l["ref_away"]) * aa * db
    lb = (l["ref_home"] if cb == "Local" else l["ref_away"]) * ab * da

    def get_eficacia(equipo, condicion_buscada):
        dr = df_actual[(df_actual["Equipo"] == equipo) & (df_actual["Condicion"] == condicion_buscada) & (df_actual["Métrica"] == "Resultado")]
        if len(dr) == 0:
            return 0.45
        pts = sum([3 if r["Propio"] > r["Concedido"] else (1 if r["Propio"] == r["Concedido"] else 0) for _, r in dr.iterrows()])
        return pts / (len(dr) * 3.0)

    efec_a = get_eficacia(eq_a, ca)
    efec_b = get_eficacia(eq_b, cb)

    media_efec_loc, media_efec_vis = 0.48, 0.35
    if ca == "Local":
        mod_a = 1.0 + ((efec_a - media_efec_loc) * 0.35)
        mod_b = 1.0 + ((efec_b - media_efec_vis) * 0.35)
    else:
        mod_a = 1.0 + ((efec_a - media_efec_vis) * 0.35)
        mod_b = 1.0 + ((efec_b - media_efec_loc) * 0.35)

    la *= float(np.clip(mod_a, 0.85, 1.15))
    lb *= float(np.clip(mod_b, 0.85, 1.15))

    if es_loc:
        la += HGA_LAMBDA

    k_goles = escala_goles(l, es_loc)
    la *= k_goles
    lb *= k_goles

    if not es_loc:
        la, lb = lb, la

    return (
        round(float(np.clip(la, LAM_MIN, LAM_MAX)), 3),
        round(float(np.clip(lb, LAM_MIN, LAM_MAX)), 3),
    )


def proyectar_metrica(df, eq_a, eq_b, metrica, es_loc):
    df_m = df[df["Métrica"] == metrica]
    if df_m.empty:
        return 0.0, 0.0

    ca, cb = ("Local", "Visitante") if es_loc else ("Visitante", "Local")

    def _obtener_fuerza(eq, is_attack, cond_objetivo):
        col = "Propio" if is_attack else "Concedido"
        d_eq = df_m[df_m["Equipo"] == eq].sort_values("nFecha")
        if d_eq.empty:
            return df_m[col].mean()

        media_global = float(d_eq[col].mean())
        d_cond = d_eq[d_eq["Condicion"] == cond_objetivo]

        if len(d_cond) >= 3:
            pesos = np.linspace(1.0, 2.5, len(d_cond))
            media_ponderada = np.average(d_cond[col], weights=pesos)
            return (media_ponderada * 0.90) + (media_global * 0.10)
        elif len(d_cond) > 0:
            return (float(d_cond[col].mean()) * 0.70) + (media_global * 0.30)
        else:
            liga_global = df_m[col].mean()
            liga_cond = df_m[df_m["Condicion"] == cond_objetivo][col].mean()
            hga_ratio = (liga_cond / liga_global) if liga_global > 0 else 1.0
            return media_global * hga_ratio

    liga_gen_ca = df_m[df_m["Condicion"] == ca]["Propio"].mean()
    liga_gen_cb = df_m[df_m["Condicion"] == cb]["Propio"].mean()
    liga_gen_ca = liga_gen_ca if liga_gen_ca > 0 else 1.0
    liga_gen_cb = liga_gen_cb if liga_gen_cb > 0 else 1.0

    val_a = (_obtener_fuerza(eq_a, True, ca) * _obtener_fuerza(eq_b, False, cb)) / liga_gen_ca
    val_b = (_obtener_fuerza(eq_b, True, cb) * _obtener_fuerza(eq_a, False, ca)) / liga_gen_cb

    if metrica == "Posesión de balón":
        tot = val_a + val_b
        if tot > 0:
            diff = (val_a - val_b) * 1.15
            val_a_adj = np.clip((tot / 2) + (diff / 2), 20.0, 80.0)
            val_b_adj = np.clip((tot / 2) - (diff / 2), 20.0, 80.0)
            tot_adj = val_a_adj + val_b_adj
            return float((val_a_adj / tot_adj) * 100), float((val_b_adj / tot_adj) * 100)
        return 50.0, 50.0

    if metrica == "Goles evitados (arquero)":
        return float(val_a), float(val_b)

    return max(0.0, float(val_a)), max(0.0, float(val_b))


def montecarlo(la, lb, rho_dinamico):
    def _pmf(lam, kmax):
        k = np.arange(kmax + 1)
        return np.exp(k * np.log(max(lam, 1e-9)) - lam - np.array([math.log(math.factorial(x)) for x in k]))

    pa, pb = _pmf(la, MAX_GOALS_MATRIX), _pmf(lb, MAX_GOALS_MATRIX)
    M = np.outer(pa, pb)
    lo_r, hi_r = _rho_limites(la, lb)
    rho = float(np.clip(rho_dinamico, lo_r, hi_r))

    def tau(x, y, lam, mu, r):
        if x == 0 and y == 0:
            return 1.0 - (lam * mu * r)
        elif x == 0 and y == 1:
            return 1.0 + (lam * r)
        elif x == 1 and y == 0:
            return 1.0 + (mu * r)
        elif x == 1 and y == 1:
            return 1.0 - r
        return 1.0

    for i in range(min(2, M.shape[0])):
        for j in range(min(2, M.shape[1])):
            M[i, j] *= tau(i, j, la, lb, rho)

    M = np.clip(M, 0, None)
    M /= M.sum()
    return {
        "victoria": float(np.tril(M, -1).sum()),
        "empate": float(np.trace(M)),
        "derrota": float(np.triu(M, 1).sum()),
        "matrix": M,
    }


def simular_partido_completo(
    df_largo: pd.DataFrame,
    elos: dict,
    tabla: pd.DataFrame,
    rho_dinamico: float,
    eq_local: str,
    eq_vis: str,
    aplicar_localia: bool = True,
    penalizacion_loc: float = 0.0,
    penalizacion_vis: float = 0.0,
    half_life: float = 4.0,
    df_jug: pd.DataFrame = None,
    titulares_loc: list[str] | None = None,
    titulares_vis: list[str] | None = None,
) -> dict:
    la, lb = calcular_lambdas(df_largo, eq_local, eq_vis, aplicar_localia, tabla, half_life=half_life)

    # Ajuste por 11 titular elegido
    mod_xi_loc = calcular_modificadores_xi(df_jug, eq_local, titulares_loc)
    mod_xi_vis = calcular_modificadores_xi(df_jug, eq_vis, titulares_vis)

    if mod_xi_loc["activo"]:
        la = la * mod_xi_loc["mult_atk"]
        lb = lb / mod_xi_loc["mult_def"]

    if mod_xi_vis["activo"]:
        lb = lb * mod_xi_vis["mult_atk"]
        la = la / mod_xi_vis["mult_def"]

    elo_l = elos["local"].get(eq_local, 1500.0)
    elo_v = elos["visitante"].get(eq_vis, 1500.0)
    fact_prof_l = max(0.8, elo_l / 1500.0)
    fact_prof_v = max(0.8, elo_v / 1500.0)

    if penalizacion_loc > 0:
        pen_atk_l = (penalizacion_loc / 100.0) / (fact_prof_l ** 2)
        pen_def_l = ((penalizacion_loc / 100.0) * 0.6) / (fact_prof_l ** 2)
        la = max(0.15, la * (1.0 - pen_atk_l))
        lb = lb * (1.0 + pen_def_l)

    if penalizacion_vis > 0:
        pen_atk_v = (penalizacion_vis / 100.0) / (fact_prof_v ** 2)
        pen_def_v = ((penalizacion_vis / 100.0) * 0.6) / (fact_prof_v ** 2)
        lb = max(0.15, lb * (1.0 - pen_atk_v))
        la = la * (1.0 + pen_def_v)

    la = float(np.clip(la, LAM_MIN, LAM_MAX))
    lb = float(np.clip(lb, LAM_MIN, LAM_MAX))

    sim = montecarlo(la, lb, rho_dinamico)
    M = sim["matrix"]

    valla_loc = float(M[:, 0].sum() * 100)
    valla_vis = float(M[0, :].sum() * 100)
    ambos_marcan = float(M[1:, 1:].sum() * 100)

    over_15 = sum(M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if (i + j) > 1.5) * 100
    over_25 = sum(M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if (i + j) > 2.5) * 100

    flat = [(M[i, j] * 100, i, j) for i in range(6) for j in range(6)]
    flat.sort(reverse=True)
    df_marc = pd.DataFrame([{"Marcador": f"{i} - {j}", "Probabilidad": round(p, 1)} for p, i, j in flat[:5]])

    pos_a, pos_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Posesión de balón", aplicar_localia)
    tiros_a, tiros_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Tiros totales", aplicar_localia)
    arco_a, arco_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Tiros al arco", aplicar_localia)
    ocas_a, ocas_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Ocasiones claras", aplicar_localia)
    corn_a, corn_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Córners", aplicar_localia)
    xgot_a, xgot_b = proyectar_metrica(df_largo, eq_local, eq_vis, "xG al arco (xGOT)", aplicar_localia)
    ataj_a, ataj_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Atajadas del arquero", aplicar_localia)
    quit_a, quit_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Quites", aplicar_localia)
    inter_a, inter_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Intercepciones", aplicar_localia)
    desp_a, desp_b = proyectar_metrica(df_largo, eq_local, eq_vis, "Despejes", aplicar_localia)

    # Escalar tiros/ocasiones proporcionalmente al ajuste ofensivo del 11
    if mod_xi_loc["activo"]:
        tiros_a *= mod_xi_loc["mult_atk"]
        arco_a *= mod_xi_loc["mult_atk"]
        ocas_a *= mod_xi_loc["mult_atk"]
        xgot_a *= mod_xi_loc["mult_atk"]
    if mod_xi_vis["activo"]:
        tiros_b *= mod_xi_vis["mult_atk"]
        arco_b *= mod_xi_vis["mult_atk"]
        ocas_b *= mod_xi_vis["mult_atk"]
        xgot_b *= mod_xi_vis["mult_atk"]

    df_comp_met = pd.DataFrame({
        "Métrica": [
            "Posesión de balón", "Goles esperados (λ)", "xG al arco (xGOT)",
            "Tiros totales", "Tiros al arco", "Ocasiones claras",
            "Córners", "Atajadas del arquero", "Quites", "Intercepciones", "Despejes",
        ],
        eq_local: [
            f"{pos_a:.0f}%", f"{la:.2f}", f"{xgot_a:.2f}",
            f"{tiros_a:.1f}", f"{arco_a:.1f}", f"{ocas_a:.1f}",
            f"{corn_a:.1f}", f"{ataj_a:.1f}", f"{quit_a:.1f}", f"{inter_a:.1f}", f"{desp_a:.1f}",
        ],
        eq_vis: [
            f"{pos_b:.0f}%", f"{lb:.2f}", f"{xgot_b:.2f}",
            f"{tiros_b:.1f}", f"{arco_b:.1f}", f"{ocas_b:.1f}",
            f"{corn_b:.1f}", f"{ataj_b:.1f}", f"{quit_b:.1f}", f"{inter_b:.1f}", f"{desp_b:.1f}",
        ],
    })

    return {
        "lam_loc": round(la, 2),
        "lam_vis": round(lb, 2),
        "p_loc": round(sim["victoria"] * 100, 1),
        "p_emp": round(sim["empate"] * 100, 1),
        "p_vis": round(sim["derrota"] * 100, 1),
        "valla_loc": round(valla_loc, 1),
        "valla_vis": round(valla_vis, 1),
        "ambos_marcan": round(ambos_marcan, 1),
        "over_15": round(over_15, 1),
        "over_25": round(over_25, 1),
        "elo_loc": round(elo_l),
        "elo_vis": round(elo_v),
        "rho": rho_dinamico,
        "mod_xi_loc": mod_xi_loc,
        "mod_xi_vis": mod_xi_vis,
        "matriz": M[:6, :6] * 100,
        "top_marcadores": df_marc,
        "metricas_partido": df_comp_met,
    }


def _repartir_cuota_con_tope(pesos: np.ndarray, lam_total: float, cap_share: float) -> np.ndarray:
    """Reparte lam_total proporcionalmente a pesos respetando un tope máximo por jugador."""
    n = len(pesos)
    if n == 0 or lam_total <= 0:
        return np.zeros(n, dtype=float)

    w = np.maximum(pesos.astype(float), 1e-6)
    shares = w / w.sum()
    cap = max(cap_share, 1.0 / n + 1e-4)

    # Redistribución iterativa si algún jugador supera el techo de participación
    fijos = np.zeros(n, dtype=bool)
    for _ in range(n):
        exceso = (~fijos) & (shares > cap)
        if not exceso.any():
            break
        shares[exceso] = cap
        fijos |= exceso
        rem = 1.0 - shares[fijos].sum()
        if rem <= 0 or (~fijos).sum() == 0:
            break
        w_libre = w[~fijos]
        shares[~fijos] = rem * (w_libre / w_libre.sum())

    return shares * lam_total


def proyectar_jugadores_partido(
    df_jug: pd.DataFrame,
    eq_local: str,
    eq_visitante: str,
    lam_loc: float,
    lam_vis: float,
    titulares_loc: list[str] | None = None,
    titulares_vis: list[str] | None = None,
) -> pd.DataFrame:
    resultados = []

    config_equipos = [
        (eq_local, lam_loc, True, titulares_loc if titulares_loc and len(titulares_loc) == 11 else None),
        (eq_visitante, lam_vis, False, titulares_vis if titulares_vis and len(titulares_vis) == 11 else None),
    ]

    for eq, lam_eq, es_loc, xi_confirmado in config_equipos:
        sub = df_jug[df_jug["equipo"] == eq].copy()
        if sub.empty:
            continue

        if "fecha_dt" in sub.columns:
            sub = sub.sort_values("fecha_dt")

        # Ventana amplia de 18 partidos para estabilizar muestras de suplentes
        ult_matches = list(dict.fromkeys(sub["match_id"].tolist()))[-18:]
        sub_rec = sub[sub["match_id"].isin(ult_matches)].copy()
        n_partidos_eq = max(1, len(ult_matches))

        xg_j = _col(sub_rec, ["expectedGoals", "xg", "xG"], 0.0)
        xa_j = _col(sub_rec, ["expectedAssists", "xa", "xA"], 0.0)
        gol_j = _col(sub_rec, ["goals", "goles"], 0.0)
        ast_j = _col(sub_rec, ["goalAssist", "asistencias"], 0.0)
        tiro_arco_j = _col(sub_rec, ["onTargetScoringAttempt", "al_arco"], 0.0)
        kp_j = _col(sub_rec, ["keyPass", "pases_clave"], 0.0)

        tiene_xg = xg_j.sum() > 0
        if tiene_xg:
            sub_rec["prod_gol"] = 0.70 * xg_j + 0.30 * gol_j
            sub_rec["prod_ast"] = 0.70 * xa_j + 0.30 * ast_j
        else:
            sub_rec["prod_gol"] = gol_j + 0.12 * tiro_arco_j
            sub_rec["prod_ast"] = ast_j + 0.12 * kp_j

        sub_rec["_min"] = _col(sub_rec, ["minutesPlayed", "minutos"], 0.0)
        sub_rec["_rat"] = _col(sub_rec, ["rating", "nota"], np.nan)
        sub_rec["_rec"] = _col(sub_rec, ["ballRecovery"], 0.0)
        sub_rec["_tk"] = _col(sub_rec, ["wonTackle"], 0.0)

        # Consolidar posición principal ponderada por minutos para no dividir jugadores polifuncionales
        pos_map = (
            sub_rec.groupby(["jugador", "posicion_desc"], as_index=False)["_min"]
            .sum()
            .sort_values("_min", ascending=False)
            .drop_duplicates("jugador")
            .set_index("jugador")["posicion_desc"]
        )

        agrup = sub_rec.groupby("jugador", as_index=False).agg(
            PJ_Rec=("_min", lambda s: int((s > 0).sum())),
            Min_Tot=("_min", "sum"),
            Nota_Media=("_rat", "mean"),
            Prod_Gol=("prod_gol", "sum"),
            Prod_Ast=("prod_ast", "sum"),
            Recup=("_rec", "sum"),
            Quites=("_tk", "sum"),
        )
        agrup["posicion_desc"] = agrup["jugador"].map(pos_map).fillna("Mediocampista")

        # Tasa por 90' con regularización bayesiana empírica por posición (frena outliers de pocos minutos)
        agrup["rate_g_90"] = _suavizar_tasa_p90(
            agrup["Prod_Gol"], agrup["Min_Tot"], agrup["posicion_desc"], PRIOR_GOL_P90_POS, 0.10
        )
        agrup["rate_a_90"] = _suavizar_tasa_p90(
            agrup["Prod_Ast"], agrup["Min_Tot"], agrup["posicion_desc"], PRIOR_AST_P90_POS, 0.09
        )

        if xi_confirmado is not None:
            agrup = agrup[(agrup["jugador"].isin(xi_confirmado)) | (agrup["Min_Tot"] >= 90)].copy()
            if agrup.empty:
                continue

            mask_xi = agrup["jugador"].isin(xi_confirmado)
            df_xi = agrup[mask_xi].copy()
            df_sup = agrup[~mask_xi].copy()

            # 88% del xG/xA del equipo se reparte entre los 11 titulares con techo máximo individual
            xg_xi_arr = _repartir_cuota_con_tope(df_xi["rate_g_90"].values, lam_eq * 0.88, MAX_SHARE_GOL_XI)
            xa_xi_arr = _repartir_cuota_con_tope(df_xi["rate_a_90"].values, lam_eq * 0.72 * 0.88, MAX_SHARE_AST_XI)
            map_xg_xi = dict(zip(df_xi["jugador"], xg_xi_arr))
            map_xa_xi = dict(zip(df_xi["jugador"], xa_xi_arr))

            # 12% restante se reparte entre relevos ponderando su tasa suavizada * minutos habituales desde el banco
            if not df_sup.empty:
                min_sup_prom = np.clip(df_sup["Min_Tot"].values / n_partidos_eq, 8.0, 35.0)
                w_sup_g = df_sup["rate_g_90"].values * (min_sup_prom / 90.0)
                w_sup_a = df_sup["rate_a_90"].values * (min_sup_prom / 90.0)
                xg_sup_arr = _repartir_cuota_con_tope(w_sup_g, lam_eq * 0.12, 0.45)
                xa_sup_arr = _repartir_cuota_con_tope(w_sup_a, lam_eq * 0.72 * 0.12, 0.45)
                map_xg_sup = dict(zip(df_sup["jugador"], xg_sup_arr))
                map_xa_sup = dict(zip(df_sup["jugador"], xa_sup_arr))
            else:
                map_xg_sup, map_xa_sup = {}, {}

            for _, r in agrup.iterrows():
                jug_nom = r["jugador"]
                es_tit = jug_nom in xi_confirmado
                if es_tit:
                    min_prom = 85.0
                    xg_proy = map_xg_xi.get(jug_nom, 0.0)
                    xa_proy = map_xa_xi.get(jug_nom, 0.0)
                else:
                    min_prom = min(35.0, r["Min_Tot"] / n_partidos_eq)
                    xg_proy = map_xg_sup.get(jug_nom, 0.0)
                    xa_proy = map_xa_sup.get(jug_nom, 0.0)

                xg_proy = float(np.clip(xg_proy, 0.0, 1.15))
                xa_proy = float(np.clip(xa_proy, 0.0, 0.85))

                prob_gol = (1.0 - np.exp(-xg_proy)) * 100
                prob_ast = (1.0 - np.exp(-xa_proy)) * 100
                prob_ga = (1.0 - np.exp(-(xg_proy + xa_proy))) * 100

                nota_base = float(r["Nota_Media"]) if pd.notna(r["Nota_Media"]) else 6.75
                def_p90 = (r["Recup"] + r["Quites"]) * (90.0 / max(r["Min_Tot"], 270.0))
                indice_impacto = (
                    nota_base
                    + (xg_proy * 1.45)
                    + (xa_proy * 1.05)
                    + (def_p90 * 0.04)
                    + (0.12 if es_loc else 0.0)
                    + (0.18 if es_tit else -0.25)
                )

                resultados.append({
                    "Equipo": eq,
                    "Condición": "Local" if es_loc else "Visitante",
                    "Rol": "Titular" if es_tit else "Suplente",
                    "Jugador": jug_nom,
                    "Posición": r["posicion_desc"],
                    "Min_Prom": int(round(min_prom)),
                    "Nota_Base": round(nota_base, 2),
                    "xG_Proy": round(xg_proy, 2),
                    "Prob_Gol": round(prob_gol, 1),
                    "xA_Proy": round(xa_proy, 2),
                    "Prob_Asist": round(prob_ast, 1),
                    "Prob_GA": round(prob_ga, 1),
                    "Indice_Figura": round(indice_impacto, 2),
                })
        else:
            agrup = agrup[agrup["Min_Tot"] >= 90].copy()
            if agrup.empty:
                continue

            min_prom_arr = np.minimum(90.0, agrup["Min_Tot"].values / n_partidos_eq)
            w_g = agrup["rate_g_90"].values * (min_prom_arr / 90.0)
            w_a = agrup["rate_a_90"].values * (min_prom_arr / 90.0)

            xg_arr = _repartir_cuota_con_tope(w_g, lam_eq, MAX_SHARE_GOL_XI)
            xa_arr = _repartir_cuota_con_tope(w_a, lam_eq * 0.72, MAX_SHARE_AST_XI)

            for idx_r, (_, r) in enumerate(agrup.iterrows()):
                min_prom = float(min_prom_arr[idx_r])
                xg_proy = float(np.clip(xg_arr[idx_r], 0.0, 1.15))
                xa_proy = float(np.clip(xa_arr[idx_r], 0.0, 0.85))

                prob_gol = (1.0 - np.exp(-xg_proy)) * 100
                prob_ast = (1.0 - np.exp(-xa_proy)) * 100
                prob_ga = (1.0 - np.exp(-(xg_proy + xa_proy))) * 100

                nota_base = float(r["Nota_Media"]) if pd.notna(r["Nota_Media"]) else 6.75
                def_p90 = (r["Recup"] + r["Quites"]) * (90.0 / max(r["Min_Tot"], 270.0))
                indice_impacto = (
                    nota_base
                    + (xg_proy * 1.45)
                    + (xa_proy * 1.05)
                    + (def_p90 * 0.04)
                    + (0.12 if es_loc else 0.0)
                )

                resultados.append({
                    "Equipo": eq,
                    "Condición": "Local" if es_loc else "Visitante",
                    "Rol": "Habitual" if min_prom >= 55 else "Rotación",
                    "Jugador": r["jugador"],
                    "Posición": r["posicion_desc"],
                    "Min_Prom": int(round(min_prom)),
                    "Nota_Base": round(nota_base, 2),
                    "xG_Proy": round(xg_proy, 2),
                    "Prob_Gol": round(prob_gol, 1),
                    "xA_Proy": round(xa_proy, 2),
                    "Prob_Asist": round(prob_ast, 1),
                    "Prob_GA": round(prob_ga, 1),
                    "Indice_Figura": round(indice_impacto, 2),
                })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values("Indice_Figura", ascending=False).reset_index(drop=True)