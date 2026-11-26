import inspect
from pathlib import Path
import pandas as pd
import streamlit as st

from tema_enciclo import (
    aplicar_estilo_enciclo,
    render_footer_enciclo,
    render_header_enciclo,
)
from vistas.simulador import render_simulador

st.set_page_config(
    page_title="Enciclo LPF",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

aplicar_estilo_enciclo()
render_header_enciclo()

BASE_DIR = Path(__file__).resolve().parent


def _preparar_fechas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "fecha_dt" in out.columns:
        out["fecha_dt"] = pd.to_datetime(out["fecha_dt"], errors="coerce")
    else:
        for cand in ["fecha", "date", "startTimestamp", "timestamp"]:
            if cand in out.columns:
                s = out[cand]
                if pd.api.types.is_numeric_dtype(s):
                    out["fecha_dt"] = pd.to_datetime(s, unit="s", errors="coerce")
                else:
                    out["fecha_dt"] = pd.to_datetime(s, errors="coerce")
                break
        if "fecha_dt" not in out.columns:
            out["fecha_dt"] = pd.NaT
    return out


def _es_df_tiros(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return bool(cols & {"zona_arco", "parte_cuerpo", "situacion", "es_gol", "xgot"})


def _es_df_jugadores(df: pd.DataFrame) -> bool:
    if _es_df_tiros(df):
        return False
    cols = {c.lower() for c in df.columns}
    return bool(cols & {"jugador", "player", "player_name", "minutesplayed", "posicion_desc"})


def _es_df_equipos(df: pd.DataFrame) -> bool:
    if _es_df_tiros(df) or _es_df_jugadores(df):
        return False
    cols = {c.lower() for c in df.columns}
    return "equipo" in cols or "team" in cols


@st.cache_data(show_spinner="Cargando base de datos de Enciclo LPF...")
def cargar_datos_locales() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_eq = pd.DataFrame()
    df_jug = pd.DataFrame()
    df_tiros = pd.DataFrame()

    carpetas = [BASE_DIR / "data", BASE_DIR / "datos", BASE_DIR]
    archivos = []
    for carpeta in carpetas:
        if carpeta.exists():
            archivos.extend(sorted(carpeta.glob("*.parquet")))
            archivos.extend(sorted(carpeta.glob("*.xlsx")))
            archivos.extend(sorted(carpeta.glob("*.csv")))

    def _clasificar(df_tmp: pd.DataFrame):
        nonlocal df_eq, df_jug, df_tiros
        if df_tmp.empty:
            return
        if df_tiros.empty and _es_df_tiros(df_tmp):
            df_tiros = df_tmp
        elif df_jug.empty and _es_df_jugadores(df_tmp):
            df_jug = df_tmp
        elif df_eq.empty and _es_df_equipos(df_tmp):
            df_eq = df_tmp

    vistos = set()
    for arch in archivos:
        if arch in vistos or arch.name.startswith("~$"):
            continue
        vistos.add(arch)

        try:
            if arch.suffix == ".parquet":
                _clasificar(pd.read_parquet(arch))
            elif arch.suffix == ".csv":
                _clasificar(pd.read_csv(arch))
            elif arch.suffix == ".xlsx":
                xls = pd.ExcelFile(arch)
                for hoja in xls.sheet_names:
                    _clasificar(xls.parse(hoja))
        except Exception:
            continue

        if not df_eq.empty and not df_jug.empty and not df_tiros.empty:
            break

    return _preparar_fechas(df_eq), _preparar_fechas(df_jug), _preparar_fechas(df_tiros)


def _llamar_vista(
    modulo_nombre: str,
    func_nombres: list[str],
    df_eq: pd.DataFrame,
    df_jug: pd.DataFrame,
    df_tiros: pd.DataFrame,
    periodo: str,
):
    try:
        mod = __import__(f"vistas.{modulo_nombre}", fromlist=["*"])
    except Exception as e:
        st.error(f"No se pudo importar `vistas/{modulo_nombre}.py`: {e}")
        return

    func = None
    for nombre in func_nombres:
        if hasattr(mod, nombre):
            func = getattr(mod, nombre)
            break

    if func is None:
        candidatas = [
            getattr(mod, attr)
            for attr in dir(mod)
            if attr.startswith("render_") and callable(getattr(mod, attr))
        ]
        if candidatas:
            func = candidatas[0]

    if func is None:
        st.warning(f"No se encontró función de renderizado en `vistas/{modulo_nombre}.py`.")
        return

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    kwargs = {}
    if "df_eq" in params:
        kwargs["df_eq"] = df_eq
    if "df_jug" in params:
        kwargs["df_jug"] = df_jug
    if "df_tiros" in params:
        kwargs["df_tiros"] = df_tiros
    if "periodo" in params:
        kwargs["periodo"] = periodo

    func(**kwargs)


df_eq, df_jug, df_tiros = cargar_datos_locales()

if df_eq.empty:
    st.warning("No se encontraron archivos de datos en la carpeta del proyecto (`data/` o raíz).")
    subido = st.file_uploader("Subí la base de datos (.xlsx o .parquet)", type=["xlsx", "parquet"])
    if subido is not None:
        if subido.name.endswith(".parquet"):
            df_eq = _preparar_fechas(pd.read_parquet(subido))
        else:
            xls = pd.ExcelFile(subido)
            for h in xls.sheet_names:
                d_h = xls.parse(h)
                if df_tiros.empty and _es_df_tiros(d_h):
                    df_tiros = _preparar_fechas(d_h)
                elif df_jug.empty and _es_df_jugadores(d_h):
                    df_jug = _preparar_fechas(d_h)
                elif df_eq.empty and _es_df_equipos(d_h):
                    df_eq = _preparar_fechas(d_h)

if not df_eq.empty:
    torneos_disp = []
    if "torneo" in df_eq.columns:
        torneos_disp = [t for t in df_eq["torneo"].dropna().unique().tolist() if str(t).strip()]

    # ── Barra Superior: Multiselect de Torneos + Interruptor de Playoffs ──
    f1, f2, f3 = st.columns([2.2, 1.0, 1.8])
    with f1:
        if torneos_disp:
            periodos_sel = st.multiselect(
                "🏆 Competencias / Períodos",
                options=torneos_disp,
                default=torneos_disp,
                placeholder="Elegí uno o más torneos...",
                key="filtro_periodos_global",
            )
            if not periodos_sel:
                periodos_sel = torneos_disp
        else:
            periodos_sel = []

    with f2:
        st.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
        incluir_playoffs = st.toggle(
            "Incluir Playoffs",
            value=True,
            key="filtro_playoffs_global",
        )

    # 1) Filtrar por torneos seleccionados
    if periodos_sel and "torneo" in df_eq.columns:
        df_eq_filt = df_eq[df_eq["torneo"].isin(periodos_sel)].copy()
    else:
        df_eq_filt = df_eq.copy()

    # 2) Filtrar Playoffs si el interruptor está apagado
    if not incluir_playoffs and "es_playoff" in df_eq_filt.columns:
        mask_po = df_eq_filt["es_playoff"].fillna(False).astype(bool)
        df_eq_filt = df_eq_filt[~mask_po].copy()

    mids_validos = set(df_eq_filt["match_id"].unique()) if "match_id" in df_eq_filt.columns else None

    def _sincronizar_df(df_sub: pd.DataFrame) -> pd.DataFrame:
        if df_sub.empty:
            return df_sub
        out = df_sub.copy()
        if "match_id" in out.columns and mids_validos is not None:
            return out[out["match_id"].isin(mids_validos)].copy()
        if periodos_sel and "torneo" in out.columns:
            out = out[out["torneo"].isin(periodos_sel)].copy()
        if not incluir_playoffs and "es_playoff" in out.columns:
            out = out[~out["es_playoff"].fillna(False).astype(bool)].copy()
        return out

    df_jug_filt = _sincronizar_df(df_jug)
    df_tiros_filt = _sincronizar_df(df_tiros)

    with f3:
        n_partidos = df_eq_filt["match_id"].nunique() if "match_id" in df_eq_filt.columns else len(df_eq_filt) // 2
        n_equipos = df_eq_filt["equipo"].nunique() if "equipo" in df_eq_filt.columns else 0
        txt_po = "con Playoffs" if incluir_playoffs else "solo Fase Regular"

        ult_txt = ""
        if "fecha_dt" in df_eq.columns and df_eq["fecha_dt"].notna().any():
            ult_dt = df_eq["fecha_dt"].max()
            ult_txt = f" · Actualizado al <b style='color:#cfb45e;'>{ult_dt.strftime('%d/%m/%Y')}</b>"

        st.markdown(
            f"<div style='padding-top:30px; font-size:0.82rem; color:#8b8b95;'>"
            f"📊 <b style='color:#ececf0;'>{n_equipos}</b> clubes · "
            f"<b style='color:#ececf0;'>{n_partidos}</b> partidos ({txt_po}){ult_txt}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Si hay un único torneo seleccionado (ej. "Torneo Clausura"), conserva su nombre exacto
    if len(periodos_sel) == 1:
        periodo_label = periodos_sel[0]
    elif len(periodos_sel) > 1:
        periodo_label = " + ".join(periodos_sel)
    else:
        periodo_label = "Temporada Completa"

    # ── Navegación Principal de Enciclo LPF ──
    tab_eq, tab_jug, tab_sim = st.tabs([
        "🏟️ Clubes y Estadísticas",
        "🏃 Jugadores y Scouting",
        "🔮 Simulador Predictivo",
    ])

    with tab_eq:
        _llamar_vista(
            "equipos",
            ["render_equipos", "render_vista_equipos"],
            df_eq_filt,
            df_jug_filt,
            df_tiros_filt,
            periodo_label,
        )

    with tab_jug:
        _llamar_vista(
            "jugadores",
            ["render_jugadores", "render_vista_jugadores"],
            df_eq_filt,
            df_jug_filt,
            df_tiros_filt,
            periodo_label,
        )

    with tab_sim:
        render_simulador(df_eq_filt, df_jug_filt, periodo=periodo_label)

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
with st.expander("ℹ️ Metodología y fuentes de datos"):
    st.markdown(
        """
        - **Datos:** Registros partido a partido de la Liga Profesional de Fútbol 2026 (Apertura, Clausura y Playoffs), excluyendo definiciones por penales de las métricas de tiempo regular.
        - **Motor Predictivo:** Modelo híbrido **xG-Elo bivariado** (local/visitante) combinado con distribución de **Poisson bivariada ajustada por Dixon-Coles** ($\\rho$ estimado por máxima verosimilitud) y decaimiento exponencial por forma reciente.
        - **Proyección Individual:** Tasas cada 90' ponderadas entre xG/xA y producción real, regularizadas mediante regresión bayesiana empírica por posición para estabilizar muestras de pocos minutos.
        """
    )

render_footer_enciclo()