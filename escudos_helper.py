import base64
from pathlib import Path
import unicodedata
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ESCUDOS_DIR = BASE_DIR / "assets" / "escudos"

MAPA_ESCUDOS = {
    "aldosivi": "aldosivi.png",
    "argentinos juniors": "argentinos.png",
    "atletico tucuman": "atleticotucuman.png",
    "banfield": "banfield.png",
    "barracas central": "barracas.png",
    "belgrano": "belgrano.png",
    "boca juniors": "boca.png",
    "central cordoba": "centralcordoba.png",
    "defensa y justicia": "defensa.png",
    "deportivo riestra": "riestra.png",
    "estudiantes de la plata": "estudiantes.png",
    "estudiantes de rio cuarto": "estudiantesrc.png",
    "gimnasia y esgrima": "gimnasia.png",
    "gimnasia y esgrima mendoza": "gimnasiamendoza.png",
    "huracan": "huracan.png",
    "independiente": "independiente.png",
    "independiente rivadavia": "independienteriv.png",
    "instituto": "instituto.png",
    "lanus": "lanus.png",
    "newells old boys": "newells.png",
    "platense": "platense.png",
    "racing club": "racing.png",
    "river plate": "river.png",
    "rosario central": "rosariocentral.png",
    "san lorenzo": "sanlorenzo.png",
    "sarmiento": "sarmiento.png",
    "talleres": "talleres.png",
    "tigre": "tigre.png",
    "union": "union.png",
    "velez sarsfield": "velez.png",
}


def _normalizar_equipo(nombre: str) -> str:
    if not isinstance(nombre, str):
        return ""
    s = unicodedata.normalize("NFD", nombre)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace("'", "").replace(".", "").strip()
    return " ".join(s.split())


def ruta_escudo(equipo: str) -> Path | None:
    clave = _normalizar_equipo(equipo)
    archivo = MAPA_ESCUDOS.get(clave)
    if not archivo:
        return None
    ruta = ESCUDOS_DIR / archivo
    return ruta if ruta.exists() else None


@st.cache_data(show_spinner=False)
def escudo_base64(equipo: str) -> str:
    """Devuelve Data URI PNG en base64 cacheado para ImageColumn y HTML."""
    ruta = ruta_escudo(equipo)
    if not ruta:
        return ""
    data = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def img_escudo_html(equipo: str, size: int = 44) -> str:
    """Devuelve etiqueta <img> limpia con el escudo del club (o vacío si no existe)."""
    uri = escudo_base64(equipo)
    if not uri:
        return ""
    return (
        f'<img src="{uri}" width="{size}" height="{size}" '
        f'style="object-fit:contain; vertical-align:middle; flex-shrink:0;" alt="{equipo}">'
    )


def agregar_columna_escudo(
    df: pd.DataFrame,
    col_equipo: str = "Equipo",
    col_destino: str = "Escudo",
    posicion: int | None = None,
) -> pd.DataFrame:
    """Inserta columna con Data URI del escudo justo antes de la columna del equipo."""
    out = df.copy()
    uris = out[col_equipo].map(escudo_base64)
    idx = posicion if posicion is not None else out.columns.get_loc(col_equipo)
    out.insert(idx, col_destino, uris)
    return out


def config_columna_escudo(label: str = ""):
    return st.column_config.ImageColumn(label, width="small")