import base64
import os
import re
import unicodedata
from pathlib import Path
import pandas as pd
import streamlit as st

CARPETA_ESCUDOS = Path(__file__).resolve().parent / "assets" / "escudos"

ALIAS_ESCUDOS = {
    "river plate": "river.png",
    "ca river plate": "river.png",
    "club atletico river plate": "river.png",
    "boca juniors": "boca.png",
    "ca boca juniors": "boca.png",
    "club atletico boca juniors": "boca.png",
    "racing club": "racing.png",
    "rosario central": "rosariocentral.png",
    "ca rosario central": "rosariocentral.png",
    "estudiantes de la plata": "estudiantes.png",
    "estudiantes lp": "estudiantes.png",
    "estudiantes de rio cuarto": "estudiantesrc.png",
    "estudiantes rc": "estudiantesrc.png",
    "san lorenzo": "sanlorenzo.png",
    "ca san lorenzo": "sanlorenzo.png",
    "talleres": "talleres.png",
    "ca talleres": "talleres.png",
    "club atletico talleres": "talleres.png",
    "independiente rivadavia": "independienteriv.png",
    "ca independiente": "independiente.png",
    "independiente": "independiente.png",
    "argentinos juniors": "argentinos.png",
    "aa argentinos juniors": "argentinos.png",
    "lanus": "lanus.png",
    "ca lanus": "lanus.png",
    "club atletico lanus": "lanus.png",
    "tigre": "tigre.png",
    "ca tigre": "tigre.png",
    "club atletico tigre": "tigre.png",
    "platense": "platense.png",
    "ca platense": "platense.png",
    "club atletico platense": "platense.png",
    "newells old boys": "newells.png",
    "newell s old boys": "newells.png",
    "gimnasia y esgrima": "gimnasia.png",
    "gimnasia y esgrima la plata": "gimnasia.png",
    "gimnasia lp": "gimnasia.png",
    "gimnasia y esgrima mendoza": "gimnasiamendoza.png",
    "gimnasia mendoza": "gimnasiamendoza.png",
    "belgrano": "belgrano.png",
    "ca belgrano": "belgrano.png",
    "club atletico belgrano": "belgrano.png",
    "defensa y justicia": "defensa.png",
    "velez sarsfield": "velez.png",
    "velez": "velez.png",
    "huracan": "huracan.png",
    "ca huracan": "huracan.png",
    "club atletico huracan": "huracan.png",
    "union": "union.png",
    "ca union de santa fe": "union.png",
    "club atletico union de santa fe": "union.png",
    "union de santa fe": "union.png",
    "barracas central": "barracas.png",
    "instituto": "instituto.png",
    "instituto de cordoba": "instituto.png",
    "sarmiento": "sarmiento.png",
    "ca sarmiento": "sarmiento.png",
    "club atletico sarmiento": "sarmiento.png",
    "banfield": "banfield.png",
    "ca banfield": "banfield.png",
    "club atletico banfield": "banfield.png",
    "atletico tucuman": "atleticotucuman.png",
    "club atletico tucuman": "atleticotucuman.png",
    "aldosivi": "aldosivi.png",
    "ca aldosivi": "aldosivi.png",
    "club atletico aldosivi": "aldosivi.png",
    "deportivo riestra": "riestra.png",
    "riestra": "riestra.png",
    "central cordoba": "centralcordoba.png",
    "central cordoba sde": "centralcordoba.png",
}


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def obtener_ruta_escudo(equipo: str) -> str | None:
    if not equipo or not CARPETA_ESCUDOS.exists():
        return None

    norm = _normalizar(equipo)
    if norm in ALIAS_ESCUDOS:
        candidato = CARPETA_ESCUDOS / ALIAS_ESCUDOS[norm]
        if candidato.exists():
            return str(candidato)

    sin_prefijo = re.sub(r"^(club atletico|ca|aa|deportivo)\s+", "", norm).strip()
    if sin_prefijo in ALIAS_ESCUDOS:
        candidato = CARPETA_ESCUDOS / ALIAS_ESCUDOS[sin_prefijo]
        if candidato.exists():
            return str(candidato)

    sin_espacios = sin_prefijo.replace(" ", "")
    for arch in CARPETA_ESCUDOS.glob("*.png"):
        base = _normalizar(arch.stem).replace(" ", "")
        if base == sin_espacios or sin_espacios.startswith(base):
            return str(arch)

    return None


@st.cache_data(show_spinner=False)
def escudo_base64(equipo: str) -> str | None:
    ruta = obtener_ruta_escudo(equipo)
    if not ruta or not os.path.exists(ruta):
        return None
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def img_escudo_html(equipo: str, size: int = 44, extra_style: str = "") -> str:
    src = escudo_base64(equipo)
    if not src:
        return ""
    return (
        f'<img src="{src}" alt="{equipo}" '
        f'style="width:{size}px; height:{size}px; object-fit:contain; vertical-align:middle; {extra_style}" />'
    )


def agregar_columna_escudo(df: pd.DataFrame, col_equipo: str = "equipo", col_destino: str = "Escudo") -> pd.DataFrame:
    """Inserta una columna con el escudo en Base64 lista para st.column_config.ImageColumn."""
    if df.empty or col_equipo not in df.columns:
        return df
    out = df.copy()
    if col_destino in out.columns:
        out = out.drop(columns=[col_destino])
    idx = out.columns.get_loc(col_equipo)
    out.insert(idx, col_destino, out[col_equipo].map(escudo_base64))
    return out


def config_columna_escudo(label: str = "") -> st.column_config.ImageColumn:
    return st.column_config.ImageColumn(label, width="small")