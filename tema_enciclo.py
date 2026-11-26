import streamlit as st

X_URL = "https://x.com/ChernoSebi"
X_HANDLE = "@ChernoSebi"


def aplicar_estilo_enciclo():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Manrope:wght@400;500;600;700;800&family=Oswald:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: #ececf0;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #191216 0%, #0a0a0c 42%, #08080a 100%);
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1380px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: rgba(10, 10, 12, 0.75);
    backdrop-filter: blur(8px);
}

.section-header {
    font-family: 'Oswald', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #ffffff;
    border-left: 3.5px solid #ED1A3B;
    padding-left: 10px;
    margin: 18px 0 12px 0;
    line-height: 1.2;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    margin: 12px 0 18px 0;
}

.kpi-card {
    background: linear-gradient(180deg, #17171c 0%, #111115 100%);
    border: 1px solid #26262d;
    border-top: 2px solid #2e2e38;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #ED1A3B;
}

.kpi-label {
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8b8b95;
    margin-bottom: 6px;
}

.kpi-value {
    font-family: 'Oswald', sans-serif;
    font-size: 1.95rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 5px;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #a0a0a8;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #121216;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #26262d;
}

.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 8px;
    padding: 0 18px;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 0.86rem;
    color: #8b8b95;
    background-color: transparent;
    border: none;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ED1A3B 0%, #b9122c 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(237, 26, 59, 0.35);
}

div[data-baseweb="select"] > div {
    background-color: #141418 !important;
    border-color: #2a2a33 !important;
    border-radius: 9px !important;
}

span[data-baseweb="tag"] {
    background-color: #241318 !important;
    border: 1px solid #ED1A3B !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

div[data-testid="stExpander"] {
    background-color: #121216;
    border: 1px solid #26262d;
    border-radius: 12px;
}

.stButton > button {
    background-color: #17171c;
    color: #ececf0;
    border: 1px solid #2c2c35;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.8rem;
    transition: all 0.18s ease;
}

.stButton > button:hover {
    border-color: #ED1A3B;
    color: #ffffff;
    background-color: #211418;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #26262d;
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


def render_header_enciclo(subtitulo: str = "Base Estadística & Modelo Predictivo · Liga Profesional de Fútbol"):
    st.markdown(f"""
<div style="background:linear-gradient(90deg, #16161b 0%, #121216 65%, #1d1115 100%); border:1px solid #26262d; border-left:4px solid #ED1A3B; border-radius:14px; padding:16px 22px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px; box-shadow:0 8px 24px rgba(0,0,0,0.4);">
<div>
<div style="display:flex; align-items:center; gap:10px;">
<span style="font-family:'Bebas Neue',sans-serif; font-size:2.35rem; color:#ffffff; letter-spacing:2px; line-height:1;">ENCICLO <span style="color:#ED1A3B;">LPF</span></span>
<span style="background:rgba(237,26,59,0.14); border:1px solid rgba(237,26,59,0.45); color:#ff4d6a; font-size:0.68rem; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; padding:3px 9px; border-radius:6px;">DATA & SIMULATOR</span>
</div>
<div style="font-size:0.82rem; color:#8b8b95; margin-top:3px; font-weight:500;">{subtitulo}</div>
</div>
<a href="{X_URL}" target="_blank" style="display:inline-flex; align-items:center; gap:9px; background:#0d0d10; border:1px solid #2e2e38; border-radius:999px; padding:8px 16px; text-decoration:none; color:#ffffff; font-size:0.84rem; font-weight:700; box-shadow:0 4px 12px rgba(0,0,0,0.3);">
<svg width="14" height="14" viewBox="0 0 24 24" fill="#ED1A3B"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
<span>{X_HANDLE}</span>
</a>
</div>
""", unsafe_allow_html=True)


def render_footer_enciclo():
    st.markdown(f"""
<div style="margin-top:38px; padding:16px 20px; background:#111115; border:1px solid #26262d; border-radius:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; font-size:0.8rem; color:#8b8b95;">
<div><b style="color:#ffffff;">ENCICLO LPF</b> · Datos, visualizaciones y modelo predictivo xG-ELO / Dixon-Coles</div>
<a href="{X_URL}" target="_blank" style="display:inline-flex; align-items:center; gap:8px; background:#17171c; border:1px solid #2e2e38; border-radius:8px; padding:6px 13px; text-decoration:none; color:#ffffff; font-weight:700;">
<svg width="13" height="13" viewBox="0 0 24 24" fill="#ED1A3B"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
<span>Seguime en X · {X_HANDLE}</span>
</a>
</div>
""", unsafe_allow_html=True)