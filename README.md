# ⚽ Enciclo LPF

Plataforma web de analítica avanzada, scouting y simulación probabilística para la **Liga Profesional de Fútbol (LPF) 2026**, desarrollada en Python y Streamlit.

🔗 **App en vivo:** [enciclolpf.streamlit.app](https://enciclolpf.streamlit.app)

---

## 📊 Módulos de la aplicación

### 1. 🏟️ Clubes y Estadísticas (`vistas/equipos.py`)
* **Tablas de rendimiento:** Vista general, de local y de visitante con métricas acumuladas o por partido (`PTS`, `xG`, `xGA`, posesión, ocasiones claras, precisión de pases y duelos).
* **Explorador bidimensional:** Gráficos de dispersión (`Eje X vs Eje Y`) con líneas de promedio de la liga.
* **Ficha individual de club:** Evolución partido a partido, esquemas tácticos utilizados, zonas de origen de remates e historial completo de encuentros.
* **Comparador y Buscador de Partidos:** Cruce directo entre clubes y detalle encuentro por encuentro con planillas completas y resumen copiable para compartir.

### 2. 🏃 Jugadores y Scouting (`vistas/jugadores.py`)
* **Rankings y filtros:** Segmentación por club, posición, bloque estadístico y normalización **Cada 90'**.
* **Ficha de jugador:** Indicadores adaptados al puesto (arqueros, defensores, mediocampistas y delanteros), evolución de calificación y perfil de remates.
* **Detalle partido a partido:** Registro individual por fecha y desglose tiro por tiro (`xG`, `xGOT`, zona del arco, parte del cuerpo y situación).
* **Comparador y Líderes por Fecha:** Radar de percentiles por posición y jugadores destacados de cada jornada.

### 3. 🔮 Simulador Predictivo (`vistas/simulador.py` + `modelo_predictivo.py`)
* **Motor híbrido xG-Elo + Dixon-Coles:** Combina fuerza ofensiva y defensiva basada en goles esperados (`xG`), rating Elo dinámico con ajuste de localía, decaimiento exponencial por forma reciente y corrección de marcadores bajos ($\rho$ estimado por máxima verosimilitud).
* **Armado y validación del 11 titular:** Permite personalizar la formación de cada equipo con detección automática de esquema y validación de equilibrio táctico por líneas (`1 ARQ`, `3–5 DEF`, `2–6 MED`, `1–4 DEL`), recalculando al instante $\lambda$, probabilidades y figuras.
* **Contexto y proyecciones:** Soporte para cancha neutral (Playoffs), penalización manual por bajas/rotación, peso configurable de forma reciente, matriz de resultados exactos, proyección individual de gol/asistencia y resumen listo para copiar.

---

## 🔗 Navegación directa por URL

La aplicación soporta parámetros en la URL para abrir directamente la ficha de un club o un jugador específico:

* **Ficha de club:** `https://enciclolpf.streamlit.app/?club=River%20Plate`
* **Ficha de jugador:** `https://enciclolpf.streamlit.app/?jugador=Nombre%20Apellido`

---

## 🗂️ Estructura del repositorio

```text
cherno559/enciclo_lpf/
├── app.py                     # Punto de entrada, filtros globales, routing y query params
├── modelo_predictivo.py       # Motor xG-Elo bivariado, Dixon-Coles y proyección de jugadores
├── tema_enciclo.py            # Identidad visual, estilos CSS, header y footer
├── escudos_helper.py          # Mapeo, caché en base64 y renderizado de escudos
├── scraper.py                 # Extracción de datos de partidos, jugadores y mapa de tiros
├── bajar_puntuales.py         # Descarga puntual de partidos específicos
├── preparar_datos.py          # Limpieza, normalización y exportación a formato Parquet
├── requirements.txt           # Dependencias del proyecto
├── vistas/
│   ├── equipos.py             # Vistas de clubes, comparador e historial de partidos
│   ├── jugadores.py           # Vistas de scouting, radar y fichas individuales
│   └── simulador.py           # Interfaz del simulador predictivo y armado de alineaciones
├── datos/
│   ├── equipos.parquet        # Métricas agregadas por equipo y partido
│   ├── jugadores.parquet      # Rendimiento individual por partido
│   └── tiros.parquet          # Registro espacial y contextual de remates
└── assets/
    └── escudos/               # Escudos oficiales de los 30 clubes de la LPF (96x96 px)
