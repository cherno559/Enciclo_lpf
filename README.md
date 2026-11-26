# ⚽ Enciclo LPF — Analytics & Predictive Modeling Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enciclolpf.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly&logoColor=white)

Plataforma interactiva de **análisis estadístico, scouting y modelado predictivo** para la **Liga Profesional de Fútbol de Argentina (LPF)**, con cobertura de temporada completa (Torneo Apertura, Torneo Clausura y fases de Playoffs).

🌐 **Aplicación en producción:** [enciclolpf.streamlit.app](https://enciclolpf.streamlit.app/)

---

## 📌 Características Principales

### 1. 🏟️ Clubes y Estadísticas (`vistas/equipos.py`)
* **Tablas de Liga Dinámicas:** Clasificación general y splits de local/visitante con alternancia entre **promedios por partido** y **totales acumulados**. Incluye un **Explorador Visual (Scatter Plot)** configurable para cruzar cualquier par de métricas ofensivas, defensivas o de posesión.
* **Ficha de Club:** Panel individual con KPIs de rendimiento, evolución partido a partido (`xG` a favor vs. `xG` en contra, goles y posesión), efectividad por esquema táctico, distribución del origen de remates y registro completo de encuentros.
* **Comparador Multi-Equipo:** Evaluación simultánea de dos o más clubes en métricas de volumen ofensivo, generación de peligro (`xG`, ocasiones claras, toques en área rival), circulación y solidez defensiva.
* **Buscador de Partidos (Match Center):**
  * Placa de marcador con detalle de **goleadores** (con minuto exacto de conversión) y **asistidores**.
  * Comparativa estadística equipo vs. equipo.
  * Planillas individuales completas de ambos planteles (titulares, suplentes, esquema táctico, calificación SofaScore, `xG`, `xGOT`, `xA`, precisión de pases, regates, acciones defensivas y duelos).

### 2. 🏃 Jugadores y Scouting (`vistas/jugadores.py`)
* **Tablas y Rankings:** Filtrado por club, posición, umbral mínimo de minutos y bloque temático (*General, Ataque, Creación y Pases, Defensa y Duelos, Arqueros*), con normalización automática a **valores totales** o **cada 90 minutos (`p90`)**.
* **Ficha de Jugador:** Perfil individual con KPIs específicos según rol en cancha, curva de evolución de calificaciones partido a partido, mapa tabular de definición (destino en el arco y perfil de impacto) e historial de apariciones.
* **Rendimiento por Partido:** Auditoría granular por encuentro o acumulado del período dividida en tres módulos (*Ataque y Definición*, *Creación y Conducción*, *Defensa y Duelos*) junto con el registro **tiro por tiro** (`xG`, `xGOT`, minuto, situación y desenlace).
* **Comparador de Jugadores:** Gráficos de **radar de percentiles cada 90'** calibrados por posición frente al universo de jugadores de la liga, acompañados de su tabla de métricas reales.
* **Líderes por Fecha:** Ranking automático de los mejores rendimientos por posición en cada jornada e instancia eliminatoria.

### 3. 🔮 Simulador Predictivo (`src/predictor_lpf.py`)
* **Modelo Estadístico Híbrido:** Estimación de probabilidades de resultado (`1X2`), marcadores exactos y expectativa de goles mediante distribuciones de **Poisson / Dixon-Coles** ponderadas por **rating Elo** dinámico, fuerza ofensiva/defensiva ajustada por rival y factor de localía.
* **Simulación de Fechas y Escenarios:** Proyección de partidos individuales y simulación de jornadas completas para evaluar probabilidades de clasificación y cierre de tabla.

---

## 🗂️ Arquitectura del Proyecto

```text
lpf_pred/
├── streamlit_app.py          # Punto de entrada, configuración de UI y enrutamiento principal
├── escudos_helper.py         # Gestión y renderizado en Base64 de escudos oficiales de clubes
├── vistas/
│   ├── equipos.py            # Módulo de tablas, fichas de club, comparador y match center
│   └── jugadores.py          # Módulo de scouting, rankings p90, radares y desglose por partido
├── src/
│   └── predictor_lpf.py      # Motor matemático (Elo / Dixon-Coles / Poisson) y simulador
├── datos/                    # Datasets procesados en formato Parquet / Excel (equipos, jugadores, tiros)
└── requirements.txt          # Dependencias del entorno Python
```

---

## 🛠️ Stack Tecnológico

* **Core & Procesamiento de Datos:** Python, `pandas`, `numpy`, `scipy`, `pyarrow` (Parquet).
* **Visualización Interactiva:** `plotly` (scatter plots, series temporales, barras agrupadas, radares polares).
* **Frontend & Despliegue:** `streamlit` (arquitectura multipestaña con caché de datos `@st.cache_data` y estilizado CSS custom en modo oscuro).

---

## 🚀 Instalación y Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/cherno559/lpf_pred.git](https://github.com/cherno559/lpf_pred.git)
   cd lpf_pred
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   # venv\Scripts\activate         # Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📊 Fuentes y Metodología de Datos

* Los datos de eventos, remates (`xG` / `xGOT`), calificaciones y estadísticas individuales provienen de registros partido a partido de la Primera División del Fútbol Argentino.
* Las métricas normalizadas (`Cada 90'`) se calculan estrictamente sobre minutos reales disputados en cancha para evitar sesgos por ingresos desde el banco de suplentes.

---

## 👤 Autor

**Sebastián Chernomoretz**
* X / Twitter: [@ChernoSebi](https://x.com/ChernoSebi)
* GitHub: [@cherno559](https://github.com/cherno559)
