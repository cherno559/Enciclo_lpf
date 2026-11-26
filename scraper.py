import os
import time
from datetime import datetime
import pandas as pd
from esd.sofascore import SofascoreClient

# ID 155 = Liga Profesional de Fútbol (Argentina) en SofaScore
TORNEO_ID = 155
ARCHIVO_SALIDA = "datos/temporada_lpf_2026.xlsx"
CARPETA_SALIDA = "datos"
GUARDAR_CADA = 15  # Guarda progreso en disco cada N partidos


def extraer_stats_equipo(stats_obj, meta):
    """Extrae las 7 categorías de MatchStats.all para Local y Visitante."""
    periodo_all = getattr(stats_obj, "all", None)
    if not periodo_all:
        return []

    dicc_local = {
        **meta,
        "equipo": meta["local"],
        "rival": meta["visitante"],
        "condicion": "Local",
        "goles_favor": meta["goles_local"],
        "goles_contra": meta["goles_visitante"],
    }
    dicc_visit = {
        **meta,
        "equipo": meta["visitante"],
        "rival": meta["local"],
        "condicion": "Visitante",
        "goles_favor": meta["goles_visitante"],
        "goles_contra": meta["goles_local"],
    }

    categorias = [
        "match_overview",
        "shots",
        "attack",
        "passes",
        "duels",
        "defending",
        "goalkeeping",
    ]

    for cat_nombre in categorias:
        cat_obj = getattr(periodo_all, cat_nombre, None)
        if not cat_obj:
            continue
        for stat_nombre, stat_item in vars(cat_obj).items():
            if not getattr(stat_item, "stat_type", ""):
                continue
            dicc_local[stat_nombre] = getattr(stat_item, "home_value", None)
            dicc_visit[stat_nombre] = getattr(stat_item, "away_value", None)
            if getattr(stat_item, "home_total", None) is not None:
                dicc_local[f"{stat_nombre}_total"] = stat_item.home_total
                dicc_visit[f"{stat_nombre}_total"] = stat_item.away_total

    return [dicc_local, dicc_visit]


def cargar_progreso(ruta):
    """Si el Excel ya existe, carga los datos previos para no repetir partidos."""
    if not os.path.exists(ruta):
        return [], [], [], set()

    xls = pd.ExcelFile(ruta)
    eq = pd.read_excel(xls, "Equipos").to_dict("records") if "Equipos" in xls.sheet_names else []
    jug = pd.read_excel(xls, "Jugadores").to_dict("records") if "Jugadores" in xls.sheet_names else []
    tiros = pd.read_excel(xls, "Tiros").to_dict("records") if "Tiros" in xls.sheet_names else []
    ids_hechos = {int(r["match_id"]) for r in eq if "match_id" in r}
    return eq, jug, tiros, ids_hechos


def guardar_excel(ruta, filas_equipos, filas_jugadores, filas_tiros):
    """Guarda las tres pestañas en Excel evitando errores si están vacías."""
    if not any([filas_equipos, filas_jugadores, filas_tiros]):
        print("No hay datos nuevos para guardar en Excel.")
        return

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        if filas_equipos:
            pd.DataFrame(filas_equipos).to_excel(writer, sheet_name="Equipos", index=False)
        if filas_jugadores:
            pd.DataFrame(filas_jugadores).to_excel(writer, sheet_name="Jugadores", index=False)
        if filas_tiros:
            pd.DataFrame(filas_tiros).to_excel(writer, sheet_name="Tiros", index=False)


def obtener_todos_los_partidos(client, torneo_id, season_id):
    """Pagina todos los partidos ya jugados de la temporada."""
    eventos = []
    page = 0
    while True:
        lote = client.get_tournament_events(
            torneo_id, season_id, upcoming=False, page=page
        )
        if not lote:
            break
        eventos.extend(lote)
        print(f"  Página {page}: +{len(lote)} eventos (total acumulado: {len(eventos)})")
        page += 1
        time.sleep(0.8)

    # Extraemos .value porque ev.status.type es un Enum (StatusType.FINISHED)
    terminados = []
    for ev in eventos:
        st_type = getattr(getattr(ev, "status", None), "type", None)
        estado = getattr(st_type, "value", st_type)
        if estado == "finished":
            terminados.append(ev)

    terminados.sort(key=lambda e: getattr(e, "start_timestamp", 0))
    return terminados


def main():
    ruta_salida = os.path.join(os.getcwd(), ARCHIVO_SALIDA)
    filas_equipos, filas_jugadores, filas_tiros, ids_hechos = cargar_progreso(ruta_salida)

    if ids_hechos:
        print(f"Progreso previo detectado: {len(ids_hechos)} partidos ya descargados.")

    with SofascoreClient() as client:
        service = getattr(client, "_SofascoreClient__service", None)

        # 1. Seleccionar temporada actual (index 0)
        temporadas = client.get_tournament_seasons(TORNEO_ID)
        temp_actual = temporadas[0]
        print(f"Torneo ID {TORNEO_ID} | Temporada: {temp_actual.name} (ID: {temp_actual.id})")

        # 2. Listar todos los partidos terminados de la temporada
        print("Buscando listado completo de partidos jugados...")
        partidos = obtener_todos_los_partidos(client, TORNEO_ID, temp_actual.id)
        pendientes = [ev for ev in partidos if ev.id not in ids_hechos]
        print(f"Total finalizados: {len(partidos)} | Pendientes por bajar: {len(pendientes)}")

        # 3. Descargar detalle de cada partido pendiente
        nuevos_procesados = 0
        for idx, ev in enumerate(pendientes, start=1):
            match_id = ev.id
            local = ev.home_team.name
            visitante = ev.away_team.name
            goles_l = getattr(ev.home_score, "current", None)
            goles_v = getattr(ev.away_score, "current", None)
            fecha = datetime.fromtimestamp(ev.start_timestamp).strftime("%Y-%m-%d %H:%M")
            ronda_num = getattr(getattr(ev, "round_info", None), "round", None)
            ronda_nom = getattr(getattr(ev, "round_info", None), "name", None)
            torneo_sub = getattr(getattr(ev, "tournament", None), "name", None)

            meta = {
                "match_id": match_id,
                "fecha": fecha,
                "fase": torneo_sub,
                "ronda_num": ronda_num,
                "ronda_nombre": ronda_nom,
                "local": local,
                "visitante": visitante,
                "goles_local": goles_l,
                "goles_visitante": goles_v,
            }

            print(f"[{idx}/{len(pendientes)}] {fecha[:10]} | {local} {goles_l}-{goles_v} {visitante} (ID: {match_id})")

            try:
                # A. Estadísticas de equipo
                stats_obj = client.get_match_stats(match_id)
                filas_equipos.extend(extraer_stats_equipo(stats_obj, meta))

                # B. Estadísticas jugador por jugador
                if service is not None:
                    url_lineups = service.endpoints.match_lineups_endpoint(match_id)
                    raw_lineups = service._get_json(url_lineups)

                    for lado_key, condicion, nombre_eq, rival_eq in [
                        ("home", "Local", local, visitante),
                        ("away", "Visitante", visitante, local),
                    ]:
                        lado_data = raw_lineups.get(lado_key, {})
                        formacion = lado_data.get("formation")
                        for p_item in lado_data.get("players", []):
                            p_info = p_item.get("player", {})
                            p_stats = p_item.get("statistics", {}) or {}
                            filas_jugadores.append({
                                "match_id": match_id,
                                "fecha": fecha,
                                "fase": torneo_sub,
                                "ronda_num": ronda_num,
                                "equipo": nombre_eq,
                                "rival": rival_eq,
                                "condicion": condicion,
                                "formacion": formacion,
                                "jugador_id": p_info.get("id"),
                                "jugador": p_info.get("name"),
                                "posicion": p_item.get("position") or p_info.get("position"),
                                "dorsal": p_item.get("shirtNumber"),
                                "suplente": p_item.get("substitute", False),
                                "capitan": p_item.get("captain", False),
                                **p_stats,
                            })

                # C. Mapa de tiros
                tiros = client.get_match_shots(match_id)
                for t in tiros:
                    filas_tiros.append({
                        "match_id": match_id,
                        "fecha": fecha,
                        "equipo": local if t.is_home else visitante,
                        "jugador": t.player.name if t.player else None,
                        "minuto": t.time,
                        "segundo": t.time_in_seconds,
                        "resultado": t.type,
                        "situacion": t.situation,
                        "parte_cuerpo": t.body_part,
                        "zona_arco": t.goal_mouth_location,
                        "xg": t.xg,
                        "xgot": t.xg_got,
                    })

                nuevos_procesados += 1

                # Autoguardado periódico
                if nuevos_procesados % GUARDAR_CADA == 0:
                    print(f"  --> Guardando checkpoint ({len(filas_equipos)//2} partidos en Excel)...")
                    guardar_excel(ruta_salida, filas_equipos, filas_jugadores, filas_tiros)

            except Exception as e:
                print(f"  [!] Error en partido {match_id}: {e} (se reintentará en la próxima ejecución)")

            time.sleep(1.3)

    # Guardado final
    guardar_excel(ruta_salida, filas_equipos, filas_jugadores, filas_tiros)
    print("\n=== DESCARGA DE TEMPORADA COMPLETADA ===")
    print(f"Archivo: {ruta_salida}")
    print(f"Partidos totales (filas Equipos / 2): {len(filas_equipos) // 2}")
    print(f"Registros Jugadores: {len(filas_jugadores)}")
    print(f"Registros Tiros: {len(filas_tiros)}")


if __name__ == "__main__":
    main()