import json
from datetime import datetime
import pandas as pd
from esd.sofascore import SofascoreClient
import scraper

IDS_FALTANTES = [15269969, 16671582, 16671631]


def limpiar_enum(val):
    return getattr(val, "value", val) if val is not None else None


def obtener_json_lineups(client, match_id):
    """Obtiene el JSON crudo de alineaciones desde SofaScore."""
    service = None
    for attr in dir(client):
        try:
            obj = getattr(client, attr)
            if hasattr(obj, "endpoints"):
                service = obj
                break
        except Exception:
            pass

    url = (
        service.endpoints.match_lineups_endpoint(match_id)
        if service
        else f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
    )
    if not str(url).startswith("http"):
        url = f"https://www.sofascore.com/api/v1/{str(url).lstrip('/')}"

    candidatos = [client]
    if service is not None:
        candidatos.append(service)
        for attr in dir(service):
            if attr.startswith("__") and attr.endswith("__"):
                continue
            try:
                obj = getattr(service, attr)
                if not callable(obj) and not isinstance(obj, (str, bytes, int, float, bool, dict, list, tuple)):
                    candidatos.append(obj)
            except Exception:
                pass

    for obj in candidatos:
        for metodo in dir(obj):
            if metodo.endswith(("__get", "_get", "get", "__fetch", "_fetch", "fetch")) and callable(getattr(obj, metodo)):
                try:
                    res = getattr(obj, metodo)(url)
                    if hasattr(res, "json"):
                        res = res.json()
                    if isinstance(res, (str, bytes)):
                        res = json.loads(res)
                    if isinstance(res, dict) and ("home" in res or "away" in res):
                        return res
                except Exception:
                    continue

    try:
        from curl_cffi import requests as cffi_req
        res = cffi_req.get(url, impersonate="chrome").json()
        if isinstance(res, dict) and ("home" in res or "away" in res):
            return res
    except Exception:
        pass

    import requests
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()
    if isinstance(res, dict) and ("home" in res or "away" in res):
        return res

    raise RuntimeError(f"No se pudo descargar el JSON de lineups para {match_id}")


def parsear_jugadores(raw_lineups, meta):
    filas = []
    lados = [
        ("home", meta["local"], meta["visitante"], "Local"),
        ("away", meta["visitante"], meta["local"], "Visitante"),
    ]
    for lado_key, equipo, rival, condicion in lados:
        bloque = raw_lineups.get(lado_key, {})
        formacion = bloque.get("formation")
        for item in bloque.get("players", []):
            pl = item.get("player", {})
            fila = {
                "match_id": meta["match_id"],
                "fecha": meta["fecha"],
                "fase": meta["fase"],
                "ronda_num": meta["ronda_num"],
                "equipo": equipo,
                "rival": rival,
                "condicion": condicion,
                "formacion": formacion,
                "jugador_id": pl.get("id"),
                "jugador": pl.get("name"),
                "posicion": item.get("position"),
                "dorsal": item.get("shirtNumber"),
                "suplente": bool(item.get("substitute", False)),
                "capitan": bool(item.get("captain", False)),
            }
            stats_jug = item.get("statistics", {})
            if isinstance(stats_jug, dict):
                for k, v in stats_jug.items():
                    fila[k] = str(v) if isinstance(v, (dict, list)) else v
            filas.append(fila)
    return filas


def parsear_tiros(tiros, meta):
    if not tiros:
        return []
    lista_tiros = tiros.shots if hasattr(tiros, "shots") else tiros
    filas = []
    for t in lista_tiros:
        pl = getattr(t, "player", None)
        filas.append({
            "match_id": meta["match_id"],
            "fecha": meta["fecha"],
            "equipo": meta["local"] if getattr(t, "is_home", False) else meta["visitante"],
            "jugador": getattr(pl, "name", None) if pl else None,
            "minuto": getattr(t, "time", None),
            "segundo": getattr(t, "time_seconds", None),
            "resultado": limpiar_enum(getattr(t, "shot_type", None)),
            "situacion": limpiar_enum(getattr(t, "situation", None)),
            "parte_cuerpo": limpiar_enum(getattr(t, "body_part", None)),
            "zona_arco": limpiar_enum(getattr(t, "goal_mouth_location", None)),
            "xg": getattr(t, "xg", None),
            "xgot": getattr(t, "xgot", None),
        })
    return filas


def main():
    xls = pd.ExcelFile(scraper.ARCHIVO_SALIDA, engine="openpyxl")
    df_eq = pd.read_excel(xls, "Equipos")
    df_jug = pd.read_excel(xls, "Jugadores")
    df_tiros = pd.read_excel(xls, "Tiros")
    ya_descargados = set(df_eq["match_id"].unique())

    nuevos_eq, nuevos_jug, nuevos_tiros = [], [], []

    with SofascoreClient() as client:
        for mid in IDS_FALTANTES:
            if mid in ya_descargados:
                print(f"[{mid}] Ya existe en el Excel, se omite.")
                continue

            ev = client.get_event(mid)
            print(f"[{mid}] {ev.home_team.name} {ev.home_score.current}-{ev.away_score.current} {ev.away_team.name}")

            stats = client.get_match_stats(mid)
            if not stats or not stats.all:
                print("  -> Sin planilla estadística en SofaScore.")
                continue

            meta = {
                "match_id": ev.id,
                "fecha": datetime.fromtimestamp(ev.start_timestamp).strftime("%Y-%m-%d %H:%M"),
                "fase": ev.tournament.name,
                "ronda_num": getattr(ev.round_info, "round", None) if ev.round_info else None,
                "ronda_nombre": getattr(ev.round_info, "name", None) if ev.round_info else None,
                "local": ev.home_team.name,
                "visitante": ev.away_team.name,
                "goles_local": ev.home_score.current,
                "goles_visitante": ev.away_score.current,
            }

            nuevos_eq.extend(scraper.extraer_stats_equipo(stats, meta))
            raw_jug = obtener_json_lineups(client, mid)
            nuevos_jug.extend(parsear_jugadores(raw_jug, meta))
            tiros = client.get_match_shots(mid)
            nuevos_tiros.extend(parsear_tiros(tiros, meta))
            print("  -> OK")

    if not nuevos_eq:
        print("\nNo hubo partidos nuevos para guardar.")
        return

    df_eq = pd.concat([df_eq, pd.DataFrame(nuevos_eq)], ignore_index=True).drop_duplicates(["match_id", "equipo"], keep="last")
    df_jug = pd.concat([df_jug, pd.DataFrame(nuevos_jug)], ignore_index=True).drop_duplicates(["match_id", "jugador_id"], keep="last")
    df_tiros = pd.concat([df_tiros, pd.DataFrame(nuevos_tiros)], ignore_index=True)

    with pd.ExcelWriter(scraper.ARCHIVO_SALIDA, engine="openpyxl") as writer:
        df_eq.to_excel(writer, sheet_name="Equipos", index=False)
        df_jug.to_excel(writer, sheet_name="Jugadores", index=False)
        df_tiros.to_excel(writer, sheet_name="Tiros", index=False)

    print(f"\n¡Listo! Agregados {len(nuevos_eq)//2} partidos. Total partidos únicos: {df_eq['match_id'].nunique()}")


if __name__ == "__main__":
    main()