# ingest/weather.py
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
from ml_cfb.clients.cfbd_client import GamesAPI, get_games_api

def _parse_weather(weather_data: List[Dict[str, Any]]) -> List[dict]:
    records = []
    for game in weather_data:
        # We need to handle 'None' for games played in domes
        def safe_float(key):
            val = game.get(key)
            return float(val) if val is not None else None

        records.append({
            "game_id": int(game["id"]),
            "season": int(game["season"]),
            "is_dome": bool(game.get("gameIndoors")),
            "temperature": safe_float("temperature"),
            "wind_speed": safe_float("windSpeed"),
            "precipitation": safe_float("precipitation"), 
        })
    return records

def fetch_weather_for_season(client, season: int) -> pd.DataFrame:
    api: GamesAPI = get_games_api(client)
    
    weather_data = api.get_weather(
        year=season,
        season_type="regular"
    )
    
    df = pd.DataFrame.from_records(_parse_weather(weather_data))
    
    return df.drop_duplicates(subset=["game_id"]).reset_index(drop=True)