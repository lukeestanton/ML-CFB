# scripts/debug_weather.py
from __future__ import annotations

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_cfb.clients.cfbd_client import build_cfbd_client, get_games_api
from ml_cfb.config import load_settings


def debug_weather():
    settings = load_settings()
    if not settings.cfbd_api_key:
        print("ERROR: CFBD_API_KEY not set.")
        return

    client = build_cfbd_client(settings.cfbd_api_key)
    api = get_games_api(client)

    try:
        print("--- Fetching weather for 2022, Week 1 ---")
        weather_data = api.get_weather(
            year=2022, 
            week=1,
            season_type="regular"
        )
        
        if not weather_data:
            print("--- DEBUG: API returned NO data (empty list). ---")
            return

        print("\n--- DEBUG: API returned data. Printing first game object: ---")
        print(json.dumps(weather_data[0], indent=2))
    
    except Exception as e:
        print(f"\n--- DEBUG: API call failed ---")
        print(e)

if __name__ == "__main__":
    debug_weather()