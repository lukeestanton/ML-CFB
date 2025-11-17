# ingest/lines.py
from __future__ import annotations
from typing import Dict, List
import pandas as pd
from ml_cfb.clients.cfbd_client import get_betting_api

def _lines_to_records(lines: List[Dict]) -> List[dict]:
    records: List[dict] = []
    for game_data in lines:
        game_id = game_data.get("id")
        season = game_data.get("season")
        for line in game_data.get("lines", []) or []:
            provider = line.get("provider")
            spread = line.get("spread")
            over_under = line.get("overUnder")
            records.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "provider": provider,
                    "spread": spread,
                    "over_under": over_under,
                }
            )
    return records


def fetch_lines_for_season(client, season: int) -> pd.DataFrame:
    api = get_betting_api(client)
    lines = api.get_lines(year=season)
    df = pd.DataFrame.from_records(_lines_to_records(lines))
    if not df.empty:
        df = df[df["over_under"].notna()].copy()
    return df.reset_index(drop=True)


