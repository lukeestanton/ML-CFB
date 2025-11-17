# ingest/returning_production.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ml_cfb.clients.cfbd_client import StatsAPI, get_stats_api


def _get_production_safe(prod_dict: Optional[Dict[str, Any]], key: str) -> float | None:
    if prod_dict is None:
        return None
    try:
        val = prod_dict.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _parse_returning_production(data: List[Dict[str, Any]]) -> List[dict]:
    records: List[dict] = []
    for item in data:
        season = item.get("season")
        team = item.get("team")
        if not season or not team:
            continue

        records.append(
            {
                "season": int(season),
                "team": str(team),
                "conference": item.get("conference"),
                "total_ppa_pct": _get_production_safe(item, "percentPPA"),
                "passing_ppa_pct": _get_production_safe(item, "percentPassingPPA"),
                "receiving_ppa_pct": _get_production_safe(item, "percentReceivingPPA"),
                "rushing_ppa_pct": _get_production_safe(item, "percentRushingPPA"),
                "usage_pct": _get_production_safe(item, "usage"),
                "passing_usage_pct": _get_production_safe(item, "passingUsage"),
                "receiving_usage_pct": _get_production_safe(item, "receivingUsage"),
                "rushing_usage_pct": _get_production_safe(item, "rushingUsage"),
            }
        )
    return records


def fetch_returning_production_for_season(client, season: int) -> pd.DataFrame:
    api: StatsAPI = get_stats_api(client)

    try:
        production_data = api.get_returning_production(year=season)
        
        if not production_data:
            print(f"Warning: No ReturningProduction data returned for season {season}")
            return pd.DataFrame()

        df = pd.DataFrame.from_records(_parse_returning_production(production_data))

        if df.empty:
            print(f"Warning: No ReturningProduction data found for season {season}")
            return df

        df = df.drop_duplicates(subset=["season", "team"])
        
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Warning: Error fetching ReturningProduction for season {season}: {e}")
        return pd.DataFrame()

