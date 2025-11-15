# ingest/advanced_stats.py
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from ml_cfb.clients.cfbd_client import StatsAPI, get_stats_api


def _get_stat_safe(stats_dict: Optional[Dict[str, Any]], key: str) -> float | None:
    if stats_dict is None:
        return None
    try:
        val = stats_dict.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _parse_advanced_stats(games: List[Dict[str, Any]]) -> List[dict]:
    records: List[dict] = []
    for game_stats in games:
        game_id = game_stats.get("gameId")
        team_name = game_stats.get("team")
        if not game_id or not team_name:
            continue
        
        offense_stats = game_stats.get("offense", {})
        defense_stats = game_stats.get("defense", {})

        records.append(
            {
                "game_id": int(game_id),
                "team": team_name,
                
                "off_ppa": _get_stat_safe(offense_stats, "ppa"),
                "off_success_rate": _get_stat_safe(offense_stats, "successRate"),
                "off_explosiveness": _get_stat_safe(offense_stats, "explosiveness"),
                
                "def_ppa": _get_stat_safe(defense_stats, "ppa"),
                "def_success_rate": _get_stat_safe(defense_stats, "successRate"),
                "def_explosiveness": _get_stat_safe(defense_stats, "explosiveness"),
            }
        )
    return records


def fetch_advanced_stats_for_season(client, season: int) -> pd.DataFrame:
    api: StatsAPI = get_stats_api(client)
    
    game_stats = api.get_advanced_game_stats(
        year=season,
        season_type="regular",
        exclude_garbage_time=True
    )

    df = pd.DataFrame.from_records(_parse_advanced_stats(game_stats))
    
    if df.empty:
        print(f"Warning: No advanced stats found for season {season}")
        return df
    
    df = df.drop_duplicates(subset=["game_id", "team"])
    
    if not df.empty:
        df["season"] = season
    return df.reset_index(drop=True)