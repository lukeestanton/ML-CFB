# ingest/team_sp.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ml_cfb.clients.cfbd_client import RatingsAPI, get_ratings_api


def _get_rating_safe(rating_dict: Optional[Dict[str, Any]], key: str) -> float | None:
    if rating_dict is None:
        return None
    try:
        val = rating_dict.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _parse_team_sp_ratings(ratings: List[Dict[str, Any]]) -> List[dict]:
    records: List[dict] = []
    for rating in ratings:
        year = rating.get("year")
        team = rating.get("team")
        if not year or not team:
            continue

        offense_stats = rating.get("offense", {})
        defense_stats = rating.get("defense", {})
        special_teams_stats = rating.get("specialTeams", {})

        records.append(
            {
                "year": int(year),
                "team": str(team),
                "conference": rating.get("conference"),
                "rating": _get_rating_safe(rating, "rating"),
                "ranking": rating.get("ranking"),
                "second_order_wins": _get_rating_safe(rating, "secondOrderWins"),
                "sos": _get_rating_safe(rating, "sos"),
                "sp_offense": _get_rating_safe(offense_stats, "rating"),
                "sp_offense_ranking": offense_stats.get("ranking"),
                "sp_defense": _get_rating_safe(defense_stats, "rating"),
                "sp_defense_ranking": defense_stats.get("ranking"),
                "sp_special_teams": _get_rating_safe(special_teams_stats, "rating"),
                "sp_special_teams_ranking": special_teams_stats.get("ranking"),
            }
        )
    return records


def fetch_team_sp_for_season(client, season: int) -> pd.DataFrame:
    api: RatingsAPI = get_ratings_api(client)

    sp_ratings = api.get_team_sp_ratings(year=season)

    df = pd.DataFrame.from_records(_parse_team_sp_ratings(sp_ratings))

    if df.empty:
        print(f"Warning: No TeamSP ratings found for season {season}")
        return df

    df = df.drop_duplicates(subset=["year", "team"])
    
    if not df.empty:
        df["season"] = season
    
    return df.reset_index(drop=True)

