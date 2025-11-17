# ingest/team_fpi.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ml_cfb.clients.cfbd_client import RatingsAPI, get_ratings_api


def _get_fpi_safe(fpi_dict: Optional[Dict[str, Any]], key: str) -> float | None:
    if fpi_dict is None:
        return None
    try:
        val = fpi_dict.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _parse_team_fpi_ratings(ratings: List[Dict[str, Any]]) -> List[dict]:
    records: List[dict] = []
    for rating in ratings:
        year = rating.get("year")
        team = rating.get("team")
        if not year or not team:
            continue

        resume_ranks = rating.get("resumeRanks", {})
        efficiencies = rating.get("efficiencies", {})

        records.append(
            {
                "year": int(year),
                "team": str(team),
                "conference": rating.get("conference"),
                "fpi": _get_fpi_safe(rating, "fpi"),
                "resume_rank": resume_ranks.get("rank"),
                "strength_of_record_rank": resume_ranks.get("strengthOfRecordRank"),
                "avg_wp_rank": resume_ranks.get("avgWpRank"),
                "offense_efficiency": _get_fpi_safe(efficiencies, "offense"),
                "defense_efficiency": _get_fpi_safe(efficiencies, "defense"),
                "special_teams_efficiency": _get_fpi_safe(efficiencies, "specialTeams"),
            }
        )
    return records


def fetch_team_fpi_for_season(client, season: int) -> pd.DataFrame:
    api: RatingsAPI = get_ratings_api(client)

    try:
        fpi_ratings = api.get_team_fpi_ratings(year=season)
        
        if not fpi_ratings:
            print(f"Warning: No TeamFPI ratings returned for season {season}")
            return pd.DataFrame()

        df = pd.DataFrame.from_records(_parse_team_fpi_ratings(fpi_ratings))

        if df.empty:
            print(f"Warning: No TeamFPI ratings found for season {season}")
            return df

        df = df.drop_duplicates(subset=["year", "team"])
        
        if not df.empty:
            df["season"] = season
        
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Warning: Error fetching TeamFPI for season {season}: {e}")
        return pd.DataFrame()

