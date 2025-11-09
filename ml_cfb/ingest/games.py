from __future__ import annotations
from typing import List, Optional
import pandas as pd
from ml_cfb.clients.cfbd_client import get_games_api

def _games_to_records(games: List) -> List[dict]:
    records: List[dict] = []
    for game in games:
        d = game.to_dict()
        records.append(
            {
                "game_id": d.get("id"),
                "season": d.get("season"),
                "week": d.get("week"),
                "season_type": d.get("season_type"),
                "start_date": d.get("start_date"),
                "neutral_site": d.get("neutral_site"),
                "conference_game": d.get("conference_game"),
                "home_team": d.get("home_team"),
                "away_team": d.get("away_team"),
                "home_points": d.get("home_points"),
                "away_points": d.get("away_points"),
                "venue": d.get("venue"),
            }
        )
    return records

def fetch_games_for_season(
    client,
    season: int,
    season_type: Optional[str] = None,
) -> pd.DataFrame:
    api = get_games_api(client)
    dfs: List[pd.DataFrame] = []

    if season_type is None:
        for st in ("regular", "postseason"):
            games = api.get_games(year=season, season_type=st)
            dfs.append(pd.DataFrame.from_records(_games_to_records(games)))
    else:
        games = api.get_games(year=season, season_type=season_type)
        dfs.append(pd.DataFrame.from_records(_games_to_records(games)))

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not df.empty:
        df["actual_total"] = (df["home_points"].fillna(0) + df["away_points"].fillna(0)).astype("Int64")
    return df


