from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from ml_cfb.clients.cfbd_client import get_stats_api


_TARGET_STATS = {
    "time_of_possession",
    "plays",
    "yards_per_play",
    "success_rate",
    "explosive_rate",
    "turnovers",
    "penalties",
    "penalty_yards",
}


def _camel_to_snake(name: str) -> str:
    out = []
    for idx, char in enumerate(name):
        if char.isupper() and idx != 0 and not name[idx - 1].isupper():
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _coerce_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _flatten_team_stat_record(record: Dict) -> Dict:
    categories = record.get("categories", []) or []
    stats: Dict[str, Optional[float]] = {}
    for category in categories:
        for stat in category.get("stats", []) or []:
            stat_name = stat.get("stat")
            if not stat_name:
                continue
            key = _camel_to_snake(stat_name)
            if key not in _TARGET_STATS:
                continue
            stats[key] = _coerce_numeric(stat.get("value"))

    flattened = {
        "game_id": record.get("gameId"),
        "season": record.get("season"),
        "week": record.get("week"),
        "season_type": record.get("seasonType"),
        "team": record.get("team"),
        "opponent": record.get("opponent"),
        "home_away": record.get("homeAway"),
        "start_date": record.get("startDate"),
    }
    flattened.update(stats)
    return flattened


def _team_game_stats_to_records(data: Iterable[Dict]) -> List[Dict]:
    return [_flatten_team_stat_record(item) for item in data]


def _flatten_advanced_record(record: Dict) -> Dict:
    offense = record.get("offense", {}) or {}
    return {
        "game_id": record.get("gameId"),
        "season": record.get("season"),
        "week": record.get("week"),
        "season_type": record.get("seasonType"),
        "team": record.get("team"),
        "opponent": record.get("opponent"),
        "home_away": record.get("homeAway"),
        "start_date": record.get("startDate"),
        "offensive_success_rate": _coerce_numeric(offense.get("successRate")),
        "offensive_explosiveness": _coerce_numeric(offense.get("explosiveness")),
        "plays_offense_advanced": _coerce_numeric(offense.get("plays")),
    }


def _advanced_stats_to_records(data: Iterable[Dict]) -> List[Dict]:
    return [_flatten_advanced_record(item) for item in data]


def fetch_team_game_stats_for_season(
    client,
    season: int,
    season_type: Optional[str] = None,
) -> pd.DataFrame:
    api = get_stats_api(client)
    responses: List[pd.DataFrame] = []

    season_types = [season_type] if season_type is not None else ["regular", "postseason"]
    for st in season_types:
        payload = api.get_team_game_stats(year=season, season_type=st)
        if not payload:
            continue
        df = pd.DataFrame.from_records(_team_game_stats_to_records(payload))
        responses.append(df)

    if not responses:
        return pd.DataFrame()

    df = pd.concat(responses, ignore_index=True)

    if "time_of_possession" in df.columns:
        df.rename(columns={"time_of_possession": "time_of_possession_seconds"}, inplace=True)
    if "plays" in df.columns:
        df.rename(columns={"plays": "plays_offense"}, inplace=True)

    return df


def fetch_advanced_team_game_stats_for_season(
    client,
    season: int,
    season_type: Optional[str] = None,
) -> pd.DataFrame:
    api = get_stats_api(client)
    responses: List[pd.DataFrame] = []

    season_types = [season_type] if season_type is not None else ["regular", "postseason"]
    for st in season_types:
        payload = api.get_advanced_team_game_stats(year=season, season_type=st)
        if not payload:
            continue
        df = pd.DataFrame.from_records(_advanced_stats_to_records(payload))
        responses.append(df)

    if not responses:
        return pd.DataFrame()

    df = pd.concat(responses, ignore_index=True)
    return df
