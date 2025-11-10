from __future__ import annotations

from typing import Dict

import pandas as pd

ROLLING_WINDOW = 3


def _select_closing_totals(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty:
        return pd.DataFrame(columns=["game_id", "closing_total", "provider"])

    provider_preference = [
        "William Hill (New Jersey)",
        "DraftKings",
        "ESPN Bet",
        "Bovada",
        "Caesars Sportsbook (Colorado)",
    ]
    provider_rank = {prov: idx for idx, prov in enumerate(provider_preference)}

    results = []
    for game_id, group in lines_df.groupby("game_id"):
        group = group.copy()
        group["provider_rank"] = group["provider"].map(lambda x: provider_rank.get(x, 999))
        group = group.sort_values("provider_rank")
        row = group.iloc[0]
        results.append(
            {
                "game_id": game_id,
                "closing_total": row.get("over_under"),
                "provider": row.get("provider"),
            }
        )

    if not results:
        return pd.DataFrame(columns=["game_id", "closing_total", "provider"])
    return pd.DataFrame(results)


def _prepare_team_stats(
    team_stats_df: pd.DataFrame | None,
    advanced_stats_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if team_stats_df is not None and not team_stats_df.empty:
        stats = team_stats_df.copy()
    else:
        stats = pd.DataFrame(columns=["game_id", "team"])

    for col in [
        "game_id",
        "team",
        "time_of_possession_seconds",
        "plays_offense",
        "yards_per_play",
        "success_rate",
        "explosive_rate",
        "turnovers",
        "penalties",
        "penalty_yards",
    ]:
        if col not in stats.columns:
            stats[col] = pd.NA

    stats = stats[
        [
            "game_id",
            "team",
            "time_of_possession_seconds",
            "plays_offense",
            "yards_per_play",
            "success_rate",
            "explosive_rate",
            "turnovers",
            "penalties",
            "penalty_yards",
        ]
    ]

    if advanced_stats_df is not None and not advanced_stats_df.empty:
        adv = advanced_stats_df.copy()
        for col in [
            "game_id",
            "team",
            "offensive_success_rate",
            "offensive_explosiveness",
            "plays_offense_advanced",
        ]:
            if col not in adv.columns:
                adv[col] = pd.NA
        adv = adv[
            [
                "game_id",
                "team",
                "offensive_success_rate",
                "offensive_explosiveness",
                "plays_offense_advanced",
            ]
        ]
        stats = stats.merge(adv, on=["game_id", "team"], how="left")
    else:
        stats["offensive_success_rate"] = pd.NA
        stats["offensive_explosiveness"] = pd.NA
        stats["plays_offense_advanced"] = pd.NA

    stats["success_rate_offense"] = stats["offensive_success_rate"].combine_first(
        stats["success_rate"]
    )
    stats["explosiveness_offense"] = stats["offensive_explosiveness"].combine_first(
        stats["explosive_rate"]
    )
    stats["plays_for_pace"] = stats["plays_offense"].combine_first(
        stats["plays_offense_advanced"]
    )

    numeric_columns = [
        "time_of_possession_seconds",
        "plays_offense",
        "yards_per_play",
        "turnovers",
        "penalties",
        "penalty_yards",
        "offensive_success_rate",
        "offensive_explosiveness",
        "plays_offense_advanced",
        "success_rate_offense",
        "explosiveness_offense",
        "plays_for_pace",
    ]
    for col in numeric_columns:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    stats = stats.drop_duplicates(subset=["game_id", "team"])

    return stats[
        [
            "game_id",
            "team",
            "time_of_possession_seconds",
            "plays_for_pace",
            "yards_per_play",
            "turnovers",
            "penalties",
            "penalty_yards",
            "success_rate_offense",
            "explosiveness_offense",
        ]
    ]


def _compute_team_game_view(
    merged_games: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.DataFrame:
    home = merged_games.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["team_points"] = home["home_points"]
    home["opponent_points"] = home["away_points"]
    home["is_home"] = True

    away = merged_games.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["team_points"] = away["away_points"]
    away["opponent_points"] = away["home_points"]
    away["is_home"] = False

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["score_margin"] = team_games["team_points"].astype(float) - team_games[
        "opponent_points"
    ].astype(float)
    team_games["start_date"] = pd.to_datetime(team_games["start_date"], errors="coerce")

    if not team_stats.empty:
        team_games = team_games.merge(team_stats, on=["game_id", "team"], how="left")
    else:
        for col in [
            "time_of_possession_seconds",
            "plays_for_pace",
            "yards_per_play",
            "turnovers",
            "penalties",
            "penalty_yards",
            "success_rate_offense",
            "explosiveness_offense",
        ]:
            team_games[col] = pd.NA

    team_games["pace_plays_per_minute"] = (
        team_games["plays_for_pace"]
        / (team_games["time_of_possession_seconds"] / 60.0)
    )
    team_games.loc[
        (team_games["time_of_possession_seconds"].isna())
        | (team_games["time_of_possession_seconds"] <= 0),
        "pace_plays_per_minute",
    ] = pd.NA

    return team_games


def _add_rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    team_games = team_games.sort_values(
        by=["team", "start_date", "season", "week", "game_id"],
        kind="mergesort",
    )

    feature_sources: Dict[str, str] = {
        "points_for": "team_points",
        "points_against": "opponent_points",
        "score_margin": "score_margin",
        "time_of_possession_seconds": "time_of_possession_seconds",
        "pace_plays_per_minute": "pace_plays_per_minute",
        "yards_per_play": "yards_per_play",
        "success_rate_offense": "success_rate_offense",
        "explosiveness_offense": "explosiveness_offense",
        "turnovers": "turnovers",
        "penalties": "penalties",
        "penalty_yards": "penalty_yards",
    }

    for source_col in set(feature_sources.values()):
        if source_col in team_games.columns:
            team_games[source_col] = pd.to_numeric(team_games[source_col], errors="coerce")

    grouped = team_games.groupby("team", group_keys=False)
    for feature_name, source_col in feature_sources.items():
        result_col = f"{feature_name}_rolling_{ROLLING_WINDOW}"
        if source_col not in team_games.columns:
            team_games[result_col] = pd.NA
            continue
        team_games[result_col] = grouped[source_col].apply(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        )

    return team_games


def build_totals_dataset(
    games_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    team_stats_df: pd.DataFrame | None = None,
    advanced_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df[games_df["season_type"] == "regular"].copy()
    if games_df.empty:
        return pd.DataFrame()

    closers = _select_closing_totals(lines_df=lines_df)
    merged = (
        games_df[
            [
                "game_id",
                "season",
                "week",
                "start_date",
                "neutral_site",
                "conference_game",
                "home_team",
                "away_team",
                "home_points",
                "away_points",
                "actual_total",
            ]
        ]
        .merge(closers, on="game_id", how="inner")
        .copy()
    )

    merged = merged[
        merged["closing_total"].notna() & merged["actual_total"].notna()
    ].copy()

    merged["actual_total_minus_total"] = (
        merged["actual_total"].astype("Float64") - merged["closing_total"].astype("Float64")
    )
    merged["push"] = (
        merged["actual_total"].astype(float) == merged["closing_total"].astype(float)
    )

    merged["over_result"] = pd.NA
    over_mask = merged["actual_total"].astype(float) > merged["closing_total"].astype(float)
    under_mask = merged["actual_total"].astype(float) < merged["closing_total"].astype(float)
    merged.loc[over_mask, "over_result"] = 1
    merged.loc[under_mask, "over_result"] = 0

    prepared_team_stats = _prepare_team_stats(team_stats_df, advanced_stats_df)
    team_games = _compute_team_game_view(merged, prepared_team_stats)
    team_games = _add_rolling_features(team_games)

    rolling_columns = [
        col for col in team_games.columns if col.endswith(f"rolling_{ROLLING_WINDOW}")
    ]

    home_features = (
        team_games[team_games["is_home"]][["game_id"] + rolling_columns]
        .copy()
        .rename(columns={col: f"{col}_home" for col in rolling_columns})
    )
    away_features = (
        team_games[~team_games["is_home"]][["game_id"] + rolling_columns]
        .copy()
        .rename(columns={col: f"{col}_away" for col in rolling_columns})
    )

    enriched = (
        merged.merge(home_features, on="game_id", how="left")
        .merge(away_features, on="game_id", how="left")
        .copy()
    )

    return enriched
