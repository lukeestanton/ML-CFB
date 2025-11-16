# transform/parsing.py
from __future__ import annotations
import pandas as pd

def _select_closing_totals(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty:
        return pd.DataFrame(columns=["game_id", "closing_total", "closing_spread", "provider"])
    provider_preference = [
        "William Hill (New Jersey)", "DraftKings", "ESPN Bet",
        "Bovada", "Caesars Sportsbook (Colorado)",
    ]
    provider_rank = {prov: idx for idx, prov in enumerate(provider_preference)}
    results = []
    for game_id, group in lines_df.groupby("game_id"):
        group = group.copy()
        try:
            group_game_id = int(game_id)
        except ValueError:
            continue
        group["provider_rank"] = group["provider"].map(lambda x: provider_rank.get(x, 999))
        group = group.sort_values("provider_rank")
        row = group.iloc[0]
        results.append({
            "game_id": group_game_id,
            "closing_total": row["over_under"],
            "closing_spread": row.get("spread"),
            "provider": row["provider"],
        })
    if not results:
        return pd.DataFrame(columns=["game_id", "closing_total", "closing_spread", "provider"])
    df = pd.DataFrame(results)
    df["game_id"] = df["game_id"].astype(int)
    return df


def build_totals_dataset(games_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty: return pd.DataFrame()
    games_df = games_df[games_df["season_type"] == "regular"].copy()
    if games_df.empty: return pd.DataFrame()
    closers = _select_closing_totals(lines_df=lines_df)
    games_df["game_id"] = pd.to_numeric(games_df["game_id"], errors="coerce").dropna().astype(int)
    closers["game_id"] = closers["game_id"].astype(int)
    merged = (
        games_df[
            [ "game_id", "season", "week", "start_date", "neutral_site",
              "conference_game", "home_team", "away_team", "home_points",
              "away_points", "actual_total", "venue_id"
            ]
        ]
        .merge(closers, on="game_id", how="inner")
        .copy()
    )
    merged = merged[merged["closing_total"].notna() & merged["actual_total"].notna()].copy()
    merged = merged.drop_duplicates(subset=["game_id"])
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
    return merged.reset_index(drop=True)


def build_training_dataset(
    games_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    advanced_stats_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df[games_df["season_type"] == "regular"].copy()
    if games_df.empty:
        return pd.DataFrame()

    closers = _select_closing_totals(lines_df=lines_df)
    
    games_df["game_id"] = pd.to_numeric(games_df["game_id"], errors="coerce").dropna().astype(int)
    closers["game_id"] = closers["game_id"].astype(int)

    merged = (
        games_df[
            [ "game_id", "start_date", "season", "week", "neutral_site",
              "conference_game", "home_team", "away_team", "home_points",
              "away_points",
            ]
        ]
        .merge(closers, on="game_id", how="inner")
        .copy()
    )
    
    merged = merged.drop_duplicates(subset=["game_id"])
    
    if not advanced_stats_df.empty:
        if "season" in advanced_stats_df.columns:
            advanced_stats_df = advanced_stats_df.drop(columns="season")
        advanced_stats_df["game_id"] = pd.to_numeric(advanced_stats_df["game_id"], errors="coerce").dropna().astype(int)
        advanced_stats_df = advanced_stats_df.drop_duplicates(subset=["game_id", "team"])
        
        home_cols_to_rename = {
            "team": "home_team", "off_ppa": "home_ppa_offense",
            "off_success_rate": "home_success_rate", "off_explosiveness": "home_explosiveness",
            "def_ppa": "home_ppa_defense", "def_success_rate": "home_def_success_rate", 
            "def_explosiveness": "home_def_explosiveness",
        }
        home_stats_df = advanced_stats_df.rename(columns=home_cols_to_rename)
        home_stats_cols = ["game_id", "home_team"] + list(home_cols_to_rename.values())[1:]
        home_stats_df = home_stats_df[home_stats_cols]
        
        merged = merged.merge(home_stats_df, on=["game_id", "home_team"], how="left")
        
        away_cols_to_rename = {
            "team": "away_team", "off_ppa": "away_ppa_offense",
            "off_success_rate": "away_success_rate", "off_explosiveness": "away_explosiveness",
            "def_ppa": "away_ppa_defense", "def_success_rate": "away_def_success_rate",
            "def_explosiveness": "away_def_explosiveness",
        }
        away_stats_df = advanced_stats_df.rename(columns=away_cols_to_rename)
        away_stats_cols = ["game_id", "away_team"] + list(away_cols_to_rename.values())[1:]
        away_stats_df = away_stats_df[away_stats_cols]

        merged = merged.merge(away_stats_df, on=["game_id", "away_team"], how="left")

    if not weather_df.empty:
        weather_df["game_id"] = pd.to_numeric(weather_df["game_id"], errors="coerce").dropna().astype(int)
        weather_df = weather_df.drop_duplicates(subset=["game_id"])
        
        if "season" in weather_df.columns:
            weather_df = weather_df.drop(columns="season")
            
        merged = merged.merge(weather_df, on="game_id", how="left")

    merged = merged[
        merged["closing_total"].notna()
        & merged["home_points"].notna()
        & merged["away_points"].notna()
    ].copy()

    merged = merged.rename(
        columns={
            "closing_total": "ou_line",
            "start_date": "date",
            "closing_spread": "spread",
        }
    )

    merged["date"] = pd.to_datetime(merged["date"])
    merged["spread"] = merged["spread"].astype("Float64")
    merged["season"] = merged["season"].astype("Int64")
    merged["week"] = merged["week"].astype("Int64")

    merged = merged.sort_values("date").reset_index(drop=True)

    required_columns = [
        "home_team", "away_team", "home_points", "away_points",
        "ou_line", "date", "spread", "season", "week",
        "neutral_site", "conference_game",
        
        "home_ppa_offense", "home_success_rate", "home_explosiveness",
        "home_ppa_defense", "home_def_success_rate", "home_def_explosiveness",
        "away_ppa_offense", "away_success_rate", "away_explosiveness",
        "away_ppa_defense", "away_def_success_rate", "away_def_explosiveness",
        
        "is_dome", "temperature", "wind_speed", "precipitation"
    ]
    
    final_columns = [col for col in required_columns if col in merged.columns]
    merged = merged[final_columns].copy()

    return merged