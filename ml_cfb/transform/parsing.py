from __future__ import annotations

import pandas as pd

def _select_closing_totals(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty:
        return pd.DataFrame(columns=["game_id", "closing_total", "provider"])

    # Line provider preference order
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
        group["provider_rank"] = group["provider"].map(
            lambda x: provider_rank.get(x, 999)
        )
        group = group.sort_values("provider_rank")
        
        row = group.iloc[0]
        results.append({
            "game_id": game_id,
            "closing_total": row["over_under"],
            "provider": row["provider"],
        })
    
    if not results:
        return pd.DataFrame(columns=["game_id", "closing_total", "provider"])
    return pd.DataFrame(results)


def build_totals_dataset(games_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    # Only regular season games
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
        .merge(closers, on="game_id", how="inner")  # Keep only games with closing totals
        .copy()
    )

    # Only games with both closing_total and actual_total
    merged = merged[
        merged["closing_total"].notna() & merged["actual_total"].notna()
    ].copy()

    # Compute residual
    merged["actual_total_minus_total"] = (
        merged["actual_total"].astype("Float64") - merged["closing_total"].astype("Float64")
    )

    # Push flag (for if actual total == closing total)
    merged["push"] = (
        merged["actual_total"].astype(float) == merged["closing_total"].astype(float)
    )

    # Get over_result from actual_total vs closing_total
    # 1 = Over, 0 = Under, NaN = Push
    merged["over_result"] = pd.NA
    over_mask = merged["actual_total"].astype(float) > merged["closing_total"].astype(float)
    under_mask = merged["actual_total"].astype(float) < merged["closing_total"].astype(float)
    
    merged.loc[over_mask, "over_result"] = 1
    merged.loc[under_mask, "over_result"] = 0

    return merged


