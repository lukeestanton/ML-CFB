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
    
    game_cols = [ "game_id", "season", "week", "start_date", "neutral_site",
                  "conference_game", "home_team", "away_team", "home_points",
                  "away_points", "actual_total"]
    if "venue_id" in games_df.columns:
        game_cols.append("venue_id")

    merged = (
        games_df[game_cols]
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
    team_sp_df: pd.DataFrame,
    team_fpi_df: pd.DataFrame,
    returning_prod_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the modeling dataset for totals.
    """
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

    if not team_sp_df.empty and "season" in merged.columns:
        team_sp_df = team_sp_df.copy()
        if "season" not in team_sp_df.columns and "year" in team_sp_df.columns:
            team_sp_df["season"] = team_sp_df["year"]
        
        sp_rating_cols = [col for col in team_sp_df.columns 
                         if col not in ["team", "season", "year", "conference"]]
        
        home_sp_df = team_sp_df[["team", "season"] + sp_rating_cols].copy()
        home_sp_df = home_sp_df.rename(columns={
            "team": "home_team",
            "rating": "home_sp_rating",
            "ranking": "home_sp_ranking",
            "second_order_wins": "home_sp_second_order_wins",
            "sos": "home_sp_sos",
            "sp_offense": "home_sp_offense",
            "sp_offense_ranking": "home_sp_offense_ranking",
            "sp_defense": "home_sp_defense",
            "sp_defense_ranking": "home_sp_defense_ranking",
            "sp_special_teams": "home_sp_special_teams",
            "sp_special_teams_ranking": "home_sp_special_teams_ranking",
        }, errors="ignore")
        home_sp_df = home_sp_df.drop_duplicates(subset=["home_team", "season"])
        merged = merged.merge(home_sp_df, on=["home_team", "season"], how="left")
        
        away_sp_df = team_sp_df[["team", "season"] + sp_rating_cols].copy()
        away_sp_df = away_sp_df.rename(columns={
            "team": "away_team",
            "rating": "away_sp_rating",
            "ranking": "away_sp_ranking",
            "second_order_wins": "away_sp_second_order_wins",
            "sos": "away_sp_sos",
            "sp_offense": "away_sp_offense",
            "sp_offense_ranking": "away_sp_offense_ranking",
            "sp_defense": "away_sp_defense",
            "sp_defense_ranking": "away_sp_defense_ranking",
            "sp_special_teams": "away_sp_special_teams",
            "sp_special_teams_ranking": "away_sp_special_teams_ranking",
        }, errors="ignore")
        away_sp_df = away_sp_df.drop_duplicates(subset=["away_team", "season"])
        merged = merged.merge(away_sp_df, on=["away_team", "season"], how="left")

    if not team_fpi_df.empty and "season" in merged.columns:
        team_fpi_df = team_fpi_df.copy()
        if "season" not in team_fpi_df.columns and "year" in team_fpi_df.columns:
            team_fpi_df["season"] = team_fpi_df["year"]
        
        fpi_cols = [col for col in team_fpi_df.columns 
                   if col not in ["team", "season", "year", "conference"]]
        
        home_fpi_df = team_fpi_df[["team", "season"] + fpi_cols].copy()
        home_fpi_df = home_fpi_df.rename(columns={
            "team": "home_team",
            "fpi": "home_fpi",
            "resume_rank": "home_fpi_resume_rank",
            "strength_of_record_rank": "home_fpi_sor_rank",
            "avg_wp_rank": "home_fpi_avg_wp_rank",
            "offense_efficiency": "home_fpi_offense",
            "defense_efficiency": "home_fpi_defense",
            "special_teams_efficiency": "home_fpi_special_teams",
        }, errors="ignore")
        home_fpi_df = home_fpi_df.drop_duplicates(subset=["home_team", "season"])
        merged = merged.merge(home_fpi_df, on=["home_team", "season"], how="left")
        
        away_fpi_df = team_fpi_df[["team", "season"] + fpi_cols].copy()
        away_fpi_df = away_fpi_df.rename(columns={
            "team": "away_team",
            "fpi": "away_fpi",
            "resume_rank": "away_fpi_resume_rank",
            "strength_of_record_rank": "away_fpi_sor_rank",
            "avg_wp_rank": "away_fpi_avg_wp_rank",
            "offense_efficiency": "away_fpi_offense",
            "defense_efficiency": "away_fpi_defense",
            "special_teams_efficiency": "away_fpi_special_teams",
        }, errors="ignore")
        away_fpi_df = away_fpi_df.drop_duplicates(subset=["away_team", "season"])
        merged = merged.merge(away_fpi_df, on=["away_team", "season"], how="left")

    if not returning_prod_df.empty and "season" in merged.columns:
        returning_prod_df = returning_prod_df.copy()
        
        rp_cols = [col for col in returning_prod_df.columns 
                  if col not in ["team", "season", "conference"]]
        
        home_rp_df = returning_prod_df[["team", "season"] + rp_cols].copy()
        home_rp_df = home_rp_df.rename(columns={
            "team": "home_team",
            "total_ppa_pct": "home_returning_ppa_pct",
            "passing_ppa_pct": "home_returning_passing_ppa_pct",
            "receiving_ppa_pct": "home_returning_receiving_ppa_pct",
            "rushing_ppa_pct": "home_returning_rushing_ppa_pct",
            "usage_pct": "home_returning_usage_pct",
            "passing_usage_pct": "home_returning_passing_usage_pct",
            "receiving_usage_pct": "home_returning_receiving_usage_pct",
            "rushing_usage_pct": "home_returning_rushing_usage_pct",
        }, errors="ignore")
        home_rp_df = home_rp_df.drop_duplicates(subset=["home_team", "season"])
        merged = merged.merge(home_rp_df, on=["home_team", "season"], how="left")
        
        away_rp_df = returning_prod_df[["team", "season"] + rp_cols].copy()
        away_rp_df = away_rp_df.rename(columns={
            "team": "away_team",
            "total_ppa_pct": "away_returning_ppa_pct",
            "passing_ppa_pct": "away_returning_passing_ppa_pct",
            "receiving_ppa_pct": "away_returning_receiving_ppa_pct",
            "rushing_ppa_pct": "away_returning_rushing_ppa_pct",
            "usage_pct": "away_returning_usage_pct",
            "passing_usage_pct": "away_returning_passing_usage_pct",
            "receiving_usage_pct": "away_returning_receiving_usage_pct",
            "rushing_usage_pct": "away_returning_rushing_usage_pct",
        }, errors="ignore")
        away_rp_df = away_rp_df.drop_duplicates(subset=["away_team", "season"])
        merged = merged.merge(away_rp_df, on=["away_team", "season"], how="left")

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
        
        "home_sp_rating", "home_sp_ranking", "home_sp_sos",
        "home_sp_offense", "home_sp_offense_ranking",
        "home_sp_defense", "home_sp_defense_ranking",
        "home_sp_special_teams", "home_sp_special_teams_ranking",
        "away_sp_rating", "away_sp_ranking", "away_sp_sos",
        "away_sp_offense", "away_sp_offense_ranking",
        "away_sp_defense", "away_sp_defense_ranking",
        "away_sp_special_teams", "away_sp_special_teams_ranking",
        
        "home_fpi", "home_fpi_resume_rank", "home_fpi_sor_rank", "home_fpi_avg_wp_rank",
        "home_fpi_offense", "home_fpi_defense", "home_fpi_special_teams",
        "away_fpi", "away_fpi_resume_rank", "away_fpi_sor_rank", "away_fpi_avg_wp_rank",
        "away_fpi_offense", "away_fpi_defense", "away_fpi_special_teams",
        
        "home_returning_ppa_pct", "home_returning_passing_ppa_pct",
        "home_returning_receiving_ppa_pct", "home_returning_rushing_ppa_pct",
        "home_returning_usage_pct", "home_returning_passing_usage_pct",
        "home_returning_receiving_usage_pct", "home_returning_rushing_usage_pct",
        "away_returning_ppa_pct", "away_returning_passing_ppa_pct",
        "away_returning_receiving_ppa_pct", "away_returning_rushing_ppa_pct",
        "away_returning_usage_pct", "away_returning_passing_usage_pct",
        "away_returning_receiving_usage_pct", "away_returning_rushing_usage_pct",
    ]
    
    final_columns = [col for col in required_columns if col in merged.columns]
    merged = merged[final_columns].copy()

    return merged