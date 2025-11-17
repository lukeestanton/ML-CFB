# train_model.py
"""
Usage:
    python train_model.py

The script expects `python scripts/build_dataset.py` has already been run so
that `data/processed/training_dataset.csv` exists.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)


class CFBBettingModel:
    def __init__(self) -> None:
        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.model: xgb.XGBClassifier | None = None
        self.calibrated_model: CalibratedClassifierCV | None = None

    def load_data(self, file_path: str | Path) -> None:
        df = pd.read_csv(file_path, parse_dates=["date"])

        df["total_points"] = df["home_points"] + df["away_points"]

        push_mask = df["total_points"] == df["ou_line"]
        df = df.loc[~push_mask].copy()

        df["went_over"] = (df["total_points"] > df["ou_line"]).astype(int)

        df = df.sort_values("date").reset_index(drop=True)
        self.data = df

    def create_features(self) -> None:
        if self.data is None:
            raise ValueError("Data must be loaded before creating features.")

        cols_to_fill = [
            "home_ppa_offense", "home_success_rate", "home_ppa_defense",
            "away_ppa_offense", "away_success_rate", "away_ppa_defense",
        ]
        
        cols_found = [col for col in cols_to_fill if col in self.data.columns]
        if cols_found:
            print(f"--- Imputing NaNs for: {cols_found} ---")
            self.data[cols_found] = self.data[cols_found].fillna(0.0)
        else:
            print("--- No advanced stats columns found. ---")
        
        sp_rating_cols = [
            "home_sp_rating", "home_sp_ranking", "home_sp_sos",
            "home_sp_offense", "home_sp_offense_ranking",
            "home_sp_defense", "home_sp_defense_ranking",
            "home_sp_special_teams", "home_sp_special_teams_ranking",
            "away_sp_rating", "away_sp_ranking", "away_sp_sos",
            "away_sp_offense", "away_sp_offense_ranking",
            "away_sp_defense", "away_sp_defense_ranking",
            "away_sp_special_teams", "away_sp_special_teams_ranking",
        ]
        sp_cols_found = [col for col in sp_rating_cols if col in self.data.columns]
        
        sp_lookup: dict[tuple[int, str], dict[str, float]] = {}
        if sp_cols_found and "season" in self.data.columns:
            for season in self.data["season"].unique():
                if pd.isna(season):
                    continue
                current_season = int(season)
                prev_season = current_season - 1
                prev_season_data = self.data[self.data["season"] == prev_season]
                
                if prev_season_data.empty:
                    continue
                
                prev_season_data_sorted = prev_season_data.sort_values("date", ascending=False).copy()
                team_sp_ratings: dict[str, dict[str, float]] = {}
                
                for _, row in prev_season_data_sorted.iterrows():
                    team = row["home_team"]
                    if pd.notna(team) and team not in team_sp_ratings:
                        team_sp_ratings[team] = {}
                        for col in sp_cols_found:
                            if col.startswith("home_sp_") and pd.notna(row.get(col)):
                                team_col = col.replace("home_sp_", "sp_")
                                try:
                                    team_sp_ratings[team][team_col] = float(row[col])
                                except (ValueError, TypeError):
                                    pass
                
                for _, row in prev_season_data_sorted.iterrows():
                    team = row["away_team"]
                    if pd.notna(team) and team not in team_sp_ratings:
                        team_sp_ratings[team] = {}
                        for col in sp_cols_found:
                            if col.startswith("away_sp_") and pd.notna(row.get(col)):
                                team_col = col.replace("away_sp_", "sp_")
                                try:
                                    team_sp_ratings[team][team_col] = float(row[col])
                                except (ValueError, TypeError):
                                    pass
                
                for team, ratings in team_sp_ratings.items():
                    key = (current_season, team)
                    sp_lookup[key] = ratings
        
        fpi_rating_cols = [
            "home_fpi", "home_fpi_resume_rank", "home_fpi_sor_rank", "home_fpi_avg_wp_rank",
            "home_fpi_offense", "home_fpi_defense", "home_fpi_special_teams",
            "away_fpi", "away_fpi_resume_rank", "away_fpi_sor_rank", "away_fpi_avg_wp_rank",
            "away_fpi_offense", "away_fpi_defense", "away_fpi_special_teams",
        ]
        fpi_cols_found = [col for col in fpi_rating_cols if col in self.data.columns]
        
        fpi_lookup: dict[tuple[int, str], dict[str, float]] = {}
        if fpi_cols_found and "season" in self.data.columns:
            for season in self.data["season"].unique():
                if pd.isna(season):
                    continue
                current_season = int(season)
                prev_season = current_season - 1
                prev_season_data = self.data[self.data["season"] == prev_season]
                
                if prev_season_data.empty:
                    continue
                
                prev_season_data_sorted = prev_season_data.sort_values("date", ascending=False).copy()
                team_fpi_ratings: dict[str, dict[str, float]] = {}
                
                for _, row in prev_season_data_sorted.iterrows():
                    team = row["home_team"]
                    if pd.notna(team) and team not in team_fpi_ratings:
                        team_fpi_ratings[team] = {}
                        for col in fpi_cols_found:
                            if col.startswith("home_fpi"):
                                if col == "home_fpi":
                                    team_col = "fpi"
                                elif col.startswith("home_fpi_"):
                                    team_col = col.replace("home_fpi_", "fpi_")
                                else:
                                    continue
                                
                                if pd.notna(row.get(col)):
                                    try:
                                        team_fpi_ratings[team][team_col] = float(row[col])
                                    except (ValueError, TypeError):
                                        pass
                
                for _, row in prev_season_data_sorted.iterrows():
                    team = row["away_team"]
                    if pd.notna(team) and team not in team_fpi_ratings:
                        team_fpi_ratings[team] = {}
                        for col in fpi_cols_found:
                            if col.startswith("away_fpi"):
                                if col == "away_fpi":
                                    team_col = "fpi"
                                elif col.startswith("away_fpi_"):
                                    team_col = col.replace("away_fpi_", "fpi_")
                                else:
                                    continue
                                
                                if pd.notna(row.get(col)):
                                    try:
                                        team_fpi_ratings[team][team_col] = float(row[col])
                                    except (ValueError, TypeError):
                                        pass
                
                for team, ratings in team_fpi_ratings.items():
                    key = (current_season, team)
                    fpi_lookup[key] = ratings
            
        df = self.data
        feature_rows: list[dict[str, float]] = []

        for _, game in df.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            game_date = game["date"]
            game_season = int(game.get("season", 0)) if pd.notna(game.get("season")) else 0

            home_history = df[
                (
                    (df["home_team"] == home_team)
                    | (df["away_team"] == home_team)
                )
                & (df["date"] < game_date)
            ].tail(5)

            away_history = df[
                (
                    (df["home_team"] == away_team)
                    | (df["away_team"] == away_team)
                )
                & (df["date"] < game_date)
            ].tail(5)

            if len(home_history) < 2 or len(away_history) < 2:
                continue

            hp_for, hp_against, h_totals = [], [], []
            h_residuals, h_over_flags = [], []
            h_ppa_off, h_success_rate, h_ppa_def = [], [], []
            
            for _, hg in home_history.iterrows():
                if hg["home_team"] == home_team:
                    pf, pa = hg["home_points"], hg["away_points"]
                    if cols_found:
                        h_ppa_off.append(hg["home_ppa_offense"])
                        h_success_rate.append(hg["home_success_rate"])
                        h_ppa_def.append(hg["home_ppa_defense"])
                else:
                    pf, pa = hg["away_points"], hg["home_points"]
                    if cols_found:
                        h_ppa_off.append(hg["away_ppa_offense"])
                        h_success_rate.append(hg["away_success_rate"])
                        h_ppa_def.append(hg["away_ppa_defense"])
                total = hg["total_points"]; hp_for.append(pf); hp_against.append(pa); h_totals.append(total)
                if pd.notna(hg["ou_line"]):
                    residual = float(total - hg["ou_line"]); h_residuals.append(residual)
                    h_over_flags.append(1.0 if total > hg["ou_line"] else 0.0)

            ap_for, ap_against, a_totals = [], [], []
            a_residuals, a_over_flags = [], []
            a_ppa_off, a_success_rate, a_ppa_def = [], [], []

            for _, ag in away_history.iterrows():
                if ag["home_team"] == away_team:
                    pf, pa = ag["home_points"], ag["away_points"]
                    if cols_found:
                        a_ppa_off.append(ag["home_ppa_offense"])
                        a_success_rate.append(ag["home_success_rate"])
                        a_ppa_def.append(ag["home_ppa_defense"])
                else:
                    pf, pa = ag["away_points"], ag["home_points"]
                    if cols_found:
                        a_ppa_off.append(ag["away_ppa_offense"])
                        a_success_rate.append(ag["away_success_rate"])
                        a_ppa_def.append(ag["away_ppa_defense"])
                total = ag["total_points"]; ap_for.append(pf); ap_against.append(pa); a_totals.append(total)
                if pd.notna(ag["ou_line"]):
                    residual = float(total - ag["ou_line"]); a_residuals.append(residual)
                    a_over_flags.append(1.0 if total > ag["ou_line"] else 0.0)

            if len(h_residuals) == 0 or len(a_residuals) == 0:
                continue

            home_avg_pts = float(np.nanmean(hp_for)); home_avg_pts_allowed = float(np.nanmean(hp_against))
            home_avg_total = float(np.nanmean(h_totals)); home_std_total = float(np.nanstd(h_totals))
            home_avg_residual_total = float(np.nanmean(h_residuals)); home_over_rate_last5 = float(np.nanmean(h_over_flags))
            home_avg_ppa_off = float(np.nanmean(h_ppa_off)); home_avg_success_rate = float(np.nanmean(h_success_rate))
            home_avg_ppa_def = float(np.nanmean(h_ppa_def))
            away_avg_pts = float(np.nanmean(ap_for)); away_avg_pts_allowed = float(np.nanmean(ap_against))
            away_avg_total = float(np.nanmean(a_totals)); away_std_total = float(np.nanstd(a_totals))
            away_avg_residual_total = float(np.nanmean(a_residuals)); away_over_rate_last5 = float(np.nanmean(a_over_flags))
            away_avg_ppa_off = float(np.nanmean(a_ppa_off)); away_avg_success_rate = float(np.nanmean(a_success_rate))
            away_avg_ppa_def = float(np.nanmean(a_ppa_def))
            combined_avg_total = float((home_avg_total + away_avg_total) / 2.0)
            diff_avg_residual_total = home_avg_residual_total - away_avg_residual_total
            diff_over_rate_last5 = home_over_rate_last5 - away_over_rate_last5
            off_def_mismatch_home_off_away_def = home_avg_pts - away_avg_pts_allowed
            off_def_mismatch_away_off_home_def = away_avg_pts - home_avg_pts_allowed
            mismatch_home_off_vs_away_def_ppa = home_avg_ppa_off - away_avg_ppa_def
            mismatch_away_off_vs_home_def_ppa = away_avg_ppa_off - home_avg_ppa_def
            mismatch_success_rate = home_avg_success_rate - away_avg_success_rate
            
            home_sp_data = sp_lookup.get((game_season, home_team), {}) if sp_lookup else {}
            away_sp_data = sp_lookup.get((game_season, away_team), {}) if sp_lookup else {}
            
            home_sp_rating = home_sp_data.get("sp_rating", 0.0)
            home_sp_ranking = home_sp_data.get("sp_ranking", 0.0)
            home_sp_offense = home_sp_data.get("sp_offense", 0.0)
            home_sp_offense_ranking = home_sp_data.get("sp_offense_ranking", 0.0)
            home_sp_defense = home_sp_data.get("sp_defense", 0.0)
            home_sp_defense_ranking = home_sp_data.get("sp_defense_ranking", 0.0)
            home_sp_special_teams = home_sp_data.get("sp_special_teams", 0.0)
            
            away_sp_rating = away_sp_data.get("sp_rating", 0.0)
            away_sp_ranking = away_sp_data.get("sp_ranking", 0.0)
            away_sp_offense = away_sp_data.get("sp_offense", 0.0)
            away_sp_offense_ranking = away_sp_data.get("sp_offense_ranking", 0.0)
            away_sp_defense = away_sp_data.get("sp_defense", 0.0)
            away_sp_defense_ranking = away_sp_data.get("sp_defense_ranking", 0.0)
            away_sp_special_teams = away_sp_data.get("sp_special_teams", 0.0)
            
            sp_rating_diff = home_sp_rating - away_sp_rating
            sp_offense_diff = home_sp_offense - away_sp_offense
            sp_defense_diff = home_sp_defense - away_sp_defense
            sp_special_teams_diff = home_sp_special_teams - away_sp_special_teams
            home_off_vs_away_def_sp = home_sp_offense - away_sp_defense
            away_off_vs_home_def_sp = away_sp_offense - home_sp_defense
            
            home_fpi_data = fpi_lookup.get((game_season, home_team), {}) if fpi_lookup else {}
            away_fpi_data = fpi_lookup.get((game_season, away_team), {}) if fpi_lookup else {}
            
            home_fpi = home_fpi_data.get("fpi", 0.0)
            home_fpi_offense = home_fpi_data.get("fpi_offense", 0.0)
            home_fpi_defense = home_fpi_data.get("fpi_defense", 0.0)
            home_fpi_special_teams = home_fpi_data.get("fpi_special_teams", 0.0)
            
            away_fpi = away_fpi_data.get("fpi", 0.0)
            away_fpi_offense = away_fpi_data.get("fpi_offense", 0.0)
            away_fpi_defense = away_fpi_data.get("fpi_defense", 0.0)
            away_fpi_special_teams = away_fpi_data.get("fpi_special_teams", 0.0)
            
            fpi_diff = home_fpi - away_fpi
            fpi_offense_diff = home_fpi_offense - away_fpi_offense
            fpi_defense_diff = home_fpi_defense - away_fpi_defense
            fpi_special_teams_diff = home_fpi_special_teams - away_fpi_special_teams
            home_off_vs_away_def_fpi = home_fpi_offense - away_fpi_defense
            away_off_vs_home_def_fpi = away_fpi_offense - home_fpi_defense
            
            home_combined_offense = (home_sp_offense + home_fpi_offense) / 2.0 if (home_sp_offense != 0.0 and home_fpi_offense != 0.0) else 0.0
            away_combined_offense = (away_sp_offense + away_fpi_offense) / 2.0 if (away_sp_offense != 0.0 and away_fpi_offense != 0.0) else 0.0
            
            home_combined_defense = (home_sp_defense + home_fpi_defense) / 2.0 if (home_sp_defense != 0.0 and home_fpi_defense != 0.0) else 0.0
            away_combined_defense = (away_sp_defense + away_fpi_defense) / 2.0 if (away_sp_defense != 0.0 and away_fpi_defense != 0.0) else 0.0
            combined_defense_diff = home_combined_defense - away_combined_defense
            
            home_sp_off_vs_away_fpi_def = home_sp_offense - away_fpi_defense
            home_fpi_off_vs_away_sp_def = home_fpi_offense - away_sp_defense
            
            home_total_strength_fpi = home_fpi_offense + home_fpi_defense
            away_total_strength_sp = away_sp_offense + away_sp_defense
            
            home_balance_sp = home_sp_offense - home_sp_defense
            away_balance_sp = away_sp_offense - away_sp_defense
            home_balance_fpi = home_fpi_offense - home_fpi_defense
            away_balance_fpi = away_fpi_offense - away_fpi_defense
            balance_sp_diff = home_balance_sp - away_balance_sp
            
            feature_rows.append(
                {
                    "ou_line": float(game["ou_line"]),
                    "spread": float(game.get("spread", 0.0)),
                    "season": int(game.get("season", 0)),
                    "home_avg_pts": home_avg_pts,
                    "home_avg_pts_allowed": home_avg_pts_allowed,
                    "home_avg_total": home_avg_total,
                    "home_std_total": home_std_total,
                    "away_avg_pts": away_avg_pts,
                    "away_avg_pts_allowed": away_avg_pts_allowed,
                    "away_avg_total": away_avg_total,
                    "away_std_total": away_std_total,
                    "combined_avg_total": combined_avg_total,
                    "home_avg_residual_total": home_avg_residual_total,
                    "home_over_rate_last5": home_over_rate_last5,
                    "away_avg_residual_total": away_avg_residual_total,
                    "away_over_rate_last5": away_over_rate_last5,
                    "diff_avg_residual_total": diff_avg_residual_total,
                    "diff_over_rate_last5": diff_over_rate_last5,
                    "off_def_mismatch_home_off_away_def": off_def_mismatch_home_off_away_def,
                    "off_def_mismatch_away_off_home_def": off_def_mismatch_away_off_home_def,
                    "home_avg_ppa_off": home_avg_ppa_off,
                    "home_avg_success_rate": home_avg_success_rate,
                    "home_avg_ppa_def": home_avg_ppa_def,
                    "away_avg_ppa_off": away_avg_ppa_off,
                    "away_avg_success_rate": away_avg_success_rate,
                    "away_avg_ppa_def": away_avg_ppa_def,
                    "mismatch_home_off_vs_away_def_ppa": mismatch_home_off_vs_away_def_ppa,
                    "mismatch_away_off_vs_home_def_ppa": mismatch_away_off_vs_home_def_ppa,
                    "mismatch_success_rate": mismatch_success_rate,
                    
                    "home_sp_rating": home_sp_rating,
                    "home_sp_ranking": home_sp_ranking,
                    "home_sp_offense": home_sp_offense,
                    "home_sp_offense_ranking": home_sp_offense_ranking,
                    "home_sp_defense": home_sp_defense,
                    "home_sp_defense_ranking": home_sp_defense_ranking,
                    "home_sp_special_teams": home_sp_special_teams,
                    "away_sp_rating": away_sp_rating,
                    "away_sp_ranking": away_sp_ranking,
                    "away_sp_offense": away_sp_offense,
                    "away_sp_offense_ranking": away_sp_offense_ranking,
                    "away_sp_defense": away_sp_defense,
                    "away_sp_defense_ranking": away_sp_defense_ranking,
                    "away_sp_special_teams": away_sp_special_teams,
                    "sp_rating_diff": sp_rating_diff,
                    "sp_offense_diff": sp_offense_diff,
                    "sp_defense_diff": sp_defense_diff,
                    "sp_special_teams_diff": sp_special_teams_diff,
                    "home_off_vs_away_def_sp": home_off_vs_away_def_sp,
                    "away_off_vs_home_def_sp": away_off_vs_home_def_sp,
                    
                    "home_fpi": home_fpi,
                    "home_fpi_offense": home_fpi_offense,
                    "home_fpi_defense": home_fpi_defense,
                    "home_fpi_special_teams": home_fpi_special_teams,
                    "away_fpi": away_fpi,
                    "away_fpi_offense": away_fpi_offense,
                    "away_fpi_defense": away_fpi_defense,
                    "away_fpi_special_teams": away_fpi_special_teams,
                    "fpi_diff": fpi_diff,
                    "fpi_offense_diff": fpi_offense_diff,
                    "fpi_defense_diff": fpi_defense_diff,
                    "fpi_special_teams_diff": fpi_special_teams_diff,
                    "home_off_vs_away_def_fpi": home_off_vs_away_def_fpi,
                    "away_off_vs_home_def_fpi": away_off_vs_home_def_fpi,
                    
                    "home_combined_offense": home_combined_offense,
                    "away_combined_defense": away_combined_defense,
                    "combined_defense_diff": combined_defense_diff,
                    "home_sp_off_vs_away_fpi_def": home_sp_off_vs_away_fpi_def,
                    "home_fpi_off_vs_away_sp_def": home_fpi_off_vs_away_sp_def,
                    "home_total_strength_fpi": home_total_strength_fpi,
                    "away_total_strength_sp": away_total_strength_sp,
                    "home_balance_sp": home_balance_sp,
                    "home_balance_fpi": home_balance_fpi,
                    "away_balance_fpi": away_balance_fpi,
                    "balance_sp_diff": balance_sp_diff,
                    
                    "went_over": int(game["went_over"]),
                }
            )

        features = pd.DataFrame(feature_rows)
        features = features.fillna(0.0)

        features["pace_factor"] = features["combined_avg_total"] - features["ou_line"]

        self.features = features

    def _simulate_bets(
        self,
        y_test: pd.Series,
        y_pred_proba: np.ndarray,
        ev_thresh: float,
        bet_amount: float = 10,
        odds_payout: float = 1.909,
    ) -> dict[str, float]:
        profits: list[float] = []
        bets_placed = 0
        wins = 0
        over_bets = 0
        under_bets = 0
        over_wins = 0
        under_wins = 0

        for idx, prob_over in enumerate(y_pred_proba):
            ev_over = (prob_over * (bet_amount * (odds_payout - 1))) - (
                (1 - prob_over) * bet_amount
            )
            ev_under = ((1 - prob_over) * (bet_amount * (odds_payout - 1))) - (
                prob_over * bet_amount
            )

            profit = 0.0
            actual = y_test.iloc[idx]

            if ev_over > (bet_amount * ev_thresh):
                profit = bet_amount * (odds_payout - 1) if actual == 1 else -bet_amount
                over_bets += 1
                if profit > 0:
                    over_wins += 1
                    wins += 1
                bets_placed += 1
            elif ev_under > (bet_amount * ev_thresh):
                profit = bet_amount * (odds_payout - 1) if actual == 0 else -bet_amount
                under_bets += 1
                if profit > 0:
                    under_wins += 1
                    wins += 1
                bets_placed += 1

            if profit != 0:
                profits.append(profit)

        total_profit = float(sum(profits))
        roi = (
            (total_profit / (bets_placed * bet_amount)) * 100 if bets_placed else 0.0
        )
        win_rate = (sum(1 for p in profits if p > 0) / bets_placed * 100) if bets_placed else 0.0
        over_win_rate = (over_wins / over_bets * 100) if over_bets > 0 else 0.0
        under_win_rate = (under_wins / under_bets * 100) if under_bets > 0 else 0.0

        win_rate_ci = self._win_rate_ci(bets_placed, wins)
        roi_ci = self._roi_ci(win_rate_ci, odds_payout)

        return {
            "bets": float(bets_placed),
            "profit": total_profit,
            "roi": roi,
            "win_rate": win_rate,
            "win_rate_ci": win_rate_ci,
            "roi_ci": roi_ci,
            "over_bets": float(over_bets),
            "over_win_rate": over_win_rate,
            "under_bets": float(under_bets),
            "under_win_rate": under_win_rate,
        }

    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Tune hyperparameters using walk-forward validation on training data."""
        print("\n--- Hyperparameter Tuning ---")
        
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.15],
            'min_child_weight': [2, 3, 5],
        }
        
        if hasattr(self, 'features') and 'season' in self.features.columns:
            season_col = self.features['season'].iloc[:len(X_train)]
            unique_seasons = sorted(season_col.unique())
            tune_seasons = unique_seasons[-3:] if len(unique_seasons) >= 3 else unique_seasons
            tune_mask = season_col.isin(tune_seasons).values
            X_tune = X_train.iloc[tune_mask] if hasattr(X_train, 'iloc') else X_train[tune_mask]
            y_tune = y_train.iloc[tune_mask] if hasattr(y_train, 'iloc') else y_train[tune_mask]
        else:
            X_tune = X_train
            y_tune = y_train
        
        best_params = None
        best_score = -float('inf')
        results = []
        
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for min_child_weight in param_grid['min_child_weight']:
                    if len(X_tune) > 1000:
                        val_size = int(len(X_tune) * 0.2)
                        X_tr = X_tune.iloc[:-val_size].values
                        y_tr = y_tune.iloc[:-val_size].values
                        X_val = X_tune.iloc[-val_size:].values
                        y_val = y_tune.iloc[-val_size:].values
                    else:
                        X_tr = X_tune.values
                        y_tr = y_tune.values
                        X_val = X_tr
                        y_val = y_tr
                    
                    model = xgb.XGBClassifier(
                        n_estimators=50,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=min_child_weight,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        enable_categorical=False,
                    )
                    
                    model.fit(X_tr, y_tr)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                    
                    results.append({
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'min_child_weight': min_child_weight,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'min_child_weight': min_child_weight,
                        }
        
        print(f"  Best params: {best_params} (AUC: {best_score:.4f})")
        return best_params
    
    def train(self, tune_hyperparameters: bool = True) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        if self.features is None:
            raise ValueError("Features must be created before training.")

        X = self.features.drop(columns=["went_over", "season"])
        y = self.features["went_over"]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values

        if tune_hyperparameters:
            best_params = self._tune_hyperparameters(X_train, y_train)
            max_depth = best_params['max_depth']
            learning_rate = best_params['learning_rate']
            min_child_weight = best_params['min_child_weight']
        else:
            max_depth = 4
            learning_rate = 0.1
            min_child_weight = 3

        base_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=min_child_weight,
            reg_alpha=0.1,  # Added: L1 regularization
            reg_lambda=1.0,  # Added: L2 regularization
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
            enable_categorical=False,
        )

        base_model.fit(X_train_np, y_train_np)
        self.model = base_model

        y_pred_proba_raw = base_model.predict_proba(X_test_np)[:, 1]

        print("\n--- Uncalibrated Model Performance ---")
        print(f"  Accuracy: {accuracy_score(y_test_np, base_model.predict(X_test_np)):.4f}")
        print(f"  AUC: {roc_auc_score(y_test_np, y_pred_proba_raw):.4f}")
        print(f"  Brier Score: {brier_score_loss(y_test_np, y_pred_proba_raw):.4f}")

        cal_base_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=min_child_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
            enable_categorical=False,
        )

        if not hasattr(cal_base_model, "_estimator_type"):
            cal_base_model._estimator_type = "classifier"
        
        tscv = TimeSeriesSplit(n_splits=3)

        calibrated_model = CalibratedClassifierCV(
            cal_base_model,
            method="isotonic",
            cv=tscv,
        )
        calibrated_model.fit(X_train_np, y_train_np)
        self.calibrated_model = calibrated_model

        y_pred_proba = calibrated_model.predict_proba(X_test_np)[:, 1]

        print("\n--- Calibrated Model Performance ---")
        print(f"  Accuracy: {accuracy_score(y_test_np, calibrated_model.predict(X_test_np)):.4f}")
        print(f"  AUC: {roc_auc_score(y_test_np, y_pred_proba):.4f}")
        print(f"  Brier Score: {brier_score_loss(y_test_np, y_pred_proba):.4f}")

        return X_test, y_test, y_pred_proba

    def _win_rate_ci(self, bets: float, wins: float, alpha: float = 0.05) -> tuple[float, float]:
        """Normal-approx 95% CI for win rate; returns percents."""
        if bets == 0:
            return (0.0, 0.0)
        p = wins / bets
        se = np.sqrt(p * (1 - p) / bets)
        z = 1.96
        lower = max(0.0, p - z * se) * 100
        upper = min(1.0, p + z * se) * 100
        return (lower, upper)

    def _roi_ci(self, win_rate_ci: tuple[float, float], odds_payout: float) -> tuple[float, float]:
        """Translate win rate CI to ROI CI (percent) using the odds payout."""
        lower_wr = win_rate_ci[0] / 100
        upper_wr = win_rate_ci[1] / 100
        def roi_from_p(p: float) -> float:
            return (p * (odds_payout - 1) - (1 - p)) * 100

        return (roi_from_p(lower_wr), roi_from_p(upper_wr))

    def walk_forward_by_season(self) -> None:
        if self.features is None:
            raise ValueError("Features must be created before training.")

        if "season" not in self.features.columns:
            print("No season column available for walk-forward evaluation.")
            return

        seasons = sorted(self.features["season"].unique())
        if len(seasons) < 2:
            print("Not enough seasons for walk-forward evaluation.")
            return

        feature_cols = [c for c in self.features.columns if c not in {"went_over", "season"}]
        ev_thresholds = [0.01, 0.03, 0.05, 0.07, 0.10]
        bet_amount = 10
        odds_payout = 1.909

        agg_results: dict[float, dict[str, float]] = {
            ev: {"profit": 0.0, "bets": 0.0} for ev in ev_thresholds
        }

        print("\n=== Season Walk-Forward Evaluation ===")
        for test_season in seasons[1:]:
            train_df = self.features[self.features["season"] < test_season]
            test_df = self.features[self.features["season"] == test_season]

            if len(train_df) < 25 or len(test_df) == 0:
                print(f"Season {test_season}: skipped (train={len(train_df)}, test={len(test_df)}).")
                continue

            X_train = train_df[feature_cols]
            y_train = train_df["went_over"]
            X_test = test_df[feature_cols]
            y_test = test_df["went_over"]

            base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.15,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                enable_categorical=False,
            )

            base_model.fit(X_train.values, y_train.values)

            cal_model = CalibratedClassifierCV(
                base_model,
                method="isotonic",
                cv=TimeSeriesSplit(n_splits=3),
            )
            cal_model.fit(X_train.values, y_train.values)

            y_pred_proba = cal_model.predict_proba(X_test.values)[:, 1]

            print(
                f"\nSeason {test_season}: "
                f"test_n={len(test_df)}, "
                f"base_over_rate={y_test.mean():.3f}, "
                f"Accuracy={accuracy_score(y_test, cal_model.predict(X_test)):.3f}, "
                f"AUC={roc_auc_score(y_test, y_pred_proba):.3f}, "
                f"Brier={brier_score_loss(y_test, y_pred_proba):.4f}"
            )

            for ev_thresh in ev_thresholds:
                stats = self._simulate_bets(
                    y_test=y_test,
                    y_pred_proba=y_pred_proba,
                    ev_thresh=ev_thresh,
                    bet_amount=bet_amount,
                    odds_payout=odds_payout,
                )
                agg_results[ev_thresh]["profit"] += stats["profit"]
                agg_results[ev_thresh]["bets"] += stats["bets"]

                print(
                    f"  EV>{ev_thresh*100:.0f}% | bets={int(stats['bets'])} | "
                    f"ROI={stats['roi']:.2f}% | win_rate={stats['win_rate']:.2f}% | "
                    f"win_CI=[{stats['win_rate_ci'][0]:.1f}, {stats['win_rate_ci'][1]:.1f}] | "
                    f"roi_CI=[{stats['roi_ci'][0]:.1f}, {stats['roi_ci'][1]:.1f}] | "
                    f"over_win_rate={stats['over_win_rate']:.2f}% ({int(stats['over_bets'])} bets) | "
                    f"under_win_rate={stats['under_win_rate']:.2f}% ({int(stats['under_bets'])} bets)"
                )

        print("\n--- Walk-Forward Summary ---")
        for ev_thresh in ev_thresholds:
            bets = agg_results[ev_thresh]["bets"]
            profit = agg_results[ev_thresh]["profit"]
            if bets == 0:
                print(f"  EV>{ev_thresh*100:.0f}% | No bets across seasons.")
                continue
            roi = (profit / (bets * bet_amount)) * 100
            print(
                f"  EV>{ev_thresh*100:.0f}% | Total bets={int(bets)} | ROI={roi:.2f}% | P/L=${profit:.2f}"
            )

    def evaluate_betting(
        self, X_test: pd.DataFrame, y_test: pd.Series, y_pred_proba: np.ndarray
    ) -> None:
        bet_amount = 10
        odds_payout = 1.909

        for ev_thresh in [0.01, 0.03, 0.05, 0.07, 0.10]:
            stats = self._simulate_bets(
                y_test=y_test,
                y_pred_proba=y_pred_proba,
                ev_thresh=ev_thresh,
                bet_amount=bet_amount,
                odds_payout=odds_payout,
            )

            print(f"\n--- Betting Results (EV > {ev_thresh * 100:.0f}%) ---")
            if stats["bets"] == 0:
                print("  No bets met the threshold.")
                continue

            print(
                f"  Bets: {int(stats['bets'])} | Win Rate: {stats['win_rate']:.2f}% "
                f"(CI [{stats['win_rate_ci'][0]:.1f}, {stats['win_rate_ci'][1]:.1f}]) | "
                f"ROI: {stats['roi']:.2f}% (CI [{stats['roi_ci'][0]:.1f}, {stats['roi_ci'][1]:.1f}]) | "
                f"P/L: ${stats['profit']:.2f}"
            )
            
            print(
                f"    Overs:  {int(stats['over_bets'])} bets ({stats['over_win_rate']:.2f}% win rate)"
            )
            print(
                f"    Unders: {int(stats['under_bets'])} bets ({stats['under_win_rate']:.2f}% win rate)"
            )

    def plot(
        self, X_test: pd.DataFrame, y_test: pd.Series, y_pred_proba: np.ndarray
    ) -> None:
        if self.model is None:
            raise ValueError("Model must be trained before plotting.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        sns.histplot(y_pred_proba, bins=25, ax=axes[0, 0], kde=True).set_title(
            "Probability Distribution"
        )

        feat_imp = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        sns.barplot(
            x="importance",
            y="feature",
            data=feat_imp.head(10),
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Top 10 Features")

        frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        axes[1, 0].plot(mean_pred, frac_pos, marker="o", label="Model")
        axes[1, 0].plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfect"
        )
        axes[1, 0].set_xlabel("Mean Predicted Probability")
        axes[1, 0].set_ylabel("Fraction of Positives")
        axes[1, 0].set_title("Calibration Plot")
        axes[1, 0].legend()

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Review the console output for performance metrics\nand betting simulations.",
            ha="center",
            va="center",
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()


def main() -> None:
    dataset_path = Path("data/processed/training_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python scripts/build_dataset.py` first."
        )

    model = CFBBettingModel()
    model.load_data(dataset_path)
    model.create_features()
    X_test, y_test, y_pred_proba = model.train()
    model.walk_forward_by_season()
    model.evaluate_betting(X_test, y_test, y_pred_proba)
    model.plot(X_test, y_test, y_pred_proba)


if __name__ == "__main__":
    main()
