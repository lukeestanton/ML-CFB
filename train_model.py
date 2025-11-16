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
            "temperature", "wind_speed", "precipitation"
        ]
        
        if "is_dome" in self.data.columns:
            self.data["is_dome"] = self.data["is_dome"].fillna(0).astype(int)
        
        cols_found = [col for col in cols_to_fill if col in self.data.columns]
        if cols_found:
            print(f"--- Imputing NaNs for: {cols_found} ---")
            self.data[cols_found] = self.data[cols_found].fillna(0.0)
        else:
            print("--- No advanced stats or weather columns found. ---")
            
        df = self.data
        feature_rows: list[dict[str, float]] = []

        for _, game in df.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            game_date = game["date"]

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
                    if "home_ppa_offense" in hg:
                        h_ppa_off.append(hg["home_ppa_offense"])
                        h_success_rate.append(hg["home_success_rate"])
                        h_ppa_def.append(hg["home_ppa_defense"])
                else:
                    pf, pa = hg["away_points"], hg["home_points"]
                    if "away_ppa_offense" in hg:
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
                    if "home_ppa_offense" in ag:
                        a_ppa_off.append(ag["home_ppa_offense"])
                        a_success_rate.append(ag["home_success_rate"])
                        a_ppa_def.append(ag["home_ppa_defense"])
                else:
                    pf, pa = ag["away_points"], ag["home_points"]
                    if "away_ppa_offense" in ag:
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
            
            feature_rows.append(
                {
                    "ou_line": float(game["ou_line"]),
                    "spread": float(game.get("spread", 0.0)),
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
                    
                    "is_dome": float(game.get("is_dome", 0.0)),
                    "temp": float(game.get("temperature", 0.0)),
                    "wind": float(game.get("wind_speed", 0.0)),
                    "precip": float(game.get("precipitation", 0.0)),

                    "went_over": int(game["went_over"]),
                }
            )

        features = pd.DataFrame(feature_rows)
        features = features.fillna(0.0)

        features["pace_factor"] = features["combined_avg_total"] - features["ou_line"]

        self.features = features

    def train(self) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        if self.features is None:
            raise ValueError("Features must be created before training.")

        X = self.features.drop(columns=["went_over"])
        y = self.features["went_over"]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values

        base_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
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
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
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

    def evaluate_betting(
        self, X_test: pd.DataFrame, y_test: pd.Series, y_pred_proba: np.ndarray
    ) -> None:
        bet_amount = 10
        odds_payout = 1.909

        for ev_thresh in [0.01, 0.03, 0.05, 0.07, 0.10]:
            profits: list[float] = []
            bets_placed: list[str] = []
            
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
                bet_made: str | None = None
                actual = y_test.iloc[idx]

                if ev_over > (bet_amount * ev_thresh):
                    profit = (
                        bet_amount * (odds_payout - 1) if actual == 1 else -bet_amount
                    )
                    bet_made = "Over"
                    over_bets += 1
                    if profit > 0:
                        over_wins += 1
                elif ev_under > (bet_amount * ev_thresh):
                    profit = (
                        bet_amount * (odds_payout - 1) if actual == 0 else -bet_amount
                    )
                    bet_made = "Under"
                    under_bets += 1
                    if profit > 0:
                        under_wins += 1

                if bet_made:
                    profits.append(profit)
                    bets_placed.append(bet_made)

            print(f"\n--- Betting Results (EV > {ev_thresh * 100:.0f}%) ---")
            if not bets_placed:
                print("  No bets met the threshold.")
                continue

            total_profit = float(sum(profits))
            roi = (total_profit / (len(bets_placed) * bet_amount)) * 100
            win_rate = (sum(1 for p in profits if p > 0) / len(profits)) * 100

            print(
                f"  Bets: {len(bets_placed)} | Win Rate: {win_rate:.2f}% | "
                f"ROI: {roi:.2f}% | P/L: ${total_profit:.2f}"
            )
            
            over_win_rate = (over_wins / over_bets * 100) if over_bets > 0 else 0
            under_win_rate = (under_wins / under_bets * 100) if under_bets > 0 else 0
            print(
                f"    Overs:  {over_bets} bets ({over_win_rate:.2f}% win rate)"
            )
            print(
                f"    Unders: {under_bets} bets ({under_win_rate:.2f}% win rate)"
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
    model.evaluate_betting(X_test, y_test, y_pred_proba)
    model.plot(X_test, y_test, y_pred_proba)


if __name__ == "__main__":
    main()
