"""Train and evaluate the CFBBettingModel on the processed dataset.

This script reproduces the notebook pipeline described in the README. It loads the
processed training dataset, engineers rolling statistics for each team, fits an
XGBoost classifier with isotonic calibration, evaluates betting performance, and
produces diagnostic plots.

Usage:
    python train_model.py

The script expects that `python scripts/build_dataset.py` has already been run so
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

warnings.filterwarnings("ignore", category=UserWarning)


class CFBBettingModel:
    """XGBoost-based classifier that predicts whether a game goes over the total."""

    def __init__(self) -> None:
        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.model: xgb.XGBClassifier | None = None
        self.calibrated_model: CalibratedClassifierCV | None = None

    def load_data(self, file_path: str | Path) -> None:
        """Load the processed CSV and compute helper columns."""
        df = pd.read_csv(file_path, parse_dates=["date"])
        df["total_points"] = df["home_points"] + df["away_points"]
        df["went_over"] = (df["total_points"] > df["ou_line"]).astype(int)
        df = df.sort_values("date").reset_index(drop=True)
        self.data = df

    def create_features(self) -> None:
        """Engineer rolling team statistics for the last five games."""
        if self.data is None:
            raise ValueError("Data must be loaded before creating features.")

        feature_rows: list[dict[str, float]] = []
        for _, game in self.data.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            game_date = game["date"]

            home_history = self.data[
                ((self.data["home_team"] == home_team) | (self.data["away_team"] == home_team))
                & (self.data["date"] < game_date)
            ].tail(5)
            away_history = self.data[
                ((self.data["home_team"] == away_team) | (self.data["away_team"] == away_team))
                & (self.data["date"] < game_date)
            ].tail(5)

            if len(home_history) < 2 or len(away_history) < 2:
                continue

            hp_for, hp_against, h_totals = [], [], []
            for _, hg in home_history.iterrows():
                if hg["home_team"] == home_team:
                    pf, pa = hg["home_points"], hg["away_points"]
                else:
                    pf, pa = hg["away_points"], hg["home_points"]
                hp_for.append(pf)
                hp_against.append(pa)
                h_totals.append(hg["total_points"])

            ap_for, ap_against, a_totals = [], [], []
            for _, ag in away_history.iterrows():
                if ag["home_team"] == away_team:
                    pf, pa = ag["home_points"], ag["away_points"]
                else:
                    pf, pa = ag["away_points"], ag["home_points"]
                ap_for.append(pf)
                ap_against.append(pa)
                a_totals.append(ag["total_points"])

            feature_rows.append(
                {
                    "ou_line": float(game["ou_line"]),
                    "spread": float(game.get("spread", 0.0)),
                    "home_avg_pts": float(np.mean(hp_for)),
                    "home_avg_pts_allowed": float(np.mean(hp_against)),
                    "home_avg_total": float(np.mean(h_totals)),
                    "home_std_total": float(np.std(h_totals)),
                    "away_avg_pts": float(np.mean(ap_for)),
                    "away_avg_pts_allowed": float(np.mean(ap_against)),
                    "away_avg_total": float(np.mean(a_totals)),
                    "away_std_total": float(np.std(a_totals)),
                    "combined_avg_total": float((np.mean(h_totals) + np.mean(a_totals)) / 2),
                    "went_over": int(game["went_over"]),
                }
            )

        features = pd.DataFrame(feature_rows)
        features["pace_factor"] = features["combined_avg_total"] - features["ou_line"]
        self.features = features

    def train(self) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Fit the XGBoost model and return test set predictions."""
        if self.features is None:
            raise ValueError("Features must be created before training.")

        X = self.features.drop(columns=["went_over"])
        y = self.features["went_over"]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        self.model = model

        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated_model.fit(X_train, y_train)
        self.calibrated_model = calibrated_model

        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

        print("\n--- Model Performance ---")
        print(f"  Accuracy: {accuracy_score(y_test, calibrated_model.predict(X_test)):.4f}")
        print(f"  AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"  Brier Score: {brier_score_loss(y_test, y_pred_proba):.4f}")

        return X_test, y_test, y_pred_proba

    def evaluate_betting(
        self, X_test: pd.DataFrame, y_test: pd.Series, y_pred_proba: np.ndarray
    ) -> None:
        """Simulate betting strategies across a range of EV thresholds."""
        bet_amount = 10
        odds_payout = 1.909  # Standard -110 American odds payout multiplier.

        for ev_thresh in [0.01, 0.03, 0.05, 0.07, 0.10]:
            profits: list[float] = []
            bets_placed: list[str] = []

            for idx, prob_over in enumerate(y_pred_proba):
                ev_over = (prob_over * (bet_amount * (odds_payout - 1))) - ((1 - prob_over) * bet_amount)
                ev_under = ((1 - prob_over) * (bet_amount * (odds_payout - 1))) - (prob_over * bet_amount)

                profit = 0.0
                bet_made: str | None = None
                actual = y_test.iloc[idx]

                if ev_over > (bet_amount * ev_thresh):
                    profit = bet_amount * (odds_payout - 1) if actual == 1 else -bet_amount
                    bet_made = "Over"
                elif ev_under > (bet_amount * ev_thresh):
                    profit = bet_amount * (odds_payout - 1) if actual == 0 else -bet_amount
                    bet_made = "Under"

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

    def plot(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred_proba: np.ndarray) -> None:
        """Generate diagnostic plots for the model."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        sns.histplot(y_pred_proba, bins=25, ax=axes[0, 0], kde=True).set_title("Probability Distribution")

        feat_imp = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        sns.barplot(x="importance", y="feature", data=feat_imp.head(10), ax=axes[0, 1])
        axes[0, 1].set_title("Top 10 Features")

        frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        axes[1, 0].plot(mean_pred, frac_pos, marker="o", label="Model")
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
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
