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

        # Drop pushes
        push_mask = df["total_points"] == df["ou_line"]
        df = df.loc[~push_mask].copy()

        # Binary target: 1 = game went over the total, 0 = under
        df["went_over"] = (df["total_points"] > df["ou_line"]).astype(int)

        # Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)
        self.data = df

    def create_features(self) -> None:
        if self.data is None:
            raise ValueError("Data must be loaded before creating features.")

        feature_rows: list[dict[str, float]] = []

        for _, game in self.data.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            game_date = game["date"]

            # Last 5 games of each team before this game
            home_history = self.data[
                (
                    (self.data["home_team"] == home_team)
                    | (self.data["away_team"] == home_team)
                )
                & (self.data["date"] < game_date)
            ].tail(5)

            away_history = self.data[
                (
                    (self.data["home_team"] == away_team)
                    | (self.data["away_team"] == away_team)
                )
                & (self.data["date"] < game_date)
            ].tail(5)

            if len(home_history) < 2 or len(away_history) < 2:
                continue

            # Home team rolling stats
            hp_for, hp_against, h_totals = [], [], []
            h_residuals, h_over_flags = [], []
            for _, hg in home_history.iterrows():
                if hg["home_team"] == home_team:
                    pf, pa = hg["home_points"], hg["away_points"]
                else:
                    pf, pa = hg["away_points"], hg["home_points"]

                total = hg["total_points"]

                hp_for.append(pf)
                hp_against.append(pa)
                h_totals.append(total)

                # Market residual
                if pd.notna(hg["ou_line"]):
                    residual = float(total - hg["ou_line"])
                    h_residuals.append(residual)
                    h_over_flags.append(1.0 if total > hg["ou_line"] else 0.0)

            # Away team rolling stats
            ap_for, ap_against, a_totals = [], [], []
            a_residuals, a_over_flags = [], []
            for _, ag in away_history.iterrows():
                if ag["home_team"] == away_team:
                    pf, pa = ag["home_points"], ag["away_points"]
                else:
                    pf, pa = ag["away_points"], ag["home_points"]

                total = ag["total_points"]

                ap_for.append(pf)
                ap_against.append(pa)
                a_totals.append(total)

                if pd.notna(ag["ou_line"]):
                    residual = float(total - ag["ou_line"])
                    a_residuals.append(residual)
                    a_over_flags.append(1.0 if total > ag["ou_line"] else 0.0)

            # If don't have any residual history skip game
            if len(h_residuals) == 0 or len(a_residuals) == 0:
                continue

            # Aggregate stats for home
            home_avg_pts = float(np.mean(hp_for))
            home_avg_pts_allowed = float(np.mean(hp_against))
            home_avg_total = float(np.mean(h_totals))
            home_std_total = float(np.std(h_totals))
            home_avg_residual_total = float(np.mean(h_residuals))
            home_over_rate_last5 = float(np.mean(h_over_flags))

            # Aggregate stats for away
            away_avg_pts = float(np.mean(ap_for))
            away_avg_pts_allowed = float(np.mean(ap_against))
            away_avg_total = float(np.mean(a_totals))
            away_std_total = float(np.std(a_totals))
            away_avg_residual_total = float(np.mean(a_residuals))
            away_over_rate_last5 = float(np.mean(a_over_flags))

            combined_avg_total = float((home_avg_total + away_avg_total) / 2.0)

            # Matchup-style features
            diff_avg_residual_total = home_avg_residual_total - away_avg_residual_total
            diff_over_rate_last5 = home_over_rate_last5 - away_over_rate_last5
            off_def_mismatch_home_off_away_def = home_avg_pts - away_avg_pts_allowed
            off_def_mismatch_away_off_home_def = away_avg_pts - home_avg_pts_allowed

            feature_rows.append(
                {
                    "ou_line": float(game["ou_line"]),
                    "spread": float(game.get("spread", 0.0)),

                    # Basic rolling scoring stats
                    "home_avg_pts": home_avg_pts,
                    "home_avg_pts_allowed": home_avg_pts_allowed,
                    "home_avg_total": home_avg_total,
                    "home_std_total": home_std_total,
                    "away_avg_pts": away_avg_pts,
                    "away_avg_pts_allowed": away_avg_pts_allowed,
                    "away_avg_total": away_avg_total,
                    "away_std_total": away_std_total,
                    "combined_avg_total": combined_avg_total,

                    # Market residual features
                    "home_avg_residual_total": home_avg_residual_total,
                    "home_over_rate_last5": home_over_rate_last5,
                    "away_avg_residual_total": away_avg_residual_total,
                    "away_over_rate_last5": away_over_rate_last5,
                    "diff_avg_residual_total": diff_avg_residual_total,
                    "diff_over_rate_last5": diff_over_rate_last5,

                    # Offense vs defense mismatch features
                    "off_def_mismatch_home_off_away_def": off_def_mismatch_home_off_away_def,
                    "off_def_mismatch_away_off_home_def": off_def_mismatch_away_off_home_def,

                    # Target
                    "went_over": int(game["went_over"]),
                }
            )

        features = pd.DataFrame(feature_rows)

        # "Pace"
        features["pace_factor"] = features["combined_avg_total"] - features["ou_line"]

        self.features = features

    def train(self) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        if self.features is None:
            raise ValueError("Features must be created before training.")

        X = self.features.drop(columns=["went_over"])
        y = self.features["went_over"]

        # Time-based split
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

        # Calibrated model (isotonic)
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

        calibrated_model = CalibratedClassifierCV(
            cal_base_model,
            method="isotonic",
            cv=3,
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
        odds_payout = 1.909  # -110 odds equivalent

        for ev_thresh in [0.01, 0.03, 0.05, 0.07, 0.10]:
            profits: list[float] = []
            bets_placed: list[str] = []

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
                elif ev_under > (bet_amount * ev_thresh):
                    profit = (
                        bet_amount * (odds_payout - 1) if actual == 0 else -bet_amount
                    )
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

    def plot(
        self, X_test: pd.DataFrame, y_test: pd.Series, y_pred_proba: np.ndarray
    ) -> None:
        if self.model is None:
            raise ValueError("Model must be trained before plotting.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Distribution of predicted probabilities
        sns.histplot(y_pred_proba, bins=25, ax=axes[0, 0], kde=True).set_title(
            "Probability Distribution"
        )

        # Feature importances from the base (uncalibrated) XGBoost model
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

        # Calibration plot
        frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        axes[1, 0].plot(mean_pred, frac_pos, marker="o", label="Model")
        axes[1, 0].plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfect"
        )
        axes[1, 0].set_xlabel("Mean Predicted Probability")
        axes[1, 0].set_ylabel("Fraction of Positives")
        axes[1, 0].set_title("Calibration Plot")
        axes[1, 0].legend()

        # Text panel
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
