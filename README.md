# ML-CFB

## Machine Learning Model for College Football Over/Under Betting

This repo contains a complete pipeline for predicting College Football Over/Under betting outcomes using machine learning. The system ingests data from CollegeFootballData (CFBD), engineers comprehensive features, and trains an XGBoost model with walk-forward validation to simulate realistic betting strategies.

### Usage Steps

1. **Create and activate a virtual environment, then install dependencies:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. **Set CFBD API key:**
Create `.env` file in root:
```bash
echo "CFBD_API_KEY=YOUR_KEY_HERE" >> .env
```
Optionally also set `CFBD_START_SEASON` and `CFBD_END_SEASON` in `.env` (defaults: 2022–2025)

3. **Build the dataset:**
```bash
python scripts/build_dataset.py --start 2022 --end 2025
```

This creates:
- **Raw data**: `data/raw/games_YYYY.csv`, `data/raw/lines_YYYY.csv`, `data/raw/advanced_stats_YYYY.csv`, `data/raw/team_sp_YYYY.csv`, `data/raw/team_fpi_YYYY.csv`, etc.
- **Processed datasets**: 
  - `data/processed/totals_dataset.csv` (backward compatibility)
  - `data/processed/training_dataset.csv` (for model training)

4. **Train the model:**
```bash
python train_model.py
```

The training script will:
- Load the training dataset and impute missing values
- Create comprehensive features from historical game data
- Perform hyperparameter tuning using walk-forward validation
- Train an XGBoost classifier with isotonic calibration
- Evaluate model performance using season-by-season walk-forward validation
- Simulate betting strategies at various expected value (EV) thresholds
- Generate visualizations (probability distributions, feature importance, calibration curves)

### Dataset Details

**Training dataset columns:**
- Identifiers: `game_id`, `season`, `week`, `date`, `home_team`, `away_team`
- Outcomes: `home_points`, `away_points`, `total_points`
- Market: `ou_line` (closing total), `spread`
- Advanced stats: `home_ppa_offense`, `home_success_rate`, `home_ppa_defense`, etc.
- Team ratings: SP+ ratings, FPI ratings, returning production
- Target: `went_over` (1=Over, 0=Under; pushes excluded)

The `ou_line` is selected from the preferred provider for each game (William Hill, DraftKings, ESPN Bet, etc.). Only regular season games with betting lines are included.

### Model Features

The model uses a comprehensive feature set combining rolling statistics and preseason ratings:

**Rolling Statistics (Last 5 Games):**
- **Scoring stats**: Average points for/against, average total points, standard deviation
- **Market residuals**: Average difference between actual totals and closing lines
- **Over/Under trends**: Recent over rate for each team
- **Advanced metrics**: Points per attempt (PPA) offense/defense, success rate
- **Matchup features**: Offense vs defense mismatches, combined averages

**Preseason/Historical Ratings:**
- **SP+ Ratings**: Overall rating, offense, defense, special teams, rankings
- **FPI Ratings**: Overall FPI, offense, defense, special teams components
- **Combined metrics**: Blended offense/defense ratings, balance indicators

**Market Context:**
- Closing line, spread, pace factor (combined average total vs closing line)

### Model Architecture

- **Algorithm**: XGBoost Classifier
- **Hyperparameters**: Tuned via walk-forward validation on recent seasons
  - Best params: `max_depth=5`, `learning_rate=0.05`, `min_child_weight=5`
- **Calibration**: Isotonic calibration using TimeSeriesSplit (3 folds) for improved probability estimates
- **Regularization**: L1 (alpha=0.1) and L2 (lambda=1.0) regularization

### Model Evaluation

The model uses **walk-forward validation by season** to simulate realistic betting scenarios:

1. For each test season (2023, 2024, 2025), the model trains on all previous seasons
2. Predictions are made on the test season
3. Betting simulations are run at multiple EV thresholds (1%, 3%, 5%, 7%, 10%)
4. Results are aggregated across all seasons

**Performance Metrics:**
- **Accuracy**: Classification accuracy on test set
- **AUC**: Area under the ROC curve
- **Brier Score**: Calibration quality (lower is better)
- **Betting simulations**: ROI, win rate, confidence intervals, P/L

### Results

#### Season-by-Season Performance

**Season 2023** (1,322 test games, base over rate: 49.9%)
- Accuracy: 52.6% | AUC: 0.532 | Brier: 0.2487
- Best strategy: EV>10% → 57 bets, 23.92% ROI, 64.91% win rate

**Season 2024** (1,484 test games, base over rate: 49.4%)
- Accuracy: 49.7% | AUC: 0.500 | Brier: 0.2517
- Best strategy: EV>5% → 127 bets, 6.72% ROI, 55.91% win rate

**Season 2025** (1,334 test games, base over rate: 49.1%)
- Accuracy: 51.4% | AUC: 0.519 | Brier: 0.2510
- Best strategy: EV>10% → 66 bets, 7.02% ROI, 56.06% win rate

#### Walk-Forward Summary (All Seasons Combined)

| EV Threshold | Total Bets | ROI | P/L |
|--------------|------------|-----|-----|
| EV > 1% | 1,635 | -2.86% | -$467.12 |
| EV > 3% | 970 | 0.17% | $16.81 |
| EV > 5% | 370 | 5.77% | $213.45 |
| EV > 7% | 237 | 9.55% | $226.24 |
| EV > 10% | 172 | 10.99% | $189.00 |

#### Under-Only Strategy (All Seasons Combined)

Analysis revealed that the model has a significantly stronger edge on Under bets. When restricting to Under bets only:

| EV Threshold | Total Bets | ROI | P/L |
|--------------|------------|-----|-----|
| EV > 1% | 740 | 0.09% | $6.92 |
| EV > 3% | 487 | 2.70% | $131.58 |
| EV > 5% | 225 | 7.75% | $174.43 |
| EV > 7% | 146 | 11.14% | $162.65 |
| EV > 10% | 103 | 11.20% | $115.40 |

**Key Findings:**
- Under bets significantly outperform Over bets: the Under-only strategy improves ROI at every threshold
- At EV>1%, the Under-only strategy turns a -2.86% loss into breakeven (+0.09%)
- The EV>7% Under-only threshold provides 11.14% ROI with 146 bets (best risk-adjusted return)
- Feature importance reveals defensive metrics (SP+ defense, balance) are the most predictive features
- This suggests the model identifies defensive mismatches that the betting market undervalues
- The 2023 season showed exceptional Under performance: 67.57% win rate at EV>10%

**Note**: Results assume -110 odds (1.909 payout) and $10 bet size. Results may vary based on when data was fetched from the CFBD API, as the API is continuously updated.