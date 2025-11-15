# ML-CFB

## Data ingestion, parsing, and model training for CFB totals

This repo includes a complete pipeline to ingest CollegeFootballData (CFBD) games and sportsbook totals lines, parse datasets for modeling, and train machine learning models to predict Over/Under outcomes.

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
Optionally also set `CFBD_START_SEASON` and `CFBD_END_SEASON` in `.env` (defaults: 2018–2025)

3. **Build the dataset:**
```bash
python scripts/build_dataset.py --start 2018 --end 2025
```

This creates:
- **Raw data**: `data/raw/games_YYYY.csv`, `data/raw/lines_YYYY.csv`
- **Processed datasets**: 
  - `data/processed/totals_dataset.csv` (backward compatibility)
  - `data/processed/training_dataset.csv` (for model training)

4. **Train the model:**
```bash
python train_model.py
```

The training script will:
- Load the training dataset
- Create features from historical game data
- Train an XGBoost classifier with calibration
- Evaluate model performance (accuracy, AUC, Brier score)
- Simulate betting strategies at various EV thresholds
- Generate visualizations (probability distributions, feature importance, calibration curves)

### Dataset Details

**Training dataset columns:**
- Identifiers: `game_id`, `season`, `week`, `date`, `home_team`, `away_team`
- Outcomes: `home_points`, `away_points`, `total_points`
- Market: `ou_line` (closing total), `spread`
- Target: `went_over` (1=Over, 0=Under; pushes excluded)

The `ou_line` is selected from the preferred provider for each game (William Hill, DraftKings, ESPN Bet, etc.). Only regular season games with betting lines are included.

### Model Features

The model uses rolling statistics from each team's last 5 games:
- **Scoring stats**: Average points for/against, average total points, standard deviation
- **Market residuals**: Average difference between actual totals and closing lines
- **Over/Under trends**: Recent over rate for each team
- **Matchup features**: Offense vs defense mismatches, combined averages
- **Market context**: Closing line, spread, pace factor

### Model Evaluation

The model is evaluated using:
- **Accuracy**: Classification accuracy on test set
- **AUC**: Area under the ROC curve
- **Brier Score**: Calibration quality
- **Betting simulations**: ROI and win rate at various expected value thresholds

The model uses isotonic calibration to improve probability estimates for betting applications.