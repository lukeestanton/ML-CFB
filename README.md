# ML-CFB

## Data ingestion and parsing for CFB totals

This repo includes a pipeline to ingest CollegeFootballData (CFBD) games and sportsbook totals lines, then parse a dataset for modeling Over/Under outcomes.

### Usage Steps
1. Create and activate a virtual environment, then install deps:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Set CFBD API key:
Create .env file in root
```bash
echo "CFBD_API_KEY=YOUR_KEY_HERE" >> .env
```
Optionally also set `CFBD_START_SEASON` and `CFBD_END_SEASON` in `.env` (defaults: 2018–2025)

3. Build the dataset (change dates as necessary):
```bash
python scripts/build_dataset.py --start 2018 --end 2025
```

Artifacts:
- Raw: `data/raw/games_YYYY.csv`, `data/raw/lines_YYYY.csv`
- Processed: `data/processed/totals_dataset.csv`

### What gets built
Processed dataset columns:
- Identifiers: `game_id`, `season`, `week`, `start_date`, `home_team`, `away_team`
- Outcomes: `home_points`, `away_points`, `actual_total`
- Market: `closing_total`, `provider`
- Labels: `over_result` (1=Over, 0=Under, NaN=Push), `push` (bool)
- Residual: `actual_total_minus_total`

`closing_total` is selected from the preferred provider for each game (William Hill, DraftKings, ESPN Bet, etc.). Only regular season games with betting lines are included.

### Next steps (add others as needed)
- Rolling team features (last 3 games) for both teams per game
- Weather and attendance enrichment
- Postgres persistence (schema: `games`, `lines`, `totals_dataset`)
