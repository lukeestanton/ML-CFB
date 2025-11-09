from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import click
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_cfb.clients.cfbd_client import build_cfbd_client
from ml_cfb.config import load_settings
from ml_cfb.ingest.games import fetch_games_for_season
from ml_cfb.ingest.lines import fetch_lines_for_season
from ml_cfb.io.storage import write_csv
from ml_cfb.transform.parsing import build_totals_dataset


def _season_iter(range_tuple: Tuple[int, int]) -> List[int]:
    start, end = range_tuple
    return list(range(start, end + 1))


@click.command()
@click.option("--start", type=int, default=None, help="Start season (e.g., 2018)")
@click.option("--end", type=int, default=None, help="End season (e.g., 2025)")
def main(start: int | None, end: int | None) -> None:
    settings = load_settings()
    if not settings.cfbd_api_key:
        click.echo("ERROR: CFBD_API_KEY not set", err=True)
        sys.exit(1)

    start_end = (
        (start, end)
        if start is not None and end is not None
        else settings.season_range
    )

    seasons = _season_iter(start_end)

    client = build_cfbd_client(settings.cfbd_api_key)

    all_games: list[pd.DataFrame] = []
    all_lines: list[pd.DataFrame] = []

    for season in seasons:
        click.echo(f"Fetching season {season} games and lines")
        games_df = fetch_games_for_season(client, season=season, season_type=None)
        lines_df = fetch_lines_for_season(client, season=season)

        # Raw snapshots
        write_csv(games_df, settings.paths.data_raw / f"games_{season}.csv")
        write_csv(lines_df, settings.paths.data_raw / f"lines_{season}.csv")

        all_games.append(games_df)
        all_lines.append(lines_df)

    if not all_games:
        click.echo("No games fetched; nothing to process.")
        sys.exit(0)

    games_all = pd.concat(all_games, ignore_index=True)
    lines_all = pd.concat(all_lines, ignore_index=True) if all_lines else pd.DataFrame()

    totals_df = build_totals_dataset(games_all, lines_all)

    out_path = settings.paths.data_processed / "totals_dataset.csv"
    write_csv(totals_df, out_path)
    click.echo(f"Wrote processed dataset to {out_path}")


if __name__ == "__main__":
    main()


