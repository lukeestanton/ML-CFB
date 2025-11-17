# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    data_processed: Path


@dataclass(frozen=True)
class Settings:
    cfbd_api_key: str
    season_range: Tuple[int, int]
    paths: Paths


def load_settings() -> Settings:
    load_dotenv(override=False)

    project_root = Path(__file__).resolve().parents[1]
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"

    data_raw.mkdir(parents=True, exist_ok=True)
    data_processed.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("CFBD_API_KEY", "").strip()
    start = int(os.getenv("CFBD_START_SEASON", "2019"))
    end = int(os.getenv("CFBD_END_SEASON", "2025"))

    return Settings(
        cfbd_api_key=api_key,
        season_range=(start, end),
        paths=Paths(
            project_root=project_root,
            data_raw=data_raw,
            data_processed=data_processed,
        ),
    )


