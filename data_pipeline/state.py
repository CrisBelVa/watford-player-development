from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class PipelineState:
    strategy: str
    fixtures_count: int
    json_count: int
    parsed_csv_count: int
    parsed_match_count: int
    downloaded_match_count: int
    pending_download_count: int
    pending_parse_count: int
    last_run_utc: str


def collect_match_ids_from_json(matches_dir: Path) -> set[str]:
    return {
        path.stem.split("_")[-1]
        for path in matches_dir.glob("*.json")
        if path.is_file() and "_" in path.stem
    }


def collect_match_ids_from_output(parsed_output_dir: Path) -> set[str]:
    return {
        path.name.split("_")[0]
        for path in parsed_output_dir.glob("*_eventData.csv")
        if path.is_file() and "_" in path.name
    }


def determine_strategy(
    requested: str,
    *,
    matches_dir: Path,
    parsed_output_dir: Path,
) -> str:
    if requested in {"bootstrap", "incremental"}:
        return requested

    has_json = any(matches_dir.glob("*.json"))
    has_output = any(parsed_output_dir.glob("*.csv"))
    return "incremental" if (has_json or has_output) else "bootstrap"


def build_state(
    *,
    strategy: str,
    fixtures_df: pd.DataFrame | None,
    matches_dir: Path,
    parsed_output_dir: Path,
) -> PipelineState:
    fixtures_count = len(fixtures_df) if fixtures_df is not None else 0
    fixture_match_ids = set()
    if fixtures_df is not None and "match_id" in fixtures_df.columns:
        fixture_match_ids = {
            str(value).strip()
            for value in fixtures_df["match_id"].dropna().tolist()
        }

    downloaded_match_ids = collect_match_ids_from_json(matches_dir)
    parsed_match_ids = collect_match_ids_from_output(parsed_output_dir)
    json_files = list(matches_dir.glob("*.json"))
    parsed_csv_files = list(parsed_output_dir.glob("*.csv"))

    pending_download = max(len(fixture_match_ids - downloaded_match_ids), 0)
    pending_parse = max(len(downloaded_match_ids - parsed_match_ids), 0)

    return PipelineState(
        strategy=strategy,
        fixtures_count=fixtures_count,
        json_count=len(json_files),
        parsed_csv_count=len(parsed_csv_files),
        parsed_match_count=len(parsed_match_ids),
        downloaded_match_count=len(downloaded_match_ids),
        pending_download_count=pending_download,
        pending_parse_count=pending_parse,
        last_run_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_state(state: PipelineState, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([state.__dict__])
    df.to_csv(state_path, index=False)
