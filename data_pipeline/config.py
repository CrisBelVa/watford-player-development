from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "scraping_sources.json"
EXAMPLE_CONFIG_PATH = ROOT_DIR / "config" / "scraping_sources.example.json"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    family: str
    fixtures_path: Path
    matches_dir: Path
    parsed_output_dir: Path
    state_path: Path
    fixtures_url: str | None = None
    competition_url: str | None = None
    metadata_path: Path | None = None
    season_start_month: int | None = None
    enabled: bool = True


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Create it from {EXAMPLE_CONFIG_PATH.name}."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_source_config(
    source_name: str | None = None,
    config_path: str | Path | None = None,
) -> SourceConfig:
    raw = load_config(config_path)
    selected_name = source_name or raw.get("default_source")
    if not selected_name:
        raise ValueError("No source selected and 'default_source' is missing in config.")

    sources = raw.get("sources", {})
    if selected_name not in sources:
        raise KeyError(f"Source '{selected_name}' is not defined in config.")

    item = sources[selected_name]
    family = item.get("family")
    if family not in {"whoscored", "scoresway"}:
        raise ValueError(
            f"Source '{selected_name}' has unsupported family '{family}'."
        )

    defaults = ROOT_DIR / "data" / "pipeline" / selected_name
    fixtures_default = defaults / "fixtures.csv"
    matches_default = defaults / "matches"
    parsed_default = matches_default / "output"
    state_default = defaults / "state.csv"

    return SourceConfig(
        name=selected_name,
        family=family,
        fixtures_path=_resolve_path(item.get("fixtures_path"), fixtures_default),
        matches_dir=_resolve_path(item.get("matches_dir"), matches_default),
        parsed_output_dir=_resolve_path(item.get("parsed_output_dir"), parsed_default),
        state_path=_resolve_path(item.get("state_path"), state_default),
        fixtures_url=item.get("fixtures_url"),
        competition_url=item.get("competition_url"),
        metadata_path=_resolve_path(item["metadata_path"], ROOT_DIR / "data")
        if item.get("metadata_path")
        else None,
        season_start_month=item.get("season_start_month"),
        enabled=bool(item.get("enabled", True)),
    )
