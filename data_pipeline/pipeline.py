from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import SourceConfig, load_source_config
from .legacy import (
    scoresway_parser,
    scoresway_scraper,
    whoscored_parser,
    whoscored_scraper,
)
from .load_db import (
    DbLoadMode,
    create_db_engine,
    get_existing_match_ids,
    load_parsed_outputs_to_db,
    validate_parsed_outputs_against_db,
)
from .state import build_state, determine_strategy, save_state

PipelineMode = Literal["fixtures", "download", "parse", "validate-db", "load-db", "full", "full-db"]
PipelineStrategy = Literal["auto", "bootstrap", "incremental"]
IncrementalStateSource = Literal["auto", "local", "db"]


@dataclass
class PipelineResult:
    source_name: str
    mode: PipelineMode
    strategy: PipelineStrategy | str = "auto"
    fixtures_rows: int = 0
    matches_dir: str = ""
    parsed_output_dir: str = ""
    state_path: str = ""
    json_count: int = 0
    parsed_csv_count: int = 0
    downloaded_match_count: int = 0
    parsed_match_count: int = 0
    pending_download_count: int = 0
    pending_parse_count: int = 0
    db_loaded_rows: int = 0
    db_validation_ok: bool | None = None
    cleaned_temp_files: bool = False


class PipelineRunner:
    def __init__(
        self,
        source_name: str | None = None,
        *,
        config_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.source = load_source_config(source_name, config_path)
        self.logger = logger or logging.getLogger("watford_pipeline")

    def run(
        self,
        mode: PipelineMode = "full",
        strategy: PipelineStrategy = "auto",
        db_mode: DbLoadMode = "append_new",
        incremental_state: IncrementalStateSource = "auto",
        cleanup_temp: bool = False,
    ) -> PipelineResult:
        if not self.source.enabled:
            raise RuntimeError(f"Source '{self.source.name}' is disabled in config.")

        self._ensure_directories()
        resolved_strategy = determine_strategy(
            strategy,
            matches_dir=self.source.matches_dir,
            parsed_output_dir=self.source.parsed_output_dir,
        )
        result = PipelineResult(
            source_name=self.source.name,
            mode=mode,
            strategy=resolved_strategy,
            matches_dir=str(self.source.matches_dir),
            parsed_output_dir=str(self.source.parsed_output_dir),
            state_path=str(self.source.state_path),
        )
        self.logger.info(
            "Running pipeline | source=%s | mode=%s | strategy=%s | incremental_state=%s | cleanup_temp=%s",
            self.source.name,
            mode,
            resolved_strategy,
            incremental_state,
            cleanup_temp,
        )

        fixtures_df: pd.DataFrame | None = None

        if mode in {"fixtures", "download", "full", "full-db"}:
            fixtures_df = self.fetch_fixtures()
            result.fixtures_rows = len(fixtures_df)

        if mode in {"download", "full", "full-db"}:
            if fixtures_df is None:
                fixtures_df = self._load_saved_fixtures()
            fixtures_df = self.prepare_incremental_fixtures(
                fixtures_df,
                strategy=resolved_strategy,
                incremental_state=incremental_state,
            )
            result.fixtures_rows = len(fixtures_df)
            self.download_matches(fixtures_df)

        if mode in {"parse", "full", "full-db"}:
            self.parse_matches()

        if mode in {"validate-db"}:
            result.db_validation_ok = self.validate_db()

        if mode in {"load-db", "full-db"}:
            validation_ok = self.validate_db()
            result.db_validation_ok = validation_ok
            if not validation_ok:
                raise RuntimeError("Database validation failed. Aborting DB load.")
            result.db_loaded_rows = self.load_db(db_mode=db_mode)
            if cleanup_temp:
                self.cleanup_temp_files()
                result.cleaned_temp_files = True

        state = build_state(
            strategy=resolved_strategy,
            fixtures_df=fixtures_df,
            matches_dir=self.source.matches_dir,
            parsed_output_dir=self.source.parsed_output_dir,
        )
        save_state(state, self.source.state_path)
        result.json_count = state.json_count
        result.parsed_csv_count = state.parsed_csv_count
        result.downloaded_match_count = state.downloaded_match_count
        result.parsed_match_count = state.parsed_match_count
        result.pending_download_count = state.pending_download_count
        result.pending_parse_count = state.pending_parse_count

        return result

    def prepare_incremental_fixtures(
        self,
        fixtures_df: pd.DataFrame,
        *,
        strategy: PipelineStrategy | str,
        incremental_state: IncrementalStateSource,
    ) -> pd.DataFrame:
        if strategy != "incremental":
            return fixtures_df

        state_mode = self.resolve_incremental_state_mode(incremental_state)
        self.logger.info("Preparing incremental fixtures using state source '%s'", state_mode)

        if state_mode == "local":
            return fixtures_df

        if "match_id" not in fixtures_df.columns:
            self.logger.warning("Fixtures DataFrame has no 'match_id' column; skipping DB incremental filter.")
            return fixtures_df

        engine = create_db_engine()
        existing_ids = get_existing_match_ids(engine, "match_data")
        if not existing_ids:
            return fixtures_df

        filtered = fixtures_df.copy()
        filtered["_match_id_int"] = pd.to_numeric(filtered["match_id"], errors="coerce").astype("Int64")
        filtered = filtered[~filtered["_match_id_int"].isin(existing_ids)].drop(columns=["_match_id_int"])
        self.logger.info(
            "Incremental DB filter reduced fixtures from %s to %s",
            len(fixtures_df),
            len(filtered),
        )
        return filtered

    def resolve_incremental_state_mode(
        self,
        requested: IncrementalStateSource,
    ) -> IncrementalStateSource:
        if requested in {"local", "db"}:
            return requested

        has_local_json = any(self.source.matches_dir.glob("*.json"))
        has_local_csv = any(self.source.parsed_output_dir.glob("*.csv"))
        return "local" if (has_local_json or has_local_csv) else "db"

    def fetch_fixtures(self) -> pd.DataFrame:
        self.logger.info("Fetching fixtures for source '%s'", self.source.name)
        fixtures_df = self._fetch_fixtures_for_source()
        self.source.fixtures_path.parent.mkdir(parents=True, exist_ok=True)
        fixtures_df.to_csv(self.source.fixtures_path, index=False)
        self.logger.info(
            "Saved %s fixtures to %s", len(fixtures_df), self.source.fixtures_path
        )
        return fixtures_df

    def download_matches(self, fixtures_df: pd.DataFrame) -> None:
        self.logger.info("Downloading match JSON files into %s", self.source.matches_dir)
        if self.source.family == "whoscored":
            scraper = whoscored_scraper()
            scraper.get_json_games(
                fixtures_df.copy(),
                str(self.source.matches_dir),
                ini=self.source.season_start_month or 200001,
            )
            return

        scraper = scoresway_scraper()
        if not self.source.competition_url:
            raise ValueError("scoresway sources require 'competition_url' in config.")
        scraper.get_json_games(
            fixtures_df.copy(),
            str(self.source.matches_dir),
            self.source.competition_url,
        )

    def parse_matches(self) -> None:
        self.logger.info("Parsing downloaded matches into tabular outputs")
        self.source.parsed_output_dir.mkdir(parents=True, exist_ok=True)
        output_name = self.source.parsed_output_dir.name

        if self.source.family == "whoscored":
            parser = whoscored_parser()
            parser.procesar_ficheros_lista(str(self.source.matches_dir), output_name)
            return

        parser = scoresway_parser()
        parser.procesar_ficheros_lista(str(self.source.matches_dir), output_name)

    def load_db(self, *, db_mode: DbLoadMode = "append_new") -> int:
        self.logger.info(
            "Loading parsed outputs into database | dir=%s | mode=%s",
            self.source.parsed_output_dir,
            db_mode,
        )
        results = load_parsed_outputs_to_db(
            self.source.parsed_output_dir,
            mode=db_mode,
        )
        total_loaded = 0
        for item in results:
            total_loaded += item.loaded_rows
            self.logger.info(
                "DB load | table=%s | files=%s | source_rows=%s | loaded=%s | skipped=%s | mode=%s | table_exists=%s | common_cols=%s | extra_csv_cols=%s | missing_csv_cols=%s",
                item.table_name,
                item.source_files,
                item.source_rows,
                item.loaded_rows,
                item.skipped_rows,
                item.mode,
                item.table_exists,
                item.common_columns,
                item.extra_in_csv,
                item.missing_in_csv,
            )
        return total_loaded

    def validate_db(self) -> bool:
        self.logger.info(
            "Validating parsed outputs against database schema | dir=%s",
            self.source.parsed_output_dir,
        )
        results = validate_parsed_outputs_against_db(self.source.parsed_output_dir)
        ok = True
        for item in results:
            if not item.can_load:
                ok = False
            self.logger.info(
                "DB validation | table=%s | exists=%s | files=%s | rows=%s | csv_cols=%s | db_cols=%s | common_cols=%s | can_load=%s | extra_csv_cols=%s | missing_csv_cols=%s",
                item.table_name,
                item.table_exists,
                item.source_files,
                item.source_rows,
                item.csv_columns,
                item.db_columns,
                item.common_columns,
                item.can_load,
                ", ".join(item.extra_in_csv[:20]),
                ", ".join(item.missing_in_csv[:20]),
            )
        return ok

    def cleanup_temp_files(self) -> None:
        self.logger.info(
            "Cleaning temporary scraping files | matches_dir=%s | parsed_dir=%s",
            self.source.matches_dir,
            self.source.parsed_output_dir,
        )
        for json_file in self.source.matches_dir.glob("*.json"):
            json_file.unlink(missing_ok=True)

        if self.source.parsed_output_dir.exists():
            shutil.rmtree(self.source.parsed_output_dir)
        self.source.parsed_output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_fixtures_for_source(self) -> pd.DataFrame:
        if self.source.family == "whoscored":
            if not self.source.fixtures_url:
                raise ValueError("whoscored sources require 'fixtures_url' in config.")
            scraper = whoscored_scraper()
            return scraper.scrape_fixtures(
                self.source.fixtures_url,
                mes_ini=self.source.season_start_month or 200001,
            )

        if not self.source.competition_url:
            raise ValueError("scoresway sources require 'competition_url' in config.")
        if self.source.metadata_path is None:
            raise ValueError("scoresway sources require 'metadata_path' in config.")

        scraper = scoresway_scraper()
        metadata = pd.read_excel(self.source.metadata_path)
        return scraper.scrape_fixtures(metadata, self.source.competition_url)

    def _load_saved_fixtures(self) -> pd.DataFrame:
        if not self.source.fixtures_path.exists():
            raise FileNotFoundError(
                f"Fixtures file not found: {self.source.fixtures_path}. Run mode='fixtures' first."
            )
        return pd.read_csv(self.source.fixtures_path)

    def _ensure_directories(self) -> None:
        self.source.matches_dir.mkdir(parents=True, exist_ok=True)
        self.source.parsed_output_dir.mkdir(parents=True, exist_ok=True)
        self.source.state_path.parent.mkdir(parents=True, exist_ok=True)
