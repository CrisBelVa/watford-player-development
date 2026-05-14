from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_pipeline.logging_utils import setup_logger
from data_pipeline.pipeline import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Watford football data pipeline."
    )
    parser.add_argument(
        "--source",
        help="Source name defined in config/scraping_sources.json",
    )
    parser.add_argument(
        "--mode",
        choices=["fixtures", "download", "parse", "validate-db", "load-db", "full", "full-db"],
        default="full",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--db-mode",
        choices=["append_new", "replace_matches", "replace_table"],
        default="append_new",
        help="Database loading behavior used by --mode load-db or full-db.",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "bootstrap", "incremental"],
        default="auto",
        help="Execution strategy. 'auto' uses bootstrap on first run and incremental afterwards.",
    )
    parser.add_argument(
        "--incremental-state",
        choices=["auto", "local", "db"],
        default="auto",
        help="Where incremental runs look to know what already exists. Use 'db' for ephemeral server runs.",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="Delete temporary JSON and CSV files after a successful DB load.",
    )
    parser.add_argument(
        "--config",
        default="config/scraping_sources.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--log-file",
        default="data/pipeline/pipeline.log",
        help="Path to pipeline log file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(log_file=Path(args.log_file))
    runner = PipelineRunner(args.source, config_path=args.config, logger=logger)
    result = runner.run(
        args.mode,
        strategy=args.strategy,
        db_mode=args.db_mode,
        incremental_state=args.incremental_state,
        cleanup_temp=args.cleanup_temp,
    )
    logger.info(
        "Pipeline finished | source=%s | mode=%s | strategy=%s | fixtures=%s | json=%s | parsed_csv=%s | downloaded_matches=%s | parsed_matches=%s | pending_download=%s | pending_parse=%s | db_validation_ok=%s | db_loaded_rows=%s | cleaned_temp_files=%s | matches_dir=%s | parsed_dir=%s | state=%s",
        result.source_name,
        result.mode,
        result.strategy,
        result.fixtures_rows,
        result.json_count,
        result.parsed_csv_count,
        result.downloaded_match_count,
        result.parsed_match_count,
        result.pending_download_count,
        result.pending_parse_count,
        result.db_validation_ok,
        result.db_loaded_rows,
        result.cleaned_temp_files,
        result.matches_dir,
        result.parsed_output_dir,
        result.state_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
