from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.sql.sqltypes import String


DbLoadMode = Literal["append_new", "replace_matches", "replace_table"]


TABLE_SPECS = {
    "matchData": {
        "table": "match_data",
        "pattern": "*_matchData.csv",
        "dedupe": ["matchId"],
        "chunksize": 1000,
        "method": "multi",
    },
    "teamData": {
        "table": "team_data",
        "pattern": "*_teamData.csv",
        "dedupe": ["matchId", "teamId"],
        "chunksize": 1000,
        "method": "multi",
        "rename": {
            "matchId": "match_id",
            "teamId": "team_id",
            "teamName": "team_name",
            "countryName": "country_name",
            "managerName": "manager_name",
            "field": "side",
            "averageAge": "average_age",
            "scores.halftime": "scores_halftime",
            "scores.fulltime": "scores_fulltime",
            "scores.running": "scores_running",
        },
    },
    "teamStats": {
        "table": "team_stats",
        "pattern": "*_teamStats.csv",
        "dedupe": ["matchId", "teamId"],
        "chunksize": 1000,
        "method": "multi",
    },
    "playerData": {
        "table": "player_data",
        "pattern": "*_playerData.csv",
        "dedupe": ["matchId", "playerId"],
        "chunksize": 1000,
        "method": "multi",
    },
    "playerStats": {
        "table": "player_stats",
        "pattern": "*_playerStats.csv",
        "dedupe": ["matchId", "playerId"],
        "chunksize": 1000,
        "method": "multi",
    },
    "eventData": {
        "table": "event_data",
        "pattern": "*_eventData.csv",
        "dedupe": ["matchId", "id", "eventId"],
        "chunksize": 100,
        "method": None,
    },
}


@dataclass
class TableLoadResult:
    table_name: str
    source_files: int
    source_rows: int
    loaded_rows: int
    skipped_rows: int
    mode: DbLoadMode
    table_exists: bool = False
    csv_columns: int = 0
    db_columns: int = 0
    common_columns: int = 0
    missing_in_csv: str = ""
    extra_in_csv: str = ""


@dataclass
class TableValidationResult:
    table_name: str
    source_files: int
    source_rows: int
    table_exists: bool
    csv_columns: int
    db_columns: int
    common_columns: int
    missing_in_csv: list[str]
    extra_in_csv: list[str]
    can_load: bool


def create_db_engine() -> Engine:
    load_dotenv()
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    missing = [
        name
        for name, value in {
            "DB_USER": user,
            "DB_PASSWORD": password,
            "DB_HOST": host,
            "DB_PORT": port,
            "DB_NAME": database,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing database environment variables: {', '.join(missing)}")

    url = URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port) if port else None,
        database=database,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)


def read_table_csvs(parsed_output_dir: Path, pattern: str) -> tuple[pd.DataFrame, int]:
    files = sorted(parsed_output_dir.glob(pattern))
    frames = []
    for path in files:
        df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
        df["_source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(), 0
    return pd.concat(frames, ignore_index=True), len(files)


def inspect_table_csvs(parsed_output_dir: Path, pattern: str) -> tuple[list[str], int]:
    files = sorted(parsed_output_dir.glob(pattern))
    columns: list[str] = []
    seen = set()
    for path in files:
        header_df = pd.read_csv(path, sep=";", decimal=",", nrows=0)
        for column in header_df.columns:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns, len(files)


def normalize_match_id(df: pd.DataFrame) -> pd.DataFrame:
    if "matchId" in df.columns:
        df["matchId"] = pd.to_numeric(df["matchId"], errors="coerce").astype("Int64")
    if "match_id" in df.columns:
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    return df


def apply_table_renames(df: pd.DataFrame, rename_map: dict[str, str] | None) -> pd.DataFrame:
    if not rename_map:
        return df
    return df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})


def clean_dataframe(
    df: pd.DataFrame,
    dedupe_columns: list[str],
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = apply_table_renames(df.copy(), rename_map)
    dedupe_columns = [rename_map.get(col, col) for col in dedupe_columns] if rename_map else dedupe_columns
    df = normalize_match_id(df.copy())
    available_dedupe = [col for col in dedupe_columns if col in df.columns]
    if available_dedupe:
        df = df.drop_duplicates(subset=available_dedupe, keep="last")
    return df.where(pd.notna(df), None)


def table_exists(engine: Engine, table_name: str) -> bool:
    return inspect(engine).has_table(table_name)


def get_db_columns(engine: Engine, table_name: str) -> list[str]:
    if not table_exists(engine, table_name):
        return []
    return [column["name"] for column in inspect(engine).get_columns(table_name)]


def get_db_column_metadata(engine: Engine, table_name: str) -> dict[str, dict]:
    if not table_exists(engine, table_name):
        return {}
    return {
        column["name"]: column
        for column in inspect(engine).get_columns(table_name)
    }


def align_to_existing_table(
    engine: Engine,
    table_name: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    db_columns = get_db_columns(engine, table_name)
    if not db_columns:
        return df, [], [], []

    csv_columns = [col for col in df.columns if col != "_source_file"]
    common_columns = [col for col in db_columns if col in csv_columns]
    missing_in_csv = [col for col in db_columns if col not in csv_columns]
    extra_in_csv = [col for col in csv_columns if col not in db_columns]

    aligned = df[common_columns + (["_source_file"] if "_source_file" in df.columns else [])].copy()
    return aligned, common_columns, missing_in_csv, extra_in_csv


def sanitize_dataframe_for_db(
    engine: Engine,
    table_name: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    metadata = get_db_column_metadata(engine, table_name)
    if not metadata or df.empty:
        return df

    out = df.copy()
    for column_name, column_meta in metadata.items():
        if column_name not in out.columns:
            continue

        column_type = column_meta.get("type")
        max_length = getattr(column_type, "length", None)
        if not isinstance(column_type, String) or not max_length:
            continue

        series = out[column_name]
        non_null = series.notna()
        if not non_null.any():
            continue

        as_text = series.loc[non_null].astype(str)
        too_long_mask = as_text.str.len() > max_length
        if too_long_mask.any():
            out.loc[as_text.index[too_long_mask], column_name] = (
                as_text[too_long_mask].str.slice(0, max_length)
            )

    return out


def get_existing_match_ids(engine: Engine, table_name: str) -> set[int]:
    if not table_exists(engine, table_name):
        return set()
    try:
        df = pd.read_sql(f"SELECT DISTINCT matchId FROM {table_name}", con=engine)
    except Exception:
        return set()
    if "matchId" not in df.columns:
        return set()
    return {
        int(value)
        for value in pd.to_numeric(df["matchId"], errors="coerce").dropna().tolist()
    }


def get_existing_key_rows(
    engine: Engine,
    table_name: str,
    key_columns: list[str],
) -> pd.DataFrame:
    if not table_exists(engine, table_name) or not key_columns:
        return pd.DataFrame()

    db_columns = get_db_columns(engine, table_name)
    usable_columns = [col for col in key_columns if col in db_columns]
    if not usable_columns:
        return pd.DataFrame()

    query = f"SELECT DISTINCT {', '.join(usable_columns)} FROM {table_name}"
    try:
        df = pd.read_sql(query, con=engine)
    except Exception:
        return pd.DataFrame()

    for column in usable_columns:
        if column in {"matchId", "match_id", "teamId", "team_id", "playerId", "player_id", "eventId"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def filter_append_new(
    engine: Engine,
    table_name: str,
    df: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        return df

    usable_columns = [col for col in key_columns if col in df.columns]
    if not usable_columns:
        if "matchId" in df.columns:
            existing_ids = get_existing_match_ids(engine, table_name)
            if not existing_ids:
                return df
            return df[~df["matchId"].isin(existing_ids)].copy()
        return df

    existing_key_rows = get_existing_key_rows(engine, table_name, usable_columns)
    if existing_key_rows.empty:
        return df

    left = df.copy()
    right = existing_key_rows.copy()
    left["_exists_marker"] = 1
    merged = left.merge(right.drop_duplicates(), on=usable_columns, how="left", indicator=True)
    filtered = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return filtered.drop(columns=["_exists_marker"], errors="ignore")


def delete_existing_matches(engine: Engine, table_name: str, match_ids: list[int]) -> None:
    if not match_ids or not table_exists(engine, table_name):
        return
    placeholders = ", ".join([f":id_{idx}" for idx, _ in enumerate(match_ids)])
    params = {f"id_{idx}": match_id for idx, match_id in enumerate(match_ids)}
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE matchId IN ({placeholders})"),
            params,
        )


def load_dataframe_to_table(
    engine: Engine,
    table_name: str,
    df: pd.DataFrame,
    *,
    mode: DbLoadMode,
    chunksize: int,
    insert_method: str | None,
) -> int:
    if df.empty:
        return 0

    df_to_load = df.drop(columns=["_source_file"], errors="ignore").copy()
    df_to_load = sanitize_dataframe_for_db(engine, table_name, df_to_load)

    if mode == "replace_table":
        if_exists = "replace"
    else:
        if_exists = "append"

    if mode == "replace_matches" and "matchId" in df_to_load.columns:
        match_ids = [
            int(value)
            for value in pd.to_numeric(df_to_load["matchId"], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        ]
        delete_existing_matches(engine, table_name, match_ids)

    df_to_load.to_sql(
        table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method=insert_method,
    )
    return len(df_to_load)


def load_parsed_outputs_to_db(
    parsed_output_dir: Path,
    *,
    mode: DbLoadMode = "append_new",
    chunksize: int = 1000,
    engine: Engine | None = None,
) -> list[TableLoadResult]:
    if not parsed_output_dir.exists():
        raise FileNotFoundError(f"Parsed output directory not found: {parsed_output_dir}")

    db_engine = engine or create_db_engine()
    results: list[TableLoadResult] = []

    for spec in TABLE_SPECS.values():
        table_name = spec["table"]
        exists = table_exists(db_engine, table_name)
        db_columns = get_db_columns(db_engine, table_name)
        raw_df, source_files = read_table_csvs(parsed_output_dir, spec["pattern"])
        if raw_df.empty:
            results.append(
                TableLoadResult(
                    table_name=table_name,
                    source_files=source_files,
                    source_rows=0,
                    loaded_rows=0,
                    skipped_rows=0,
                    mode=mode,
                    table_exists=exists,
                    csv_columns=0,
                    db_columns=len(db_columns),
                    common_columns=0,
                )
            )
            continue

        df = clean_dataframe(raw_df, spec["dedupe"], spec.get("rename"))
        df, common_columns, missing_in_csv, extra_in_csv = align_to_existing_table(
            db_engine,
            table_name,
            df,
        )
        source_rows = len(df)
        key_columns = [col for col in spec["dedupe"] if col in df.columns]

        df_to_load = filter_append_new(db_engine, table_name, df, key_columns) if mode == "append_new" else df
        loaded_rows = load_dataframe_to_table(
            db_engine,
            table_name,
            df_to_load,
            mode=mode,
            chunksize=spec.get("chunksize", chunksize),
            insert_method=spec.get("method", "multi"),
        )
        results.append(
            TableLoadResult(
                table_name=table_name,
                source_files=source_files,
                source_rows=source_rows,
                loaded_rows=loaded_rows,
                skipped_rows=source_rows - loaded_rows,
                mode=mode,
                table_exists=exists,
                csv_columns=len([col for col in raw_df.columns if col != "_source_file"]),
                db_columns=len(db_columns),
                common_columns=len(common_columns) if db_columns else len(df.columns),
                missing_in_csv=", ".join(missing_in_csv[:20]),
                extra_in_csv=", ".join(extra_in_csv[:20]),
            )
        )

    return results


def validate_parsed_outputs_against_db(
    parsed_output_dir: Path,
    *,
    engine: Engine | None = None,
) -> list[TableValidationResult]:
    if not parsed_output_dir.exists():
        raise FileNotFoundError(f"Parsed output directory not found: {parsed_output_dir}")

    db_engine = engine or create_db_engine()
    results: list[TableValidationResult] = []

    for spec in TABLE_SPECS.values():
        table_name = spec["table"]
        csv_columns, source_files = inspect_table_csvs(parsed_output_dir, spec["pattern"])
        rename_map = spec.get("rename") or {}
        csv_columns = [rename_map.get(col, col) for col in csv_columns]
        exists = table_exists(db_engine, table_name)
        db_columns = get_db_columns(db_engine, table_name)

        if not exists:
            results.append(
                TableValidationResult(
                    table_name=table_name,
                    source_files=source_files,
                    source_rows=-1,
                    table_exists=False,
                    csv_columns=len(csv_columns),
                    db_columns=0,
                    common_columns=0,
                    missing_in_csv=[],
                    extra_in_csv=[],
                    can_load=True,
                )
            )
            continue

        missing_in_csv = [col for col in db_columns if col not in csv_columns]
        extra_in_csv = [col for col in csv_columns if col not in db_columns]
        common_columns = [col for col in db_columns if col in csv_columns]
        required_keys = [col for col in spec["dedupe"] if col in db_columns]
        can_load = bool(common_columns) and all(col in csv_columns for col in required_keys)

        results.append(
            TableValidationResult(
                table_name=table_name,
                source_files=source_files,
                source_rows=-1,
                table_exists=True,
                csv_columns=len(csv_columns),
                db_columns=len(db_columns),
                common_columns=len(common_columns),
                missing_in_csv=missing_in_csv,
                extra_in_csv=extra_in_csv,
                can_load=can_load,
            )
        )

    return results
