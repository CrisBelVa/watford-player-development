import json
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from dotenv import load_dotenv


class GoogleSheetsClient:
    """Lightweight Google Sheets client with lazy connection and DataFrame helpers."""

    PLAYERS_SHEET = "Players"
    SESSIONS_SHEET = "Sesions"

    def __init__(
        self,
        spreadsheet_id: Optional[str] = None,
        service_account_file: Optional[str] = None,
        service_account_json: Optional[str] = None,
    ):
        load_dotenv()
        self.spreadsheet_id = (
            spreadsheet_id
            or os.getenv("ID_SHEETS")
            or os.getenv("GOOGLE_SHEETS_ID")
        )
        self.service_account_file = (
            service_account_file
            or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
            or os.getenv("SERVICE_ACCOUNT_JSON_PATH")
            or os.getenv("SERVICE_ACCOUNT_FILE")
        )
        self.service_account_json = (
            service_account_json
            or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            or os.getenv("SERVICE_ACCOUNT_JSON")
        )

        self._client = None
        self._spreadsheet = None

        if not self.service_account_file:
            default_sa = Path("service_account.json")
            if default_sa.exists():
                self.service_account_file = str(default_sa)

    def is_configured(self) -> bool:
        if not self.spreadsheet_id:
            return False
        if self.service_account_json:
            return True
        if not self.service_account_file:
            return False
        return Path(self.service_account_file).expanduser().exists()

    def read_sheet_df(
        self,
        sheet_name: str,
        expected_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        ws = self._get_worksheet(sheet_name, create_if_missing=False)
        if ws is None:
            return pd.DataFrame(columns=list(expected_columns or []))

        values = ws.get_all_values()
        if not values:
            return pd.DataFrame(columns=list(expected_columns or []))

        header_row = values[0] if len(values) > 0 else []
        if isinstance(header_row, (list, tuple)):
            header_cells = list(header_row)
        elif header_row is None:
            header_cells = []
        else:
            header_cells = [header_row]
        header = [str(c).strip() for c in header_cells]
        if not header:
            header = list(expected_columns or [])
            if not header:
                return pd.DataFrame()
        header = [c if c else f"column_{i + 1}" for i, c in enumerate(header)]

        rows = values[1:]
        normalized_rows = [self._pad_row(r, len(header)) for r in rows]
        df = pd.DataFrame(normalized_rows, columns=header)

        if not df.empty:
            # Drop fully empty rows.
            keep_mask = df.apply(
                lambda row: any(str(v).strip() != "" for v in row.values),
                axis=1,
            )
            df = df.loc[keep_mask].reset_index(drop=True)

        if expected_columns:
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None

        return df

    def write_sheet_df(self, sheet_name: str, df: pd.DataFrame):
        ws = self._get_worksheet(sheet_name, create_if_missing=True)
        if ws is None:
            raise RuntimeError(f"Sheet '{sheet_name}' not found and could not be created.")

        out_df = df.copy()
        out_df.columns = [str(c).strip() for c in out_df.columns]

        if out_df.empty:
            ws.clear()
            if len(out_df.columns) > 0:
                ws.update("A1", [list(out_df.columns)], value_input_option="USER_ENTERED")
            return

        for col in out_df.columns:
            out_df[col] = out_df[col].apply(self._to_sheet_cell)

        values = [list(out_df.columns)] + out_df.values.tolist()

        required_rows = max(1000, len(values) + 10)
        required_cols = max(26, len(out_df.columns) + 3)
        ws.resize(rows=required_rows, cols=required_cols)
        ws.clear()
        ws.update("A1", values, value_input_option="USER_ENTERED")

    def read_players_df(self) -> pd.DataFrame:
        return self.read_sheet_df(
            self.PLAYERS_SHEET,
            expected_columns=[
                "internal_id",
                "playerName",
                "playerId",
                "internal_position",
                "activo",
                "photo_url",
            ],
        )

    def write_players_df(self, df: pd.DataFrame):
        self.write_sheet_df(self.PLAYERS_SHEET, df)

    def read_sessions_df(self) -> pd.DataFrame:
        return self.read_sheet_df(
            self.SESSIONS_SHEET,
            expected_columns=[
                "Mes",
                "Player",
                "Individual Training",
                "Meeting",
                "Review Clips",
            ],
        )

    def write_sessions_df(self, df: pd.DataFrame):
        self.write_sheet_df(self.SESSIONS_SHEET, df)

    def _get_worksheet(self, sheet_name: str, create_if_missing: bool):
        spreadsheet = self._get_spreadsheet()
        try:
            return spreadsheet.worksheet(sheet_name)
        except Exception as exc:
            if not create_if_missing:
                return None
            try:
                return spreadsheet.add_worksheet(title=sheet_name, rows=2000, cols=40)
            except Exception as create_exc:
                raise RuntimeError(
                    f"Unable to access/create worksheet '{sheet_name}': {create_exc}"
                ) from exc

    def _get_spreadsheet(self):
        if self._spreadsheet is not None:
            return self._spreadsheet

        client = self._get_client()
        try:
            self._spreadsheet = client.open_by_key(self.spreadsheet_id)
        except Exception as exc:
            raise RuntimeError(
                "Could not open Google Sheet by ID. "
                "Check ID_SHEETS and service account sharing permissions."
            ) from exc
        return self._spreadsheet

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import gspread
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'gspread'. Install project requirements."
            ) from exc

        if not self.spreadsheet_id:
            raise RuntimeError("Missing spreadsheet ID. Set ID_SHEETS or GOOGLE_SHEETS_ID.")

        if self.service_account_json:
            cred_text = self.service_account_json.strip()
            if cred_text.startswith("{"):
                try:
                    creds_dict = json.loads(cred_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("SERVICE_ACCOUNT_JSON is not valid JSON.") from exc
                self._client = gspread.service_account_from_dict(creds_dict)
                return self._client

            # Allow using the env var as a file path by mistake.
            candidate_path = Path(cred_text).expanduser()
            if candidate_path.exists():
                self._client = gspread.service_account(filename=str(candidate_path))
                return self._client

        if not self.service_account_file:
            raise RuntimeError(
                "Missing service account path. Set GOOGLE_SERVICE_ACCOUNT_FILE or SERVICE_ACCOUNT_JSON_PATH."
            )

        creds_path = Path(self.service_account_file).expanduser()
        if not creds_path.exists():
            raise RuntimeError(f"Service account file not found: {creds_path}")

        self._client = gspread.service_account(filename=str(creds_path))
        return self._client

    @staticmethod
    def _pad_row(row: object, width: int) -> list:
        if row is None:
            return [""] * max(width, 0)
        if not isinstance(row, (list, tuple)):
            row = [row]
        if len(row) >= width:
            return list(row[:width])
        return list(row) + [""] * (width - len(row))

    @staticmethod
    def _to_sheet_cell(value):
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value).strip()
