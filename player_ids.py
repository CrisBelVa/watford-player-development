from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def clean_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s.lower() in {"nan", "none", "<na>"}:
        return ""
    return s


def normalize_whoscored_player_id(value: object) -> Optional[str]:
    """
    Normalize a WhoScored player id coming from Excel/CSV/DB.

    - Converts Excel-style floats like 495339.0 -> "495339"
    - Converts scientific notation like 4.95339e5 -> "495339" (only if integer)
    - Strips whitespace and commas
    - Returns None only when empty/NA
    """
    s = clean_str(value)
    if not s:
        return None

    s = s.replace(",", "")

    # Common Excel float string representation
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]

    # Scientific notation or other numeric strings
    try:
        f = float(s)
        if f.is_integer() and f >= 0:
            return str(int(f))
    except Exception:
        pass

    return s


def whoscored_player_url(value: object) -> Optional[str]:
    pid = normalize_whoscored_player_id(value)
    if not pid:
        return None
    if not re.fullmatch(r"\d+", pid):
        return None
    return f"https://www.whoscored.com/players/{pid}"

