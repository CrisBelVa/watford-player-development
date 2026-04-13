import os
import re
import html
import base64
import mimetypes
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
from db_utils import connect_to_db
from player_ids import normalize_whoscored_player_id, whoscored_player_url
from PIL import Image, ImageOps
from typing import Optional
from utils.sheets_client import GoogleSheetsClient
# --- Config --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "img")
LOGO_PATH = os.path.join(IMG_DIR, "watford_logo.png")
PLAYER_PHOTOS_DIR = os.path.join(BASE_DIR, "data", "player_photos")
INTERNAL_ID_COL = "internal_id"
INTERNAL_ID_PREFIX = "INT"
INTERNAL_ID_PAD = 5
INTERNAL_POSITION_COL = "internal_position"
AUTO_POSITION_LABEL = "Auto (WhoScored)"

POSITION_CODE_TO_LABEL = {
    "GK": "Goalkeeper",
    "DC": "Center Back", "CB": "Center Back",
    "DL": "Left Back", "LB": "Left Back", "DML": "Left Back", "LWB": "Left Back",
    "DR": "Right Back", "RB": "Right Back", "DMR": "Right Back", "RWB": "Right Back",
    "DMC": "Defensive Midfielder", "DM": "Defensive Midfielder",
    "MC": "Midfielder", "CM": "Midfielder", "ML": "Midfielder", "LM": "Midfielder",
    "MR": "Midfielder", "RM": "Midfielder",
    "AMC": "Attacking Midfielder", "AM": "Attacking Midfielder",
    "AML": "Left Winger", "FWL": "Left Winger", "LW": "Left Winger",
    "AMR": "Right Winger", "FWR": "Right Winger", "RW": "Right Winger",
    "FW": "Striker", "ST": "Striker", "CF": "Striker",
    "Sub": "Substitute",
}

POSITION_ORDER = [
    "Goalkeeper",
    "Center Back",
    "Left Back",
    "Right Back",
    "Defensive Midfielder",
    "Midfielder",
    "Attacking Midfielder",
    "Left Winger",
    "Right Winger",
    "Striker",
    "Unknown",
]
INTERNAL_POSITION_OPTIONS = [AUTO_POSITION_LABEL] + POSITION_ORDER.copy()

st.set_page_config(
    page_title="Manage players list",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Auth ----------------------------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be logged in to view this page.")
    st.stop()

if st.session_state.get("user_type") != "staff":
    st.warning("Only staff users can access this page.")
    st.stop()

# --- Helpers -------------------------------------------------
PLAYERS_FILE_XLSX = os.path.join(BASE_DIR, "data", "watford_players_login_info.xlsx")

@st.cache_resource(show_spinner=False)
def get_sheets_client() -> GoogleSheetsClient:
    return GoogleSheetsClient()

@st.cache_data(show_spinner=False)
def load_players_df(path: str) -> pd.DataFrame:
    """Load players DataFrame from Google Sheets (fallback local Excel).
    Ensures the columns: internal_id (str), playerName (str), playerId (string),
    internal_position (str), activo (int), photo_url (string).
    """
    df = None
    sheets_client = get_sheets_client()
    if sheets_client.is_configured():
        try:
            df = sheets_client.read_players_df()
        except Exception as exc:
            st.warning(f"Could not read Google Sheets tab 'Players'. Using local fallback. ({exc})")

    if df is None:
        if not os.path.exists(path):
            return pd.DataFrame(
                columns=[INTERNAL_ID_COL, "playerName", "playerId", INTERNAL_POSITION_COL, "activo", "photo_url"]
            )
        # Keep playerId as string to preserve format
        df = pd.read_excel(
            path,
            converters={
                "playerId": lambda x: str(x).strip() if pd.notna(x) else None,
                INTERNAL_ID_COL: lambda x: str(x).strip() if pd.notna(x) else None,
                INTERNAL_POSITION_COL: lambda x: str(x).strip() if pd.notna(x) else None,
            },
        )
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    if INTERNAL_ID_COL not in df.columns:
        df[INTERNAL_ID_COL] = None
    if "playerName" not in df.columns:
        df["playerName"] = ""
    if "playerId" not in df.columns:
        df["playerId"] = None
    if INTERNAL_POSITION_COL not in df.columns:
        df[INTERNAL_POSITION_COL] = None
    if "activo" not in df.columns:
        df["activo"] = 1
    if "photo_url" not in df.columns:
        df["photo_url"] = None
    # Coerce types
    df[INTERNAL_ID_COL] = df[INTERNAL_ID_COL].astype("string").where(df[INTERNAL_ID_COL].notna(), None)
    df[INTERNAL_ID_COL] = df[INTERNAL_ID_COL].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df["playerName"] = df["playerName"].astype(str).str.strip()
    df["playerId"] = df["playerId"].astype("string").where(df["playerId"].notna(), None)
    df["playerId"] = df["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df["playerId"] = df["playerId"].apply(normalize_whoscored_player_id)
    df["playerId"] = df["playerId"].astype("string").where(df["playerId"].notna(), None)
    df[INTERNAL_POSITION_COL] = df[INTERNAL_POSITION_COL].astype("string").where(df[INTERNAL_POSITION_COL].notna(), None)
    df[INTERNAL_POSITION_COL] = df[INTERNAL_POSITION_COL].apply(_normalize_internal_position)
    df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    df["photo_url"] = df["photo_url"].astype("string").where(df["photo_url"].notna(), None)
    df["photo_url"] = df["photo_url"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = _ensure_internal_ids(df)
    return df[[INTERNAL_ID_COL, "playerName", "playerId", INTERNAL_POSITION_COL, "activo", "photo_url"]]


def save_players_df(df: pd.DataFrame, path: str):
    """Save players DataFrame to Google Sheets (fallback local Excel)."""
    df = df.copy()
    df = _ensure_internal_ids(df)
    # Ensure 'activo' column is int and only contains 0/1
    if "activo" in df.columns:
        df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    # Normalize order and whitespace
    if INTERNAL_ID_COL in df.columns:
        df[INTERNAL_ID_COL] = df[INTERNAL_ID_COL].astype("string").where(df[INTERNAL_ID_COL].notna(), None)
        df[INTERNAL_ID_COL] = df[INTERNAL_ID_COL].apply(lambda x: x.strip() if isinstance(x, str) else x)
    else:
        df[INTERNAL_ID_COL] = None
    df["playerName"] = df["playerName"].astype(str).str.strip()
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].astype("string")
        df["playerId"] = df["playerId"].where(df["playerId"].notna(), None)
        df["playerId"] = df["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df["playerId"] = df["playerId"].apply(normalize_whoscored_player_id)
        df["playerId"] = df["playerId"].astype("string").where(df["playerId"].notna(), None)
    else:
        df["playerId"] = None
    if INTERNAL_POSITION_COL in df.columns:
        df[INTERNAL_POSITION_COL] = df[INTERNAL_POSITION_COL].astype("string").where(df[INTERNAL_POSITION_COL].notna(), None)
        df[INTERNAL_POSITION_COL] = df[INTERNAL_POSITION_COL].apply(_normalize_internal_position)
    else:
        df[INTERNAL_POSITION_COL] = None
    if "photo_url" not in df.columns:
        df["photo_url"] = None
    df["photo_url"] = df["photo_url"].astype("string").where(df["photo_url"].notna(), None)
    df["photo_url"] = df["photo_url"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[[INTERNAL_ID_COL, "playerName", "playerId", INTERNAL_POSITION_COL, "activo", "photo_url"]]

    sheets_client = get_sheets_client()
    if sheets_client.is_configured():
        try:
            sheets_client.write_players_df(df)
            return
        except Exception as exc:
            st.warning(f"Could not save to Google Sheets (Players). Saving locally. ({exc})")

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        with open(path, "wb") as f:
            f.write(buffer.read())

def _safe_slug(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "player")[:max_len]

def _clean_str(value: object) -> str:
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

def _normalize_internal_position(value: object) -> Optional[str]:
    s = _clean_str(value)
    if not s or s == AUTO_POSITION_LABEL:
        return None
    return s

def _active_status_badge_html(is_active: bool) -> str:
    if is_active:
        bg = "#dcfce7"
        fg = "#166534"
        label = "Active"
    else:
        bg = "#fee2e2"
        fg = "#991b1b"
        label = "Inactive"
    return (
        f"<span style='display:inline-block;padding:0.15rem 0.5rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:700;font-size:0.78rem'>{label}</span>"
    )

def _player_card_title_html(player_name: str, position_label: str) -> str:
    name = html.escape(_clean_str(player_name))
    pos = html.escape(_clean_str(position_label) or "Unknown")
    return (
        "<div style='margin-top:0.35rem;margin-bottom:0.35rem;"
        "background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:0.5rem 0.65rem;'>"
        f"<div style='font-weight:800;font-size:0.95rem;color:#0f172a;line-height:1.2'>{name}</div>"
        f"<div style='font-weight:600;font-size:0.82rem;color:#475569;margin-top:0.15rem'>{pos}</div>"
        "</div>"
    )

def _player_photo_block_html(photo_src: Optional[str]) -> str:
    if photo_src:
        src = html.escape(photo_src, quote=True)
        return (
            "<div style='width:100%;height:320px;border-radius:12px;overflow:hidden;"
            "background:#e5e7eb;margin-bottom:0.35rem;'>"
            f"<img src='{src}' style='width:100%;height:100%;object-fit:cover;object-position:center top;display:block;'/>"
            "</div>"
        )
    return (
        "<div style='width:100%;height:320px;border-radius:12px;overflow:hidden;"
        "background:linear-gradient(180deg,#f1f5f9 0%,#e2e8f0 100%);margin-bottom:0.35rem;"
        "display:flex;align-items:center;justify-content:center;color:#64748b;font-weight:700;'>"
        "No photo"
        "</div>"
    )

def _normalize_internal_id(value: object) -> Optional[str]:
    s = _clean_str(value)
    if not s:
        return None
    candidate = s.upper().replace("_", "-")
    match = re.match(rf"^{INTERNAL_ID_PREFIX}-?(\d+)$", candidate)
    if not match:
        return s
    return f"{INTERNAL_ID_PREFIX}-{int(match.group(1)):0{INTERNAL_ID_PAD}d}"

def _parse_internal_seq(value: object) -> int:
    s = _clean_str(value).upper()
    match = re.match(rf"^{INTERNAL_ID_PREFIX}-(\d+)$", s)
    return int(match.group(1)) if match else 0

def _next_internal_id(df: pd.DataFrame) -> str:
    used = set()
    max_seq = 0
    if INTERNAL_ID_COL in df.columns:
        for raw in df[INTERNAL_ID_COL].tolist():
            norm = _normalize_internal_id(raw)
            if not norm:
                continue
            used.add(norm)
            max_seq = max(max_seq, _parse_internal_seq(norm))
    seq = max_seq + 1
    while True:
        candidate = f"{INTERNAL_ID_PREFIX}-{seq:0{INTERNAL_ID_PAD}d}"
        if candidate not in used:
            return candidate
        seq += 1

def _ensure_internal_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if INTERNAL_ID_COL not in out.columns:
        out[INTERNAL_ID_COL] = None

    normalized = [_normalize_internal_id(v) for v in out[INTERNAL_ID_COL].tolist()]
    out[INTERNAL_ID_COL] = normalized

    max_seq = 0
    for val in normalized:
        max_seq = max(max_seq, _parse_internal_seq(val))

    seen = set()
    seq = max_seq + 1

    for idx in out.index:
        val = _normalize_internal_id(out.at[idx, INTERNAL_ID_COL])
        if val and val not in seen:
            out.at[idx, INTERNAL_ID_COL] = val
            seen.add(val)
            continue

        while True:
            candidate = f"{INTERNAL_ID_PREFIX}-{seq:0{INTERNAL_ID_PAD}d}"
            seq += 1
            if candidate not in seen:
                out.at[idx, INTERNAL_ID_COL] = candidate
                seen.add(candidate)
                break

    return out

def _whoscored_player_url(player_id: object) -> Optional[str]:
    return whoscored_player_url(player_id)

def _photo_src_to_abs_path(photo_src: str) -> Path:
    p = Path(_clean_str(photo_src))
    if not p.is_absolute():
        p = Path(BASE_DIR) / p
    return p

@st.cache_data(show_spinner=False)
def _local_image_to_data_url(abs_path: str, mtime: float) -> Optional[str]:
    try:
        p = Path(abs_path)
        if not p.exists():
            return None
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "image/jpeg"
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _build_photo_preview(photo_src: Optional[str]) -> Optional[str]:
    s = _clean_str(photo_src)
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://") or s.startswith("data:"):
        return s
    abs_path = _photo_src_to_abs_path(s)
    if not abs_path.exists():
        return None
    return _local_image_to_data_url(str(abs_path), abs_path.stat().st_mtime)

def _save_uploaded_photo(
    uploaded_file,
    player_id: str,
    player_name: Optional[str] = None,
    internal_id: Optional[str] = None,
) -> str:
    """Save uploaded photo to data/player_photos and return a relative path to store in Excel."""
    os.makedirs(PLAYER_PHOTOS_DIR, exist_ok=True)

    pid = (player_id or "").strip()
    iid = (internal_id or "").strip()
    base = _safe_slug(pid) if pid else (_safe_slug(iid) if iid else _safe_slug(player_name or "player"))
    filename = f"{base}.jpg"
    rel_path = os.path.join("data", "player_photos", filename)
    abs_path = os.path.join(BASE_DIR, rel_path)

    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.split()[-1]
        background.paste(img.convert("RGB"), mask=alpha)
        img = background
    else:
        img = img.convert("RGB")

    # Square crop + resize for consistent UI and size
    try:
        resample = Image.Resampling.LANCZOS  # Pillow >= 9
    except Exception:
        resample = Image.LANCZOS
    img = ImageOps.fit(img, (512, 512), method=resample)

    img.save(abs_path, format="JPEG", quality=85, optimize=True, progressive=True)
    return rel_path

def _update_player_row(
    players_excel_df: pd.DataFrame,
    internal_id: str,
    player_id: str,
    player_name: str,
    updates: dict,
) -> pd.DataFrame:
    df = players_excel_df.copy()
    iid = _clean_str(internal_id)
    pid = _clean_str(player_id)
    pname = _clean_str(player_name)

    if INTERNAL_ID_COL not in df.columns:
        df[INTERNAL_ID_COL] = None
    if "playerId" not in df.columns:
        df["playerId"] = None
    if INTERNAL_POSITION_COL not in df.columns:
        df[INTERNAL_POSITION_COL] = None
    if "playerName" not in df.columns:
        df["playerName"] = ""

    # Prefer updating by internal_id, then by playerId; fallback to playerName.
    if iid:
        mask = df[INTERNAL_ID_COL].apply(_clean_str) == iid
    elif pid:
        mask = df["playerId"].apply(_clean_str) == pid
    else:
        mask = df["playerName"].apply(lambda x: _clean_str(x).lower()) == pname.lower()

    if not mask.any():
        raise ValueError("Player not found in the spreadsheet for update.")

    for key, value in updates.items():
        if key not in df.columns:
            df[key] = None
        df.loc[mask, key] = value

    return df

def _delete_player_row(
    players_excel_df: pd.DataFrame,
    internal_id: str,
    player_id: str,
    player_name: str,
) -> pd.DataFrame:
    df = players_excel_df.copy()
    iid = _clean_str(internal_id)
    pid = _clean_str(player_id)
    pname = _clean_str(player_name)

    if INTERNAL_ID_COL not in df.columns:
        df[INTERNAL_ID_COL] = None
    if "playerId" not in df.columns:
        df["playerId"] = None
    if "playerName" not in df.columns:
        df["playerName"] = ""

    if iid:
        mask = df[INTERNAL_ID_COL].apply(_clean_str) == iid
    elif pid:
        mask = df["playerId"].apply(_clean_str) == pid
    else:
        mask = df["playerName"].apply(lambda x: _clean_str(x).lower()) == pname.lower()

    if not mask.any():
        raise ValueError("Player not found in the spreadsheet for deletion.")

    return df.loc[~mask].reset_index(drop=True)

def _load_player_positions(engine, player_ids: list[str]) -> pd.DataFrame:
    """Return primary (most frequent) position code per playerId from player_data."""
    if engine is None or not player_ids:
        return pd.DataFrame(columns=["playerId", "position_code"])

    cleaned = [pid.strip() for pid in player_ids if pid and str(pid).strip() != ""]
    cleaned = list(dict.fromkeys(cleaned))
    if not cleaned:
        return pd.DataFrame(columns=["playerId", "position_code"])

    placeholders = ",".join(["%s"] * len(cleaned))
    q = f"""
    WITH pos_counts AS (
      SELECT pd.playerId, pd.position, COUNT(*) AS cnt
      FROM player_data pd
      WHERE pd.playerId IN ({placeholders})
        AND pd.position IS NOT NULL
        AND pd.position <> 'Sub'
      GROUP BY pd.playerId, pd.position
    ),
    ranked AS (
      SELECT playerId, position, cnt,
             ROW_NUMBER() OVER (PARTITION BY playerId ORDER BY cnt DESC) AS rn
      FROM pos_counts
    )
    SELECT playerId, position AS position_code
    FROM ranked
    WHERE rn = 1
    """
    df = pd.read_sql(q, con=engine, params=tuple(cleaned))
    df["playerId"] = df["playerId"].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    df["position_code"] = df["position_code"].astype(str).str.strip()
    return df

def _load_player_summaries(engine, player_ids: list[str]) -> pd.DataFrame:
    """Return per-player summary for UI: matches_played + last_team + last_match_date."""
    if engine is None or not player_ids:
        return pd.DataFrame(columns=["playerId", "matches_played", "last_team", "last_match_date"])

    # Clean ids
    cleaned = [pid.strip() for pid in player_ids if pid and str(pid).strip() != ""]
    cleaned = list(dict.fromkeys(cleaned))
    if not cleaned:
        return pd.DataFrame(columns=["playerId", "matches_played", "last_team", "last_match_date"])

    placeholders = ",".join(["%s"] * len(cleaned))

    # Matches played
    q_counts = f"""
    SELECT pd.playerId, COUNT(DISTINCT pd.matchId) AS matches_played
    FROM player_data pd
    WHERE pd.playerId IN ({placeholders})
    GROUP BY pd.playerId
    """
    counts = pd.read_sql(q_counts, con=engine, params=tuple(cleaned))
    counts["playerId"] = counts["playerId"].apply(lambda x: str(x).strip() if pd.notna(x) else None)

    # Last team based on latest match date (MySQL 8+ window functions)
    q_last = f"""
    WITH ranked AS (
      SELECT
        pd.playerId,
        pd.teamName,
        md.startDate,
        ROW_NUMBER() OVER (PARTITION BY pd.playerId ORDER BY md.startDate DESC) AS rn
      FROM player_data pd
      JOIN match_data md ON md.matchId = pd.matchId
      WHERE pd.playerId IN ({placeholders})
        AND md.startDate IS NOT NULL
    )
    SELECT
      playerId,
      teamName AS last_team,
      startDate AS last_match_date
    FROM ranked
    WHERE rn = 1
    """
    last = pd.read_sql(q_last, con=engine, params=tuple(cleaned))
    last["playerId"] = last["playerId"].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    last["last_team"] = last["last_team"].astype(str).str.strip()
    last["last_match_date"] = pd.to_datetime(last["last_match_date"], errors="coerce")

    out = counts.merge(last, on="playerId", how="left")
    out["matches_played"] = pd.to_numeric(out["matches_played"], errors="coerce").fillna(0).astype(int)
    return out

# --- UI ------------------------------------------------------
st.title("⚙️ Manage players list")

# Display logo on the sidebar for consistency
with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=120)
    except FileNotFoundError:
        pass

st.markdown(
    "Review and modify player *Active/Inactive* status. "
    "Each player has an **internal_id** auto-generated by the system. "
    "WhoScored `playerId` is optional so you can add players without WhoScored history."
)

players_df = load_players_df(PLAYERS_FILE_XLSX).copy()
# Sort alphabetically by playerName (case-insensitive)
if not players_df.empty and "playerName" in players_df.columns:
    players_df = (
        players_df
        .assign(_name_lower=players_df["playerName"].astype(str).str.lower())
        .sort_values(by="_name_lower", kind="stable")
        .drop(columns=["_name_lower"]) 
        .reset_index(drop=True)
    )

# --- DB players selector ------------------------------------------------
engine = connect_to_db()

with st.expander("➕ Add player", expanded=False):
    st.markdown(
        "Type a name to search suggested WhoScored IDs. "
        "You can also add a player without `playerId`; the system will assign `internal_id` automatically."
    )
    manual_name = st.text_input(
        "Player name",
        value="",
        key="manual_player_name",
        help="Buscamos coincidencias en `player_data` para sugerir posibles IDs de WhoScored."
    ).strip()
    manual_internal_position = st.selectbox(
        "Internal position (optional)",
        options=INTERNAL_POSITION_OPTIONS,
        index=0,
        key="manual_player_internal_position",
        help="If you choose 'Auto (WhoScored)', the inferred WhoScored position will be used when available.",
    )

    photo_file_input = st.file_uploader(
        "Photo (file, optional - recommended)",
        type=["jpg", "jpeg", "png", "webp"],
        key="manual_player_photo_file",
        help="It will be optimized and saved to data/player_photos/ and linked to the player."
    )
    photo_url_input = st.text_input(
        "Photo (URL, optional)",
        value="",
        key="manual_player_photo_url",
        help="Alternative if you do not want to upload a file: paste a public URL (jpg/png)."
    ).strip()

    def find_player_candidates_by_name(engine, name: str) -> pd.DataFrame:
        # Prefer exact case-insensitive match; if it returns nothing, try a LIKE.
        exact_q = """
        SELECT playerId, playerName, COUNT(*) AS cnt
        FROM player_data
        WHERE playerId IS NOT NULL
          AND playerName IS NOT NULL
          AND LOWER(playerName) = LOWER(%s)
        GROUP BY playerId, playerName
        ORDER BY cnt DESC
        LIMIT 10
        """
        df = pd.read_sql(exact_q, con=engine, params=(name,))
        if not df.empty:
            return df
        like_q = """
        SELECT playerId, playerName, COUNT(*) AS cnt
        FROM player_data
        WHERE playerId IS NOT NULL
          AND playerName IS NOT NULL
          AND LOWER(playerName) LIKE CONCAT('%%', LOWER(%s), '%%')
        GROUP BY playerId, playerName
        ORDER BY cnt DESC
        LIMIT 10
        """
        return pd.read_sql(like_q, con=engine, params=(name,))

    # Reset lookup results if the name changes (avoid mismatched candidates)
    if st.session_state.get("lookup_last_name") != manual_name:
        st.session_state["lookup_last_name"] = manual_name
        st.session_state.pop("lookup_candidates", None)
        st.session_state.pop("lookup_selected_idx", None)

    if st.button("Search DB", key="lookup_btn", disabled=(not manual_name)):
        if engine is None:
            st.error("No DB connection. Cannot search playerId.")
        else:
            candidates = find_player_candidates_by_name(engine, manual_name)
            if candidates.empty:
                st.info("No matches found in `player_data` for that name. You can add the player without `playerId`.")
            else:
                candidates = candidates.copy()
                candidates["playerId"] = candidates["playerId"].apply(lambda x: str(x).strip() if pd.notna(x) else None)
                candidates["playerName"] = candidates["playerName"].astype(str).str.strip()
                candidates["whoscored_url"] = candidates["playerId"].apply(_whoscored_player_url)
                st.session_state["lookup_candidates"] = candidates.to_dict(orient="records")
                st.session_state["lookup_selected_idx"] = 0

    lookup_candidates = st.session_state.get("lookup_candidates") or []
    if lookup_candidates:
        options = [
            f"{c.get('playerName')} (ID: {c.get('playerId')}) [matches: {int(c.get('cnt') or 0)}]"
            for c in lookup_candidates
        ]
        chosen_label = st.selectbox(
            "Found matches",
            options=options,
            index=int(st.session_state.get("lookup_selected_idx") or 0),
            key="lookup_candidate_selector",
            help="If there are multiple matches, choose the correct one before adding."
        )
        st.session_state["lookup_selected_idx"] = options.index(chosen_label)

        candidates_df = pd.DataFrame(lookup_candidates)
        if not candidates_df.empty:
            st.dataframe(
                candidates_df[["playerName", "playerId", "cnt", "whoscored_url"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "playerName": st.column_config.TextColumn("Player"),
                    "playerId": st.column_config.TextColumn("WhoScored ID"),
                    "cnt": st.column_config.NumberColumn("Detected matches"),
                    "whoscored_url": st.column_config.LinkColumn("WhoScored link"),
                },
            )
            st.markdown("**Suggested list with WhoScored link**")
            for c in lookup_candidates:
                pid = _clean_str(c.get("playerId"))
                pname = _clean_str(c.get("playerName"))
                if pid:
                    st.markdown(f"- {pname} (`{pid}`): [WhoScored]({_whoscored_player_url(pid)})")

        if st.button("Add selected WhoScored match", key="lookup_add_btn"):
            chosen = lookup_candidates[int(st.session_state.get("lookup_selected_idx") or 0)]
            chosen_name = (chosen.get("playerName") or "").strip()
            chosen_id = (chosen.get("playerId") or "").strip()

            if not chosen_name or not chosen_id:
                st.error("The selected match has no valid playerId.")
                st.stop()

            # Avoid duplicates by playerId (stronger than by name)
            existing_ids = set(
                players_df["playerId"].dropna().astype(str).str.strip().tolist()
            ) if "playerId" in players_df.columns else set()
            if chosen_id in existing_ids:
                st.warning("That playerId already exists in the spreadsheet. Duplicate was not added.")
                st.stop()

            internal_id = _next_internal_id(players_df)
            photo_value = photo_url_input or None
            if photo_file_input is not None:
                try:
                    photo_value = _save_uploaded_photo(
                        photo_file_input,
                        player_id=chosen_id,
                        player_name=chosen_name,
                        internal_id=internal_id,
                    )
                except Exception as e:
                    st.warning(f"Could not process uploaded photo: {e}")
            new_row = pd.DataFrame([{
                INTERNAL_ID_COL: internal_id,
                "playerName": chosen_name,
                "playerId": chosen_id,
                INTERNAL_POSITION_COL: _normalize_internal_position(manual_internal_position),
                "activo": 1,
                "photo_url": photo_value,
            }])
            players_df = pd.concat([players_df, new_row], ignore_index=True)
            save_players_df(players_df, PLAYERS_FILE_XLSX)
            load_players_df.clear()
            st.session_state.pop("lookup_candidates", None)
            st.session_state.pop("lookup_selected_idx", None)
            st.success(
                f"Player '{chosen_name}' added successfully "
                f"(internal_id: {internal_id}, playerId: {chosen_id})."
            )
            st.rerun()

    if st.button("Add without WhoScored ID", key="add_without_whoscored_btn", disabled=(not manual_name)):
        internal_id = _next_internal_id(players_df)
        photo_value = photo_url_input or None
        if photo_file_input is not None:
            try:
                photo_value = _save_uploaded_photo(
                    photo_file_input,
                    player_id="",
                    player_name=manual_name,
                    internal_id=internal_id,
                )
            except Exception as e:
                st.warning(f"Could not process uploaded photo: {e}")
        new_row = pd.DataFrame([{
            INTERNAL_ID_COL: internal_id,
            "playerName": manual_name,
            "playerId": None,
            INTERNAL_POSITION_COL: _normalize_internal_position(manual_internal_position),
            "activo": 1,
            "photo_url": photo_value,
        }])
        players_df = pd.concat([players_df, new_row], ignore_index=True)
        save_players_df(players_df, PLAYERS_FILE_XLSX)
        load_players_df.clear()
        st.session_state.pop("lookup_candidates", None)
        st.session_state.pop("lookup_selected_idx", None)
        st.success(f"Player '{manual_name}' added without WhoScored ID (internal_id: {internal_id}).")
        st.rerun()

if players_df.empty:
    st.info("No players found. You can load the corresponding file from the main page.")

# ---- Filtro de Active/Inactive ----
status_filter = st.selectbox(
    "Filter by status",
    options=["All", "Active", "Inactive"],
    index=0,
    help="Filter the table by active status"
)
st.markdown(
    "<span style='color:#166534;font-weight:700'>Active</span> | "
    "<span style='color:#991b1b;font-weight:700'>Inactive</span>",
    unsafe_allow_html=True,
)

if status_filter == "Active":
    filtered_df = players_df[players_df["activo"] == 1].copy()
elif status_filter == "Inactive":
    filtered_df = players_df[players_df["activo"] == 0].copy()
else:
    filtered_df = players_df.copy()

# ---- Enrich with DB summary (matches + last team) for display ----
summary_df = pd.DataFrame()
position_df = pd.DataFrame()
try:
    player_ids = (
        filtered_df.get("playerId", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    summary_df = _load_player_summaries(engine, player_ids)
    position_df = _load_player_positions(engine, player_ids)
except Exception:
    summary_df = pd.DataFrame()
    position_df = pd.DataFrame()

display_df = filtered_df.copy()
if not summary_df.empty and "playerId" in display_df.columns:
    display_df["playerId"] = display_df["playerId"].astype("string").where(display_df["playerId"].notna(), None)
    display_df["playerId"] = display_df["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    display_df = display_df.merge(summary_df, on="playerId", how="left")
    if not position_df.empty:
        display_df = display_df.merge(position_df, on="playerId", how="left")
else:
    display_df["matches_played"] = None
    display_df["last_team"] = None
    display_df["last_match_date"] = None
    display_df["position_code"] = None

def _position_label_from_code(code: object) -> str:
    s = str(code).strip() if code is not None else ""
    if not s:
        return "Unknown"
    s_norm = s.upper() if (" " not in s and len(s) <= 4) else s
    return POSITION_CODE_TO_LABEL.get(s_norm, s)

if INTERNAL_POSITION_COL not in display_df.columns:
    display_df[INTERNAL_POSITION_COL] = None
display_df[INTERNAL_POSITION_COL] = display_df[INTERNAL_POSITION_COL].apply(_normalize_internal_position)
display_df["position_label_ws"] = display_df.get("position_code", pd.Series(dtype="object")).apply(_position_label_from_code)
display_df["position_label"] = display_df[INTERNAL_POSITION_COL].where(
    display_df[INTERNAL_POSITION_COL].apply(lambda x: _clean_str(x) != ""),
    display_df["position_label_ws"],
)
display_df["position_label"] = display_df["position_label"].fillna("Unknown")
display_df["photo_preview"] = display_df.get("photo_url", pd.Series(dtype="object")).apply(_build_photo_preview)
display_df["whoscored_link"] = display_df.get("playerId", pd.Series(dtype="object")).apply(_whoscored_player_url)

# Sort UI by position, then name (does not affect Excel persistence order)
pos_rank = {p: i for i, p in enumerate(POSITION_ORDER)}
display_df["_pos_rank"] = display_df["position_label"].map(pos_rank).fillna(len(POSITION_ORDER)).astype(int)
display_df["_name_sort"] = display_df["playerName"].astype(str).str.strip().str.lower()
display_df = display_df.sort_values(by=["_pos_rank", "_name_sort"], kind="stable").drop(columns=["_pos_rank", "_name_sort"])

st.markdown("### Quick view")
cols = st.columns(4)
active_count = int((display_df["activo"] == 1).sum()) if "activo" in display_df.columns else 0
inactive_count = int((display_df["activo"] == 0).sum()) if "activo" in display_df.columns else 0
cols[0].metric("Players (view)", int(len(display_df)))
cols[1].metric("Active", active_count)
cols[2].metric("Inactive", inactive_count)
has_photo_count = 0
if "photo_url" in display_df.columns:
    has_photo_count = int(
        display_df["photo_url"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
cols[3].metric("With photo", has_photo_count)

with st.expander("🖼️ Player cards", expanded=True):
    # Group by position label
    display_df_cards = display_df.copy()
    display_df_cards["playerName"] = display_df_cards["playerName"].astype(str).str.strip()
    player_name_options = sorted(
        [
            n
            for n in display_df_cards["playerName"].dropna().astype(str).str.strip().unique().tolist()
            if n
        ],
        key=lambda x: x.lower(),
    )
    selected_name = st.selectbox(
        "Search by player name",
        options=[None, *player_name_options],
        index=0,
        key="cards_search_name_select",
        placeholder="Type to see suggestions...",
        format_func=lambda x: "All players" if x is None else x,
    )
    if selected_name:
        display_df_cards = display_df_cards[
            display_df_cards["playerName"].astype(str).str.strip() == str(selected_name).strip()
        ].copy()

    broad_position_map = {
        "Goalkeeper": "Goalkeepers",
        "Center Back": "Defenders",
        "Left Back": "Defenders",
        "Right Back": "Defenders",
        "Defensive Midfielder": "Midfielders",
        "Midfielder": "Midfielders",
        "Attacking Midfielder": "Midfielders",
        "Left Winger": "Midfielders",
        "Right Winger": "Midfielders",
        "Striker": "Forwards",
    }
    group_order = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]
    display_df_cards["line_group"] = display_df_cards["position_label"].map(broad_position_map).fillna("Midfielders")
    display_df_cards = display_df_cards.sort_values(by=["line_group", "position_label", "playerName"], kind="stable")

    found_any_group = False
    for line_group in group_order:
        group = display_df_cards[display_df_cards["line_group"] == line_group].copy()
        if group.empty:
            continue
        found_any_group = True
        group_container = st.expander(f"{line_group} ({len(group)})", expanded=True)
        card_cols = group_container.columns(3)
        for i, row in enumerate(group.fillna("").to_dict(orient="records")):
            with card_cols[i % 3]:
                with st.container(border=True):
                    iid = _clean_str(row.get(INTERNAL_ID_COL))
                    pid = _clean_str(row.get("playerId"))
                    pname = _clean_str(row.get("playerName"))
                    player_key = iid if iid else (pid if pid else f"name:{pname.lower()}:{i}")
                    widget_suffix = _safe_slug(f"{player_key}_{i}")
                    current_active = int(row.get("activo") or 0) == 1

                    photo_preview = _clean_str(row.get("photo_preview"))
                    if not photo_preview:
                        photo_preview = _build_photo_preview(row.get("photo_url")) or ""
                    st.markdown(_player_photo_block_html(photo_preview if photo_preview else None), unsafe_allow_html=True)

                    position_label = _clean_str(row.get("position_label")) or "Unknown"
                    st.markdown(_player_card_title_html(pname, position_label), unsafe_allow_html=True)
                    st.markdown(_active_status_badge_html(current_active), unsafe_allow_html=True)

                    delete_armed_key = f"delete_armed_{widget_suffix}"
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("ℹ️", key=f"open_{widget_suffix}", help="More details"):
                            st.session_state["manage_players_selected"] = player_key
                            st.rerun()
                    with col_b:
                        quick_toggle_label = "🔴" if current_active else "🟢"
                        quick_toggle_help = "Deactivate player" if current_active else "Activate player"
                        if st.button(
                            quick_toggle_label,
                            key=f"toggle_active_{widget_suffix}",
                            help=quick_toggle_help,
                        ):
                            try:
                                players_df = _update_player_row(
                                    players_df,
                                    internal_id=iid,
                                    player_id=pid,
                                    player_name=pname,
                                    updates={"activo": 0 if current_active else 1},
                                )
                                save_players_df(players_df, PLAYERS_FILE_XLSX)
                                load_players_df.clear()
                                st.success(f"Player {'deactivated' if current_active else 'activated'}.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not update status: {e}")
                    with col_c:
                        if st.button("🗑️", key=f"ask_delete_{widget_suffix}", type="secondary", help="Delete player"):
                            st.session_state[delete_armed_key] = True

                    if st.session_state.get(delete_armed_key):
                        st.warning("Confirm player deletion.")
                        del_yes, del_no = st.columns(2)
                        with del_yes:
                            if st.button("Confirm", key=f"confirm_delete_{widget_suffix}", type="primary"):
                                try:
                                    players_df = _delete_player_row(
                                        players_df,
                                        internal_id=iid,
                                        player_id=pid,
                                        player_name=pname,
                                    )
                                    save_players_df(players_df, PLAYERS_FILE_XLSX)
                                    load_players_df.clear()
                                    st.session_state.pop(delete_armed_key, None)
                                    if st.session_state.get("manage_players_selected") == player_key:
                                        st.session_state.pop("manage_players_selected", None)
                                    st.success(f"Player '{pname}' deleted successfully.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Could not delete player: {e}")
                        with del_no:
                            if st.button("Cancel", key=f"cancel_delete_{widget_suffix}"):
                                st.session_state.pop(delete_armed_key, None)
                                st.rerun()

                    if st.session_state.get("manage_players_selected") == player_key:
                        st.divider()
                        st.markdown("**Editable details**")
                        st.markdown(_active_status_badge_html(current_active), unsafe_allow_html=True)
                        st.caption(f"Internal ID: {iid if iid else '—'}")
                        st.caption(f"Current WhoScored ID: {pid if pid else '—'}")
                        if pid:
                            st.markdown(f"[WhoScored]({_whoscored_player_url(pid)})")
                        mp = row.get("matches_played", "")
                        lt = row.get("last_team", "")
                        lmd = row.get("last_match_date", "")
                        st.caption(f"Matches: {mp if mp != '' else '—'}")
                        st.caption(f"Last team: {lt if lt != '' else '—'}")
                        st.caption(f"Last match: {str(lmd)[:10] if lmd else '—'}")

                        current_internal_position = _normalize_internal_position(row.get(INTERNAL_POSITION_COL))
                        pos_options = INTERNAL_POSITION_OPTIONS.copy()
                        if current_internal_position and current_internal_position not in pos_options:
                            pos_options.append(current_internal_position)
                        current_photo_src = _clean_str(row.get("photo_url"))

                        with st.form(key=f"edit_form_{widget_suffix}"):
                            new_active = st.checkbox("Active", value=current_active)
                            new_internal_position = st.selectbox(
                                "Internal position",
                                options=pos_options,
                                index=pos_options.index(current_internal_position or AUTO_POSITION_LABEL),
                                key=f"card_internal_position_{widget_suffix}",
                                help="This position takes priority over the position detected from WhoScored.",
                            )
                            new_player_id_raw = st.text_input(
                                "WhoScored playerId (optional)",
                                value=pid,
                                key=f"card_player_id_{widget_suffix}",
                                help="If left empty, the player will have no WhoScored ID."
                            ).strip()
                            uploaded = st.file_uploader(
                                "Upload new photo",
                                type=["jpg", "jpeg", "png", "webp"],
                                key=f"card_upload_{widget_suffix}",
                                help="Se guarda optimizada (512x512 JPEG) en data/player_photos/."
                            )
                            new_photo_url = st.text_input(
                                "Photo (URL or local path)",
                                value="",
                                key=f"card_photo_src_{widget_suffix}",
                                help="If you upload a file, this is ignored. Example: https://... or data/player_photos/<id>.jpg"
                            ).strip()

                            submitted = st.form_submit_button("Save changes")

                        col_close, col_rm, col_sp = st.columns(3)
                        with col_close:
                            if st.button("Close", key=f"close_{widget_suffix}"):
                                st.session_state.pop("manage_players_selected", None)
                                st.rerun()
                        with col_rm:
                            if st.button("Remove photo", key=f"remove_photo_{widget_suffix}", disabled=(not current_photo_src)):
                                try:
                                    players_df = _update_player_row(
                                        players_df,
                                        internal_id=iid,
                                        player_id=pid,
                                        player_name=pname,
                                        updates={"photo_url": None},
                                    )
                                    save_players_df(players_df, PLAYERS_FILE_XLSX)
                                    load_players_df.clear()
                                    st.success("Photo removed (from spreadsheet).")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Could not remove photo: {e}")
                        with col_sp:
                            if submitted:
                                try:
                                    new_player_id = normalize_whoscored_player_id(new_player_id_raw)
                                    existing_ids = (
                                        players_df.loc[
                                            players_df[INTERNAL_ID_COL].apply(_clean_str) != iid,
                                            "playerId",
                                        ]
                                        .apply(normalize_whoscored_player_id)
                                        .dropna()
                                        .astype(str)
                                        .str.strip()
                                        .tolist()
                                    )
                                    if new_player_id and new_player_id in set(existing_ids):
                                        st.error(f"The playerId {new_player_id} already exists for another player.")
                                        st.stop()

                                    photo_value = current_photo_src or None
                                    if uploaded is not None:
                                        photo_value = _save_uploaded_photo(
                                            uploaded,
                                            player_id=new_player_id or pid,
                                            player_name=pname,
                                            internal_id=iid,
                                        )
                                    elif new_photo_url:
                                        photo_value = new_photo_url

                                    players_df = _update_player_row(
                                        players_df,
                                        internal_id=iid,
                                        player_id=pid,
                                        player_name=pname,
                                        updates={
                                            "activo": 1 if new_active else 0,
                                            "playerId": new_player_id,
                                            INTERNAL_POSITION_COL: _normalize_internal_position(new_internal_position),
                                            "photo_url": photo_value,
                                        },
                                    )
                                    save_players_df(players_df, PLAYERS_FILE_XLSX)
                                    load_players_df.clear()
                                    st.success("Changes saved.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error saving changes: {e}")

    if not found_any_group:
        st.info("No players found for this search.")

with st.expander("📋 Advanced view (table)", expanded=False):
    editor_df = display_df.copy()
    editor_df[INTERNAL_POSITION_COL] = editor_df.get(INTERNAL_POSITION_COL, pd.Series(dtype="object")).apply(
        lambda x: _normalize_internal_position(x) or AUTO_POSITION_LABEL
    )
    editor_df["activo"] = pd.to_numeric(editor_df.get("activo", 0), errors="coerce").fillna(0).astype(int).eq(1)

    edited_df = st.data_editor(
        editor_df[
            [
                INTERNAL_ID_COL,
                "playerName",
                "playerId",
                INTERNAL_POSITION_COL,
                "whoscored_link",
                "position_label",
                "activo",
                "photo_preview",
                "photo_url",
                "matches_played",
                "last_team",
                "last_match_date",
            ]
        ],
        num_rows="dynamic",
        column_config={
            INTERNAL_ID_COL: st.column_config.TextColumn(
                "Internal ID",
                required=True,
                help="Internal identifier auto-generated by the system."
            ),
            "playerName": st.column_config.TextColumn(
                "Player name",
                required=True,
                help="Full name as it appears in the database"
            ),
            "playerId": st.column_config.TextColumn(
                "playerId",
                required=False,
                help="WhoScored identifier. Optional for players without WhoScored history."
            ),
            INTERNAL_POSITION_COL: st.column_config.SelectboxColumn(
                "Internal position",
                options=INTERNAL_POSITION_OPTIONS,
                required=True,
                help="If you choose 'Auto (WhoScored)', the detected position is used automatically.",
            ),
            "whoscored_link": st.column_config.LinkColumn("WhoScored"),
            "position_label": st.column_config.TextColumn("Position", disabled=True),
            "photo_preview": st.column_config.ImageColumn("Photo"),
            "photo_url": st.column_config.TextColumn(
                "Photo (src)",
                required=False,
                help="Public URL or local path (for example: data/player_photos/123.jpg)."
            ),
            "activo": st.column_config.CheckboxColumn(
                "Active",
                help="Activate or deactivate the player with one click.",
                width="small",
            ),
            "matches_played": st.column_config.NumberColumn("Matches", disabled=True),
            "last_team": st.column_config.TextColumn("Last team", disabled=True),
            "last_match_date": st.column_config.DatetimeColumn("Last match", disabled=True),
        },
        disabled=[INTERNAL_ID_COL, "whoscored_link", "position_label", "photo_preview", "matches_played", "last_team", "last_match_date"],
        key="players_table_editor"
    )

    if st.button("Save changes (table)", type="primary"):
        try:
            df_to_save = players_df.copy()
            df_to_save = _ensure_internal_ids(df_to_save)
            if "photo_url" not in df_to_save.columns:
                df_to_save["photo_url"] = None
            edits = edited_df.copy()
            if INTERNAL_ID_COL not in edits.columns:
                st.error("Internal ID column not found in table changes.")
                st.stop()

            df_to_save[INTERNAL_ID_COL] = df_to_save[INTERNAL_ID_COL].apply(_clean_str)
            edits[INTERNAL_ID_COL] = edits[INTERNAL_ID_COL].apply(_clean_str)

            df_to_save["internal_key"] = df_to_save[INTERNAL_ID_COL]
            edits["internal_key"] = edits[INTERNAL_ID_COL]

            edit_cols = ["playerName", "playerId", INTERNAL_POSITION_COL, "activo"]
            if "photo_url" in edits.columns:
                edit_cols.append("photo_url")
            edits_lookup = edits[edits["internal_key"] != ""].drop_duplicates(subset=["internal_key"], keep="last")
            update_map = edits_lookup.set_index("internal_key")[edit_cols].to_dict(orient="index")

            def apply_update(row):
                key = row["internal_key"]
                if key in update_map:
                    upd = update_map[key]
                    row["playerName"] = upd.get("playerName", row["playerName"])
                    row["playerId"] = upd.get("playerId", row.get("playerId"))
                    row[INTERNAL_POSITION_COL] = upd.get(INTERNAL_POSITION_COL, row.get(INTERNAL_POSITION_COL))
                    row["activo"] = upd.get("activo", row.get("activo", 1))
                    if "photo_url" in upd:
                        row["photo_url"] = upd.get("photo_url", row.get("photo_url"))
                return row

            df_to_save = df_to_save.apply(apply_update, axis=1)
            df_to_save = df_to_save.drop(columns=["internal_key"], errors='ignore')

            # Normalize whitespace and types
            df_to_save["playerName"] = df_to_save["playerName"].astype(str).str.strip()
            df_to_save["playerId"] = df_to_save["playerId"].astype("string").where(df_to_save["playerId"].notna(), None)
            df_to_save["playerId"] = df_to_save["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df_to_save["playerId"] = df_to_save["playerId"].apply(normalize_whoscored_player_id)
            df_to_save["playerId"] = df_to_save["playerId"].astype("string").where(df_to_save["playerId"].notna(), None)
            if INTERNAL_POSITION_COL not in df_to_save.columns:
                df_to_save[INTERNAL_POSITION_COL] = None
            df_to_save[INTERNAL_POSITION_COL] = df_to_save[INTERNAL_POSITION_COL].apply(_normalize_internal_position)
            df_to_save["activo"] = pd.to_numeric(df_to_save["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
            if "photo_url" in df_to_save.columns:
                df_to_save["photo_url"] = df_to_save["photo_url"].astype("string").where(df_to_save["photo_url"].notna(), None)
                df_to_save["photo_url"] = df_to_save["photo_url"].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df_to_save = _ensure_internal_ids(df_to_save)

            # Validation rules
            errors = []
            # 1) Names required
            if (df_to_save["playerName"].str.len() == 0).any():
                errors.append("There are players with empty names.")
            # 2) playerId uniqueness among non-null IDs
            ids = df_to_save["playerId"].dropna().astype(str).str.strip()
            ids = ids[ids != ""]
            if ids.duplicated().any():
                dupes = ids[ids.duplicated()].unique().tolist()
                errors.append(f"Duplicate playerId values: {', '.join(dupes)}")
            # 3) internal_id must exist and be unique
            internal_ids = df_to_save[INTERNAL_ID_COL].apply(_clean_str)
            if (internal_ids == "").any():
                errors.append("There are players without Internal ID.")
            if internal_ids.duplicated().any():
                dupes = internal_ids[internal_ids.duplicated()].unique().tolist()
                errors.append(f"Duplicate internal_id values: {', '.join(dupes)}")

            if errors:
                for e in errors:
                    st.error(f"❌ {e}")
                st.stop()

            save_players_df(df_to_save, PLAYERS_FILE_XLSX)
            st.success("✅ Changes saved successfully!")
            # clear cache to reflect new data if user returns later
            load_players_df.clear()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving changes: {e}")
