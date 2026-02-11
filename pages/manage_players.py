import os
import re
import base64
import mimetypes
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
from db_utils import connect_to_db
from PIL import Image, ImageOps
from typing import Optional
# --- Config --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "img")
LOGO_PATH = os.path.join(IMG_DIR, "watford_logo.png")
PLAYER_PHOTOS_DIR = os.path.join(BASE_DIR, "data", "player_photos")

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

st.set_page_config(
    page_title="Gestionar lista de jugadores",
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

@st.cache_data(show_spinner=False)
def load_players_df(path: str) -> pd.DataFrame:
    """Load players Excel into a DataFrame. If the file doesn't exist create an empty df.
    Ensures the columns: playerName (str), playerId (string), activo (int), photo_url (string).
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["playerName", "playerId", "activo", "photo_url"])
    # Keep playerId as string to preserve format
    df = pd.read_excel(path, converters={"playerId": lambda x: str(x).strip() if pd.notna(x) else None})
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    if "playerName" not in df.columns:
        df["playerName"] = ""
    if "playerId" not in df.columns:
        df["playerId"] = None
    if "activo" not in df.columns:
        df["activo"] = 1
    if "photo_url" not in df.columns:
        df["photo_url"] = None
    # Coerce types
    df["playerName"] = df["playerName"].astype(str).str.strip()
    df["playerId"] = df["playerId"].astype("string").where(df["playerId"].notna(), None)
    df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    df["photo_url"] = df["photo_url"].astype("string").where(df["photo_url"].notna(), None)
    df["photo_url"] = df["photo_url"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df[["playerName", "playerId", "activo", "photo_url"]]


def save_players_df(df: pd.DataFrame, path: str):
    """Save dataframe back to Excel safely."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure 'activo' column is int and only contains 0/1
    if "activo" in df.columns:
        df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    # Normalize order and whitespace
    df["playerName"] = df["playerName"].astype(str).str.strip()
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].astype("string")
        df["playerId"] = df["playerId"].where(df["playerId"].notna(), None)
        df["playerId"] = df["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if "photo_url" not in df.columns:
        df["photo_url"] = None
    df["photo_url"] = df["photo_url"].astype("string").where(df["photo_url"].notna(), None)
    df["photo_url"] = df["photo_url"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[["playerName", "playerId", "activo", "photo_url"]]
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

def _save_uploaded_photo(uploaded_file, player_id: str, player_name: Optional[str] = None) -> str:
    """Save uploaded photo to data/player_photos and return a relative path to store in Excel."""
    os.makedirs(PLAYER_PHOTOS_DIR, exist_ok=True)

    pid = (player_id or "").strip()
    base = _safe_slug(pid) if pid else _safe_slug(player_name or "player")
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

def _update_player_row(players_excel_df: pd.DataFrame, player_id: str, player_name: str, updates: dict) -> pd.DataFrame:
    df = players_excel_df.copy()
    pid = _clean_str(player_id)
    pname = _clean_str(player_name)

    if "playerId" not in df.columns:
        df["playerId"] = None
    if "playerName" not in df.columns:
        df["playerName"] = ""

    # Prefer updating by playerId; fallback to playerName.
    if pid:
        mask = df["playerId"].astype(str).str.strip() == pid
    else:
        mask = df["playerName"].astype(str).str.strip().str.lower() == pname.lower()

    if not mask.any():
        raise ValueError("No se encontró el jugador en el Excel para actualizar.")

    for key, value in updates.items():
        if key not in df.columns:
            df[key] = None
        df.loc[mask, key] = value

    return df

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
st.title("⚙️ Gestionar lista de jugadores")

# Display logo on the sidebar for consistency
with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=120)
    except FileNotFoundError:
        pass

st.markdown("Revise y modifique el estado *Activo/Inactivo* de los jugadores. Use el menú de opciones en la columna **activo** para cambiar 1=Activo, 0=Inactivo. Haga clic en **Guardar cambios** para persistir los datos.")

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

with st.expander("➕ Añadir jugador desde base de datos", expanded=False):
    st.markdown("Añade un jugador escribiendo su nombre y buscando su `playerId` en `player_data`.")
    manual_name = st.text_input(
        "Nombre del jugador (tal cual aparece en la BD)",
        value="",
        key="manual_player_name",
        help="Buscamos coincidencias en la tabla player_data para rellenar playerId automáticamente."
    ).strip()

    photo_file_input = st.file_uploader(
        "Foto (archivo, opcional — recomendado)",
        type=["jpg", "jpeg", "png", "webp"],
        key="manual_player_photo_file",
        help="Se guardará optimizada en data/player_photos/ y se vinculará al jugador."
    )
    photo_url_input = st.text_input(
        "Foto (URL, opcional)",
        value="",
        key="manual_player_photo_url",
        help="Alternativa si no quieres subir un archivo: pega una URL pública (jpg/png)."
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

    if st.button("Buscar en BD", key="lookup_btn", disabled=(not manual_name)):
        if engine is None:
            st.error("No hay conexión a BD. No se puede buscar el playerId.")
        else:
            candidates = find_player_candidates_by_name(engine, manual_name)
            if candidates.empty:
                st.error("No se encontraron coincidencias en `player_data` para ese nombre.")
            else:
                candidates = candidates.copy()
                candidates["playerId"] = candidates["playerId"].apply(lambda x: str(x).strip() if pd.notna(x) else None)
                candidates["playerName"] = candidates["playerName"].astype(str).str.strip()
                st.session_state["lookup_candidates"] = candidates.to_dict(orient="records")
                st.session_state["lookup_selected_idx"] = 0

    lookup_candidates = st.session_state.get("lookup_candidates") or []
    if lookup_candidates:
        options = [
            f"{c.get('playerName')} (ID: {c.get('playerId')}) [matches: {int(c.get('cnt') or 0)}]"
            for c in lookup_candidates
        ]
        chosen_label = st.selectbox(
            "Coincidencias encontradas",
            options=options,
            index=int(st.session_state.get("lookup_selected_idx") or 0),
            key="lookup_candidate_selector",
            help="Si hay varias, elige la correcta antes de añadir."
        )
        st.session_state["lookup_selected_idx"] = options.index(chosen_label)

        if st.button("Añadir seleccionado", key="lookup_add_btn"):
            chosen = lookup_candidates[int(st.session_state.get("lookup_selected_idx") or 0)]
            chosen_name = (chosen.get("playerName") or "").strip()
            chosen_id = (chosen.get("playerId") or "").strip()

            if not chosen_name or not chosen_id:
                st.error("La coincidencia seleccionada no tiene playerId válido.")
                st.stop()

            # Avoid duplicates by playerId (stronger than by name)
            existing_ids = set(
                players_df["playerId"].dropna().astype(str).str.strip().tolist()
            ) if "playerId" in players_df.columns else set()
            if chosen_id in existing_ids:
                st.warning("Ese playerId ya existe en el Excel. No se añadió duplicado.")
                st.stop()

            photo_value = photo_url_input or None
            if photo_file_input is not None:
                try:
                    photo_value = _save_uploaded_photo(photo_file_input, player_id=chosen_id, player_name=chosen_name)
                except Exception as e:
                    st.warning(f"No se pudo procesar la foto subida: {e}")
            new_row = pd.DataFrame([{
                "playerName": chosen_name,
                "playerId": chosen_id,
                "activo": 1,
                "photo_url": photo_value,
            }])
            players_df = pd.concat([players_df, new_row], ignore_index=True)
            save_players_df(players_df, PLAYERS_FILE_XLSX)
            load_players_df.clear()
            st.session_state.pop("lookup_candidates", None)
            st.session_state.pop("lookup_selected_idx", None)
            st.success(f"Jugador '{chosen_name}' añadido correctamente con ID {chosen_id}.")
            st.rerun()

if players_df.empty:
    st.info("No se encontraron jugadores. Puede cargar el archivo correspondiente desde la página principal.")

# ---- Filtro de Activo/Inactivo ----
status_filter = st.selectbox(
    "Filtrar por estado",
    options=["Todos", "Activos", "Inactivos"],
    index=0,
    help="Filtra la tabla por el estado de activo"
)

if status_filter == "Activos":
    filtered_df = players_df[players_df["activo"] == 1].copy()
elif status_filter == "Inactivos":
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

display_df["position_label"] = display_df.get("position_code", pd.Series(dtype="object")).apply(_position_label_from_code)
display_df["photo_preview"] = display_df.get("photo_url", pd.Series(dtype="object")).apply(_build_photo_preview)

# Sort UI by position, then name (does not affect Excel persistence order)
pos_rank = {p: i for i, p in enumerate(POSITION_ORDER)}
display_df["_pos_rank"] = display_df["position_label"].map(pos_rank).fillna(len(POSITION_ORDER)).astype(int)
display_df["_name_sort"] = display_df["playerName"].astype(str).str.strip().str.lower()
display_df = display_df.sort_values(by=["_pos_rank", "_name_sort"], kind="stable").drop(columns=["_pos_rank", "_name_sort"])

st.markdown("### Vista rápida")
cols = st.columns(4)
active_count = int((display_df["activo"] == 1).sum()) if "activo" in display_df.columns else 0
inactive_count = int((display_df["activo"] == 0).sum()) if "activo" in display_df.columns else 0
cols[0].metric("Jugadores (vista)", int(len(display_df)))
cols[1].metric("Activos", active_count)
cols[2].metric("Inactivos", inactive_count)
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
cols[3].metric("Con foto", has_photo_count)

with st.expander("🖼️ Tarjetas de jugadores", expanded=True):
    # Group by position label
    display_df_cards = display_df.copy()
    display_df_cards["playerName"] = display_df_cards["playerName"].astype(str).str.strip()
    display_df_cards = display_df_cards.sort_values(by=["position_label", "playerName"], kind="stable")

    positions = [p for p in POSITION_ORDER if p in display_df_cards["position_label"].unique().tolist()]
    extra = [p for p in display_df_cards["position_label"].unique().tolist() if p not in POSITION_ORDER]
    positions = positions + sorted(extra)

    for pos in positions:
        group = display_df_cards[display_df_cards["position_label"] == pos].copy()
        if group.empty:
            continue
        st.subheader(f"{pos} ({len(group)})")
        card_cols = st.columns(3)
        for i, row in enumerate(group.fillna("").to_dict(orient="records")):
            with card_cols[i % 3]:
                with st.container(border=True):
                    pid = _clean_str(row.get("playerId"))
                    pname = _clean_str(row.get("playerName"))
                    player_key = pid if pid else f"name:{pname.lower()}"
                    widget_suffix = _safe_slug(f"{player_key}_{i}")

                    photo_src = _clean_str(row.get("photo_url"))
                    if photo_src:
                        if photo_src.startswith(("http://", "https://", "data:")):
                            st.image(photo_src, use_container_width=True)
                        else:
                            abs_p = _photo_src_to_abs_path(photo_src)
                            if abs_p.exists():
                                st.image(str(abs_p), use_container_width=True)
                            else:
                                st.caption("Foto no encontrada")
                    else:
                        st.caption("Sin foto")

                    st.markdown(f"**{pname}**")
                    st.caption(f"Posición: {row.get('position_label','Unknown')}")
                    st.caption(f"ID: {pid if pid else '—'}")
                    st.caption(f"Estado: {'Activo' if str(row.get('activo')) == '1' else 'Inactivo'}")
                    mp = row.get("matches_played", "")
                    lt = row.get("last_team", "")
                    lmd = row.get("last_match_date", "")
                    st.caption(f"Partidos: {mp if mp != '' else '—'}")
                    st.caption(f"Último equipo: {lt if lt != '' else '—'}")
                    st.caption(f"Último partido: {str(lmd)[:10] if lmd else '—'}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Ver / editar", key=f"open_{widget_suffix}"):
                            st.session_state["manage_players_selected"] = player_key
                            st.rerun()
                    with col_b:
                        is_open = st.session_state.get("manage_players_selected") == player_key
                        if st.button("Cerrar", key=f"close_{widget_suffix}", disabled=(not is_open)):
                            st.session_state.pop("manage_players_selected", None)
                            st.rerun()

                    if st.session_state.get("manage_players_selected") == player_key:
                        st.divider()
                        st.markdown("**Editar jugador**")

                        current_active = int(row.get("activo") or 0) == 1
                        current_photo_src = _clean_str(row.get("photo_url"))

                        with st.form(key=f"edit_form_{widget_suffix}"):
                            new_active = st.checkbox("Activo", value=current_active)
                            uploaded = st.file_uploader(
                                "Subir nueva foto",
                                type=["jpg", "jpeg", "png", "webp"],
                                key=f"card_upload_{widget_suffix}",
                                help="Se guarda optimizada (512x512 JPEG) en data/player_photos/."
                            )
                            new_photo_url = st.text_input(
                                "Foto (URL o ruta local)",
                                value="",
                                key=f"card_photo_src_{widget_suffix}",
                                help="Si subes un archivo, esto se ignora. Ej: https://... o data/player_photos/<id>.jpg"
                            ).strip()

                            submitted = st.form_submit_button("Guardar cambios")

                        col_rm, col_sp = st.columns(2)
                        with col_rm:
                            if st.button("Quitar foto", key=f"remove_photo_{widget_suffix}", disabled=(not current_photo_src)):
                                try:
                                    players_df = _update_player_row(
                                        players_df,
                                        player_id=pid,
                                        player_name=pname,
                                        updates={"photo_url": None},
                                    )
                                    save_players_df(players_df, PLAYERS_FILE_XLSX)
                                    load_players_df.clear()
                                    st.success("Foto eliminada (del Excel).")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"No se pudo quitar la foto: {e}")
                        with col_sp:
                            if submitted:
                                try:
                                    photo_value = current_photo_src or None
                                    if uploaded is not None:
                                        photo_value = _save_uploaded_photo(uploaded, player_id=pid, player_name=pname)
                                    elif new_photo_url:
                                        photo_value = new_photo_url

                                    players_df = _update_player_row(
                                        players_df,
                                        player_id=pid,
                                        player_name=pname,
                                        updates={
                                            "activo": 1 if new_active else 0,
                                            "photo_url": photo_value,
                                        },
                                    )
                                    save_players_df(players_df, PLAYERS_FILE_XLSX)
                                    load_players_df.clear()
                                    st.success("Cambios guardados.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error al guardar cambios: {e}")

with st.expander("📋 Vista avanzada (tabla)", expanded=False):
    edited_df = st.data_editor(
        display_df[["playerName", "playerId", "position_label", "activo", "photo_preview", "photo_url", "matches_played", "last_team", "last_match_date"]],
        num_rows="dynamic",
        column_config={
            "playerName": st.column_config.TextColumn(
                "Nombre del jugador",
                required=True,
                help="Nombre completo tal como aparece en la base de datos"
            ),
            "playerId": st.column_config.TextColumn(
                "playerId",
                required=False,
                help="Identificador del jugador (requerido para jugadores activos)"
            ),
            "position_label": st.column_config.TextColumn("Posición", disabled=True),
            "photo_preview": st.column_config.ImageColumn("Foto"),
            "photo_url": st.column_config.TextColumn(
                "Foto (src)",
                required=False,
                help="URL pública o ruta local (por ejemplo: data/player_photos/123.jpg)."
            ),
            "activo": st.column_config.SelectboxColumn(
                "Activo (1=Sí, 0=No)",
                help="1 = jugador activo, 0 = jugador inactivo",
                options=[1, 0],
                required=True,
                width="small"
            ),
            "matches_played": st.column_config.NumberColumn("Partidos", disabled=True),
            "last_team": st.column_config.TextColumn("Último equipo", disabled=True),
            "last_match_date": st.column_config.DatetimeColumn("Último partido", disabled=True),
        },
        disabled=["position_label", "photo_preview", "matches_played", "last_team", "last_match_date"],
        key="players_table_editor"
    )

    if st.button("Guardar cambios (tabla)", type="primary"):
        try:
            # Merge edits back into the full dataframe by playerName
            df_to_save = players_df.copy()
            if "photo_url" not in df_to_save.columns:
                df_to_save["photo_url"] = None
            edits = edited_df.copy()
            # Normalize keys for merge
            df_to_save["playerName_key"] = df_to_save["playerName"].astype(str).str.strip().str.lower()
            edits["playerName_key"] = edits["playerName"].astype(str).str.strip().str.lower()

            # Update rows existing in df_to_save with edits
            edit_cols = ["playerName", "playerId", "activo"]
            if "photo_url" in edits.columns:
                edit_cols.append("photo_url")
            update_map = edits.set_index("playerName_key")[edit_cols].to_dict(orient="index")
            def apply_update(row):
                key = row["playerName_key"]
                if key in update_map:
                    upd = update_map[key]
                    row["playerName"] = upd.get("playerName", row["playerName"])
                    row["playerId"] = upd.get("playerId", row.get("playerId"))
                    row["activo"] = upd.get("activo", row.get("activo", 1))
                    if "photo_url" in upd:
                        row["photo_url"] = upd.get("photo_url", row.get("photo_url"))
                return row
            df_to_save = df_to_save.apply(apply_update, axis=1)
            df_to_save = df_to_save.drop(columns=["playerName_key"], errors='ignore')

            # Normalize whitespace and types
            df_to_save["playerName"] = df_to_save["playerName"].astype(str).str.strip()
            df_to_save["playerId"] = df_to_save["playerId"].astype("string").where(df_to_save["playerId"].notna(), None)
            df_to_save["playerId"] = df_to_save["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df_to_save["activo"] = pd.to_numeric(df_to_save["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
            if "photo_url" in df_to_save.columns:
                df_to_save["photo_url"] = df_to_save["photo_url"].astype("string").where(df_to_save["photo_url"].notna(), None)
                df_to_save["photo_url"] = df_to_save["photo_url"].apply(lambda x: x.strip() if isinstance(x, str) else x)

            # Validation rules
            errors = []
            # 1) Names required
            if (df_to_save["playerName"].str.len() == 0).any():
                errors.append("Hay jugadores con nombre vacío.")
            # 2) Active players must have non-empty playerId
            active_without_id = df_to_save[(df_to_save["activo"] == 1) & ((df_to_save["playerId"].isna()) | (df_to_save["playerId"].astype(str).str.strip() == ""))]
            if not active_without_id.empty:
                errors.append("Jugadores activos sin playerId. Asigne un playerId antes de guardar.")
            # 3) playerId uniqueness among non-null IDs
            ids = df_to_save["playerId"].dropna().astype(str).str.strip()
            if ids.duplicated().any():
                dupes = ids[ids.duplicated()].unique().tolist()
                errors.append(f"playerId duplicados: {', '.join(dupes)}")

            if errors:
                for e in errors:
                    st.error(f"❌ {e}")
                st.stop()

            save_players_df(df_to_save, PLAYERS_FILE_XLSX)
            st.success("✅ Cambios guardados correctamente!")
            # clear cache to reflect new data if user returns later
            load_players_df.clear()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error al guardar los cambios: {e}")
