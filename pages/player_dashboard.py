import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
import os
import datetime
from sqlalchemy import text
from PIL import Image
from db_utils import connect_to_db, load_player_data, load_event_data_for_matches, get_player_position, process_player_metrics
from db_utils import get_all_players, process_player_comparison_metrics
from math import ceil
from typing import Tuple, Dict, Any
from sqlalchemy import create_engine
from pandas.io.formats.style import Styler
import streamlit.components.v1 as components  # ✅ needed for working HTML injection
from player_ids import normalize_whoscored_player_id, whoscored_player_url
from utils.sheets_client import GoogleSheetsClient
from utils.pdf_cover_photos import list_cover_photos, save_cover_photo

# Optional: helper to navigate between pages
try:
    from streamlit_extras.switch_page_button import switch_page
except ModuleNotFoundError:
    switch_page = None


# === PAGE CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')
BACKGROUND_COVER_PATH = os.path.join(IMG_DIR, "Watford_portada_d.jpg")

st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_sheets_client() -> GoogleSheetsClient:
    cache_key = "_pages_player_dashboard_sheets_client"
    cached_client = st.session_state.get(cache_key)
    if isinstance(cached_client, GoogleSheetsClient):
        return cached_client

    client = GoogleSheetsClient()
    st.session_state[cache_key] = client
    return client


def _safe_pdf_filename(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name or "").strip())
    sanitized = sanitized.strip("_")
    return sanitized or "player_report"


if hasattr(st, "dialog"):
    @st.dialog("PDF Cover Photo")
    def _pdf_cover_photo_dialog(player_key: str, player_label: str, session_key: str, key_prefix: str):
        st.caption("Upload a player photo and keep it in history for future PDF reports.")
        uploaded_photo = st.file_uploader(
            "Upload player photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"{key_prefix}_upload",
        )
        if st.button("Save uploaded photo", key=f"{key_prefix}_save_upload"):
            if uploaded_photo is None:
                st.warning("Please upload an image first.")
            else:
                try:
                    saved_path = save_cover_photo(
                        uploaded_file=uploaded_photo,
                        base_dir=BASE_DIR,
                        player_key=player_key,
                    )
                    st.session_state[session_key] = saved_path
                    st.success("Photo saved and set as active cover.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save photo: {exc}")

        history = list_cover_photos(base_dir=BASE_DIR, player_key=player_key)
        if not history:
            st.info(f"No saved cover photos yet for {player_label}.")
            if st.button("Use no photo", key=f"{key_prefix}_none_only"):
                st.session_state[session_key] = None
                st.success("PDF cover will use no player photo.")
            return

        options = [None] + [item["path"] for item in history]
        labels = {None: "No photo"}
        for item in history:
            labels[item["path"]] = item["label"]

        current_value = st.session_state.get(session_key)
        if current_value not in options:
            current_value = history[0]["path"]
            st.session_state[session_key] = current_value

        selected_value = st.selectbox(
            "Photo history",
            options=options,
            index=options.index(current_value) if current_value in options else 0,
            format_func=lambda value: labels.get(value, "No photo"),
            key=f"{key_prefix}_history_select",
        )
        if selected_value and os.path.exists(selected_value):
            st.image(selected_value, width=220, caption="Cover preview")

        col_apply, col_clear = st.columns(2)
        with col_apply:
            if st.button("Use selected photo", key=f"{key_prefix}_apply"):
                st.session_state[session_key] = selected_value
                st.success("Active cover photo updated.")
        with col_clear:
            if st.button("Use no photo", key=f"{key_prefix}_clear"):
                st.session_state[session_key] = None
                st.success("PDF cover will use no player photo.")
else:
    def _pdf_cover_photo_dialog(*args, **kwargs):
        st.warning("This Streamlit version does not support popup dialogs.")

# === AUTHENTICATION CHECK ===
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be logged in to view this page.")
    st.stop()

# Determine user type
is_staff = st.session_state.user_type == "staff"

# =============================================================
# === GLOBAL DOWNLOAD PDF BUTTON (visible for staff & players) ===
# Captures the visible Streamlit page and downloads it as PDF
# =============================================================

# --- PRINT-TO-PDF BUTTON + PRINT CSS (captures full page, multi-page, crisp) ---
player_name = (st.session_state.get("player_name") or "Player").replace(" ", "_")
section_name = (st.session_state.get("current_section") or "Overview").replace(" ", "_")
pdf_title = f"{player_name}_{section_name}_Watford_Dashboard"

components.html(
    f"""
    <html>
    <head>
      <style>
        /* Floating button */
        #print-btn {{
          position: fixed; top: 15px; right: 25px;
          background-color: #FFD700; color: #000;
          padding: 8px 14px; border: none; border-radius: 10px;
          font-weight: 600; cursor: pointer; z-index: 999999;
          box-shadow: 1px 1px 6px rgba(0,0,0,0.3);
        }}
        #print-btn:hover {{ background-color: #ffea61; }}

        /* PRINT STYLES: full-page export, clean layout */
        @page {{
          size: A4 portrait;              /* change to 'landscape' if you prefer */
          margin: 10mm;                   /* page margins */
        }}
        @media print {{
          /* Hide Streamlit chrome */
          header, footer, [data-testid="stToolbar"],
          [data-testid="stSidebar"], .stDeployButton, .viewerBadge_container__1QSob {{
            display: none !important;
          }}

          /* Make background white and use all width */
          .stApp, body {{
            background: #ffffff !important;
          }}

          /* Tighten top/bottom padding for print */
          .block-container {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;     /* use full width on print */
          }}

          /* Avoid broken cards/charts across pages */
          .element-container, .stPlotlyChart, .js-plotly-plot, .stAltairChart,
          .stMarkdown, .stDataFrame, .stTable {{
            break-inside: avoid-page;
            page-break-inside: avoid;
          }}

          /* Hide the button itself on the PDF */
          #print-btn {{ display: none !important; }}
        }}
      </style>
    </head>
    <body>
      <button id="print-btn">📄 Download PDF</button>

      <script>
        function printWholeApp() {{
          try {{
            const doc = parent && parent.document ? parent.document : document;
            if (!doc) return window.print();

            // Set a descriptive title so browsers use it as the PDF filename
            const previousTitle = doc.title;
            doc.title = "{pdf_title}";

            // Give the DOM a tick to apply styles, then print
            setTimeout(() => {{
              (parent && parent.window ? parent.window : window).print();

              // Restore the original title afterwards
              setTimeout(() => {{ doc.title = previousTitle; }}, 750);
            }}, 100);
          }} catch (e) {{
            // Fallback
            window.print();
          }}
        }}

        document.getElementById("print-btn").addEventListener("click", printWholeApp);
      </script>
    </body>
    </html>
    """,
    height=60
)

# ---- Resolve current player (works for staff and players) ----

def load_players_list() -> Dict[str, Dict[str, Any]]:
    """
    Load players from Google Sheets tab 'Players' (fallback local file).
    Normalizes playerId to STRING (or None) to avoid dtype issues in the UI.
    """
    try:
        players_df = None
        sheets_client = get_sheets_client()
        if sheets_client.is_configured():
            try:
                players_df = sheets_client.read_players_df()
            except Exception as exc:
                st.warning(f"Could not read 'Players' from Google Sheets. Falling back to local file. ({exc})")

        if players_df is None:
            base_csv = os.path.join('data', 'watford_players_login_info.csv')
            base_xlsx = os.path.join('data', 'watford_players_login_info.xlsx')

            if os.path.exists(base_csv):
                players_df = pd.read_csv(base_csv, dtype={"internal_id": "string", "playerId": "string"})
            elif os.path.exists(base_xlsx):
                players_df = pd.read_excel(
                    base_xlsx,
                    converters={
                        "internal_id": lambda x: str(x).strip() if pd.notna(x) else None,
                        "playerId": lambda x: str(x).strip() if pd.notna(x) else None,
                    },
                )
            else:
                st.warning("No players source found. Configure Google Sheets tab 'Players' or local CSV/XLSX.")
                return {}

        players_df.columns = [str(c).strip() for c in players_df.columns]
        if "internal_id" not in players_df.columns:
            players_df["internal_id"] = None

        if 'playerName' not in players_df.columns:
            st.error("❌ Column 'playerName' is required in the players file.")
            return {}

        if 'activo' in players_df.columns:
            players_df['activo'] = pd.to_numeric(players_df['activo'], errors='coerce').fillna(1).astype(int)
        else:
            players_df['activo'] = 1

        has_id = 'playerId' in players_df.columns
        if has_id:
            players_df['playerId'] = players_df['playerId'].astype('string')
            players_df['playerId'] = players_df['playerId'].where(players_df['playerId'].notna(), None)
            players_df['playerId'] = players_df['playerId'].apply(lambda x: x.strip() if isinstance(x, str) else x)
            players_df["playerId"] = players_df["playerId"].apply(normalize_whoscored_player_id)
            players_df["playerId"] = players_df["playerId"].astype("string").where(players_df["playerId"].notna(), None)
        else:
            st.warning("⚠️ Column 'playerId' not found. Some features may fail.")

        players: Dict[str, Dict[str, Any]] = {}
        for _, row in players_df.iterrows():
            full_name = str(row.get('playerName', '')).strip()
            if not full_name:
                continue
            pid = row.get('playerId', None) if has_id else None
            internal_id = str(row.get("internal_id", "")).strip()
            activo = int(row.get('activo', 1))
            id_suffix = f" [{internal_id}]" if internal_id else ""
            label = f"{full_name}{id_suffix}{'' if activo == 1 else ' (Inactive)'}"
            # Handle pandas NA values properly
            if pd.isna(pid) or pid in (None, "", "nan"):
                pid = None
            else:
                pid_str = str(pid).strip()
                pid = pid_str if pid_str.isdigit() else None
            players[label] = {
                "internal_id": internal_id or None,
                "playerId": pid,
                "playerName": full_name,
                "activo": activo,
            }

        return players

    except Exception as e:
        st.error(f"❌ Error loading players source: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {}

def resolve_current_player(is_staff: bool) -> Tuple[str, str]:
    """
    Returns (player_id, player_name) as STRINGS.
    - Staff: pick from selector (playerId stays string or None → error).
    - Player: read from session_state set at login.
    """
    if is_staff:
        st.subheader("Player Dashboard")
        st.write(f"Staff: {st.session_state.get('staff_info', {}).get('full_name', 'User')}")

        if st.button("⚙️ Manage players list"):
            try:
                st.switch_page("pages/manage_players.py")
            except Exception:
                st.info("Open 'manage players' from the sidebar.")

        players = load_players_list()
        if not players:
            st.stop()

        # Filter by status
        status_choice = st.selectbox("Filter Players by Status",
                                     ["All Players", "Active Players", "Inactive Players"],
                                     key="player_status_filter")

        filtered = {}
        for label, data in players.items():
            if status_choice == "Active Players" and data["activo"] != 1:
                continue
            if status_choice == "Inactive Players" and data["activo"] == 1:
                continue
            filtered[label] = data

        options = list(filtered.keys())
        # sort case-insensitive
        options_sorted = sorted(options, key=lambda s: s.lower())
        if status_choice == "All Players":
            options_sorted = [""] + options_sorted

        selected = st.selectbox("Select Player", options=options_sorted, index=0 if status_choice == "All Players" else None,
                                key="player_selector")

        # Sidebar: logout
        st.sidebar.markdown("---")
        st.sidebar.subheader("User Info")
        si = st.session_state.get("staff_info", {})
        st.sidebar.write(f"**Name:** {si.get('full_name','')}")
        st.sidebar.write(f"**Role:** {si.get('role','')}")
        if st.sidebar.button("Logout", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("login.py")

        if not selected:
            st.warning("Please select a player")
            st.stop()

        sel = filtered[selected]
        pid = sel.get("playerId")
        pname = sel.get("playerName", "")
        internal_id = sel.get("internal_id")

        if pid is None or str(pid).strip() == "":
            extra = f" (internal_id: {internal_id})" if internal_id else ""
            st.error(
                "Selected player has no WhoScored 'playerId'. "
                f"Assign one in Manage Players to open match-based dashboard data{extra}."
            )
            st.stop()

        # Always return strings
        return str(pid).strip(), str(pname).strip()

    # Player flow (from session)
    pid = (st.session_state.get("player_id")
           or st.session_state.get("playerInfo", {}).get("player_id")
           or st.session_state.get("player_info", {}).get("player_id")
           or st.session_state.get("player_info", {}).get("playerId"))
    pname = (st.session_state.get("player_name")
             or st.session_state.get("playerInfo", {}).get("player_name")
             or st.session_state.get("player_info", {}).get("player_name")
             or st.session_state.get("player_info", {}).get("full_name")
             or st.session_state.get("player_info", {}).get("playerName"))

    if not pid or not pname:
        st.error("Could not resolve player from session. Ensure login sets 'player_id' and 'player_name'.")
        st.stop()

    return str(pid).strip(), str(pname).strip()



# >>> Call the resolver BEFORE using player_name <<<
player_id, player_name = resolve_current_player(is_staff)

# For players: hide multipage nav links and show a Logout button
if not is_staff:
    st.markdown(
        """
        <style>
            /* Hide Streamlit multipage navigation list in sidebar for players */
            section[data-testid="stSidebarNav"],
            div[data-testid="stSidebarNav"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Menu")
    if st.sidebar.button("Logout", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("login.py")

# Page title

# Global CSS tweak requested
st.markdown(
    """
    <style>
      .st-emotion-cache-zy6yx3 { 
      padding-top: 0rem !important; 
      padding-left: 5rem !important;
      padding-right: 5rem !important;
      }
      .st-emotion-cache-1ip4023 {
      padding-top: 0rem !important;
      }

      .st-emotion-cache-595tnf {
      background-size: 30% auto; !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar header logo via CSS background
def inject_sidebar_logo():
    try:
        img = Image.open(LOGO_PATH)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f"""
            <style>
            div[data-testid="stSidebarHeader"] {{
                background-image: url('data:image/png;base64,{b64}');
                background-repeat: no-repeat;
                background-position: center;
                background-size: contain;
                max-width: 200px;
                max-height: 80px;
                min-height: 80px;
                margin: 0 auto 0.5rem auto;
                padding: 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

inject_sidebar_logo()

# Page title
st.title(f"{player_name}")
whoscored_url = whoscored_player_url(player_id)
if whoscored_url:
    st.markdown(f"[WhoScored]({whoscored_url})")

# --- Load Data ---

match_data, player_data, team_data, player_stats, total_minutes, games_as_starter = load_player_data(player_id, player_name)

if player_data is None:
    st.error("Failed to load player data. Please check the database connection and ensure the required views/tables exist.")
    st.stop()

# After loading player_data as before
player_id = str(player_id)

try:
    match_ids = [str(m) for m in player_data["matchId"].unique()]
    team_id = str(player_data["teamId"].iloc[0])  # assuming single team
except (KeyError, IndexError) as e:
    st.error(f"Error processing player data: {str(e)}")
    st.error("This might be due to missing columns in the data or empty data.")
    st.stop()

try:
    event_data = load_event_data_for_matches(player_id, match_ids, team_id=team_id)
except Exception as e:
    st.error(f"Error loading event data: {str(e)}")
    st.stop()


position_kpi_map = {
    "Goalkeeper": [
        "totalSaves", "save_pct", "goals_conceded", "claimsHigh",
        "collected", "def_actions_outside_box", "ps_xG"
    ],
    "Right Back": [
        "interceptions", "progressive_passes", "recoveries",
        "crosses", "take_on_success_pct", "pass_completion_pct"
    ],
    "Center Back": [
        "interceptions", "progressive_passes", "pass_completion_pct",
        "clearances", "long_pass_pct", "aerial_duel_pct"
    ],
    "Left Back": [
        "interceptions", "progressive_passes", "recoveries",
        "crosses", "take_on_success_pct", "pass_completion_pct"
    ],
    "Defensive Midfielder": [
        "recoveries", "interceptions", "aerial_duel_pct",
        "pass_completion_pct", "progressive_passes", "long_pass_pct",
        "passes_into_penalty_area", "key_passes"
    ],
    "Midfielder": [
        "recoveries", "interceptions", "aerial_duel_pct",
        "pass_completion_pct", "progressive_passes", "long_pass_pct",
        "passes_into_penalty_area", "key_passes"
    ],
    "Right Winger": [
        "pass_completion_pct", "key_passes",
        "passes_into_penalty_area", "crosses", "take_on_success_pct",
        "goal_creating_actions", "shot_creating_actions", "shots_on_target_pct",
        "carries_into_final_third", "carries_into_penalty_area", "xG", "xA"
    ],
    "Attacking Midfielder": [
        "pass_completion_pct", "key_passes",
        "passes_into_penalty_area", "aerial_duel_pct", "take_on_success_pct",
        "goal_creating_actions", "shot_creating_actions", "shots_on_target_pct",
        "carries_into_final_third", "carries_into_penalty_area", "goals", "assists", "xG", "xA"
    ],
    "Left Winger": [
        "pass_completion_pct", "key_passes",
        "passes_into_penalty_area", "crosses", "take_on_success_pct",
        "goal_creating_actions", "shot_creating_actions", "shots_on_target_pct",
        "carries_into_final_third", "carries_into_penalty_area", "xG", "xA"
    ],
    "Striker": [
        "goals", "assists", "xG", "xA", "pass_completion_pct", "key_passes",
        "passes_into_penalty_area", "aerial_duel_pct", "take_on_success_pct",
        "goal_creating_actions", "shot_creating_actions", "shots_on_target_pct",
        "carries_into_final_third", "carries_into_penalty_area"
    ]
}


# Get the actual player position
player_position = get_player_position(player_data, event_data, player_id, player_name)


# Get Metrics

metrics_summary = process_player_metrics(player_stats, event_data, player_id, player_name)




# Labels for all possible metrics (expanded for v3)
metric_labels = {
    "pass_completion_pct": "Passes Completed %",
    "key_passes": "Key Passes",
    "aerial_duel_pct": "Aerial Duels %",
    "take_on_success_pct": "Take-Ons Success %",
    "goal_creating_actions": "Goal Creating Actions",
    "shot_creating_actions": "Shot Creating Actions",
    "shots_on_target_pct": "Shots on Target %",
    "passes_into_penalty_area": "Passes into Penalty Area",
    "carries_into_final_third": "Carries into Final Third",
    "carries_into_penalty_area": "Carries into Penalty Area",
    "goals": "Goals",
    "assists": "Assists",
    "xG": "Expected Goals (xG)",
    "xA": "Assisted Shots",
    "ps_xG": "Post-Shot xG",
    "recoveries": "Recoveries",
    "interceptions": "Interceptions",
    "clearances": "Clearances",
    "crosses": "Crosses",
    "long_pass_pct": "Long Pass %",
    "progressive_passes": "Progressive Pass Distance",
    "progressive_carry_distance": "Progressive Carry Distance",
    "totalSaves": "Saves",
    "claimsHigh": "High Claims",
    "collected": "Collected Balls",
    "def_actions_outside_box": "Defensive Actions Outside Box",
    "throwin_accuracy_pct": "Throw-In Accuracy %",
    "tackle_success_pct": "Tackle Success %",
    "shotsBlocked": "Shots Blocked",
    "shotsOffTarget": "Shots Off Target",
    "shotsOnPost": "Shots on Post",
    "save_pct": "Saves Success %",
     "goals_conceded": "Goals Conceded"
}

# KPI glossary text shown in the KPI selector info popup
metric_definitions = {
    "pass_completion_pct": "Percentage of completed passes out of total passes.",
    "key_passes": "Passes that directly create a teammate shot attempt.",
    "aerial_duel_pct": "Percentage of aerial duels won.",
    "take_on_success_pct": "Percentage of successful dribbles/take-ons.",
    "goal_creating_actions": "Actions that directly lead to a goal.",
    "shot_creating_actions": "Actions that directly lead to a shot.",
    "shots_on_target_pct": "Percentage of shots that are on target.",
    "passes_into_penalty_area": "Completed passes played into the penalty area.",
    "carries_into_final_third": "Ball carries that enter the final third.",
    "carries_into_penalty_area": "Ball carries that enter the penalty area.",
    "goals": "Total goals scored.",
    "assists": "Total assists provided.",
    "xG": "Expected goals based on shot quality.",
    "xA": "Expected assists based on chance creation quality.",
    "ps_xG": "Post-shot expected goals based on shot placement and power.",
    "recoveries": "Times the player regains possession for their team.",
    "interceptions": "Passes/interactions stopped by anticipating play.",
    "clearances": "Defensive actions that remove danger from the area.",
    "crosses": "Crosses attempted into attacking areas.",
    "long_pass_pct": "Percentage of successful long passes.",
    "progressive_passes": "Forward passes that significantly move play toward goal.",
    "progressive_carry_distance": "Distance advanced while carrying the ball forward.",
    "totalSaves": "Total goalkeeper saves.",
    "claimsHigh": "High balls claimed/caught by the goalkeeper.",
    "collected": "Loose balls collected safely by the goalkeeper.",
    "def_actions_outside_box": "Goalkeeper defensive actions outside the penalty area.",
    "throwin_accuracy_pct": "Percentage of accurate throw-ins.",
    "tackle_success_pct": "Percentage of tackles won.",
    "shotsBlocked": "Shots blocked by the player.",
    "shotsOffTarget": "Shots taken that miss the target.",
    "shotsOnPost": "Shots that hit the post or crossbar.",
    "save_pct": "Percentage of shots on target faced that are saved.",
    "goals_conceded": "Goals allowed by the team while this player is in goal.",
}

# Metric type: affects delta logic
metric_type_map = {
    "pass_completion_pct": "percentage",
    "aerial_duel_pct": "percentage",
    "take_on_success_pct": "percentage",
    "shots_on_target_pct": "percentage",
    "long_pass_pct": "percentage",
    "throwin_accuracy_pct": "percentage",
    "tackle_success_pct": "percentage",
    "key_passes": "per_match",
    "goal_creating_actions": "per_match",
    "shot_creating_actions": "per_match",
    "passes_into_penalty_area": "per_match",
    "carries_into_final_third": "per_match",
    "carries_into_penalty_area": "per_match",
    "goals": "per_match",
    "assists": "per_match",
    "xG": "per_match",
    "xA": "per_match",
    "ps_xG": "per_match",
    "recoveries": "per_match",
    "interceptions": "per_match",
    "clearances": "per_match",
    "crosses": "per_match",
    "progressive_passes": "per_match",
    "progressive_carry_distance": "per_match",
    "totalSaves": "per_match",
    "claimsHigh": "per_match",
    "collected": "per_match",
    "def_actions_outside_box": "per_match",
    "shotsBlocked": "per_match",
    "shotsOffTarget": "per_match",
    "shotsOnPost": "per_match"
}

# Tooltip helpers for details in hover or cards
metric_tooltip_fields = {
    "pass_completion_pct": ["passesTotal", "passesAccurate"],
    "aerial_duel_pct": ["aerialsTotal", "aerialsWon"],
    "take_on_success_pct": ["dribblesAttempted", "dribblesWon"],
    "shots_on_target_pct": ["shotsTotal", "shotsOnTarget"],
    "long_pass_pct": ["long_passes_total", "long_passes_success"],
    "throwin_accuracy_pct": ["throwInsTotal", "throwInsAccurate"],
    "tackle_success_pct": ["tacklesTotal", "tackleSuccessful"],
    "key_passes": ["passesKey"],
    "goal_creating_actions": [],
    "shot_creating_actions": [],
    "goals": [],
    "assists": [],
    "xG": [],
    "xA": [],
    "ps_xG": [],
    "recoveries": [],
    "interceptions": [],
    "clearances": [],
    "crosses": [],
    "passes_into_penalty_area": [],
    "carries_into_final_third": [],
    "carries_into_penalty_area": [],
    "progressive_passes": [],
    "progressive_carry_distance": [],
    "totalSaves": [],
    "claimsHigh": [],
    "collected": [],
    "def_actions_outside_box": [],
    "shotsBlocked": [],
    "shotsOffTarget": [],
    "shotsOnPost": []
}

percentage_formula_map = {
        "pass_completion_pct": ("passesAccurate", "passesTotal"),
        "aerial_duel_pct": ("aerialsWon", "aerialsTotal"),
        "take_on_success_pct": ("dribblesWon", "dribblesAttempted"),
        "shots_on_target_pct": ("shotsOnTarget", "shotsTotal"),
        "tackle_success_pct": ("tackleSuccessful", "tacklesTotal"),
        "throwin_accuracy_pct": ("throwInsAccurate", "throwInsTotal"),
        "long_pass_pct": ("long_passes_success", "long_passes_total"),
    }

# Get the metrics player position
position_default_kpis_raw = position_kpi_map.get(player_position, [])
available_kpis = [k for k in metric_labels.keys() if k in metrics_summary.columns]
position_default_kpis = [k for k in position_default_kpis_raw if k in available_kpis]
if position_default_kpis_raw and not position_default_kpis:
    st.warning(f"Default KPIs for position '{player_position}' are not available in this dataset.")
elif not position_default_kpis_raw:
    st.warning(f"No default KPIs found for position: {player_position}.")

all_kpi_options = []
for kpi in position_default_kpis + available_kpis:
    if kpi not in all_kpi_options:
        all_kpi_options.append(kpi)

if not all_kpi_options:
    all_kpi_options = [k for k in metric_type_map.keys() if k in metrics_summary.columns]


# ---------------------------
# DATE JOIN + DELTAS + CARDS
# ---------------------------

def add_match_dates(df: pd.DataFrame, match_data_df: pd.DataFrame):
    """
    Adds 'matchDate' to df by joining with match_data_df on matchId.
    - Robust to dtype mismatches.
    - Removes any pre-existing 'matchDate' on left.
    - Guarantees a timezone-naive pandas datetime column.
    Returns: (df_with_dates, min_date, max_date) where min/max are Python date objects.
    """
    import pandas as pd

    today = pd.Timestamp.today().date()

    # Basic guards
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), today, today
    if not isinstance(match_data_df, pd.DataFrame) or match_data_df.empty:
        out = df.copy()
        out["matchDate"] = pd.NaT
        return out, today, today

    # Work on copies
    left = df.copy()
    right = match_data_df.copy()

    # Drop any pre-existing matchDate on the left to avoid ambiguity
    if "matchDate" in left.columns:
        left = left.drop(columns=["matchDate"])

    # Ensure matchId exists on both sides
    if "matchId" not in left.columns or "matchId" not in right.columns:
        out = left.copy()
        out["matchDate"] = pd.NaT
        return out, today, today

    # Choose the best available date column from match_data
    for candidate in ["startDate", "matchDate", "date", "kickoff"]:
        if candidate in right.columns:
            date_col = candidate
            break
    else:
        out = left.copy()
        out["matchDate"] = pd.NaT
        return out, today, today

    # Parse dates (UTC -> naive) for consistency
    right[date_col] = pd.to_datetime(right[date_col], errors="coerce", utc=True)
    try:
        right[date_col] = right[date_col].dt.tz_convert(None)
    except Exception:
        # Already naive or tz-naive after parsing
        right[date_col] = right[date_col].dt.tz_localize(None)

    # De-dupe match_data on matchId (keep the last non-null date)
    right = (
        right.sort_values(date_col)
             .drop_duplicates(subset=["matchId"], keep="last")
    )

    # Build consistent string join key (simpler/safer than numeric heuristics)
    def as_key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip()

    left["_matchIdKey"] = as_key(left["matchId"])
    right["_matchIdKey"] = as_key(right["matchId"])

    # Merge
    out = left.merge(
        right[["_matchIdKey", date_col]],
        on="_matchIdKey",
        how="left"
    )
    out = out.drop(columns=["_matchIdKey"]).rename(columns={date_col: "matchDate"})

    # Ensure matchDate is datetime64[ns] (naive)
    out["matchDate"] = pd.to_datetime(out["matchDate"], errors="coerce")

    # Compute min/max over valid dates
    valid = out["matchDate"].dropna()
    if valid.empty:
        return out, today, today

    min_date = valid.min().date()
    max_date = valid.max().date()
    return out, min_date, max_date



# ---- DELTAS ----

def compute_metric_average(df: pd.DataFrame, column: str) -> float:
    """Compute metric value per period (percentage from weighted ratios, others per match)."""
    if df.empty:
        return 0.0

    metric_type = metric_type_map.get(column, "per_match")
    if metric_type == "percentage" and column in percentage_formula_map:
        num_col, denom_col = percentage_formula_map[column]
        if num_col in df.columns and denom_col in df.columns:
            numerator = pd.to_numeric(df[num_col], errors="coerce").fillna(0).sum()
            denominator = pd.to_numeric(df[denom_col], errors="coerce").fillna(0).sum()
            return float((numerator / denominator) * 100) if denominator != 0 else 0.0

    if column not in df.columns:
        return 0.0

    series = pd.to_numeric(df[column], errors="coerce")
    if metric_type == "percentage":
        mean_val = series.mean()
        return 0.0 if pd.isna(mean_val) else float(mean_val)

    return float(series.fillna(0).sum() / max(1, len(df)))


def calculate_delta(delta_df: pd.DataFrame, reference_df: pd.DataFrame, column: str) -> Tuple[float, float]:
    """
    Calculates delta between Delta window and Reference window for one metric.
    """
    if delta_df.empty or reference_df.empty:
        return 0.0, 0.0

    delta_value = compute_metric_average(delta_df, column)
    reference_value = compute_metric_average(reference_df, column)

    delta = delta_value - reference_value
    delta_percent = (delta / reference_value * 100) if reference_value != 0 else 0
    return round(delta, 1), round(delta_percent, 1)


# ---- METRIC CARD ----

def format_metric_value(value, column):
    """
    Formats a metric value based on its type:
    - Percentages: one decimal
    - Float metrics (xG, xA, etc.): two decimals
    - Count-based metrics: integer
    """
    metric_type = metric_type_map.get(column, "per_match")
    if pd.isnull(value):
        return "0"
    if metric_type == "percentage":
        return f"{value:.1f}"
    elif column in ["xG", "xA", "ps_xG", "progressive_passes", "progressive_carry_distance"]:
        return f"{value:.2f}"
    else:
        return f"{int(round(value))}"


def display_metric_card(col, title, value, delta_df, reference_df, column, color=None):
    with col:
        with st.container(border=True):
            delta, delta_percent = calculate_delta(delta_df, reference_df, column)
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""

            # Format value
            metric_type = metric_type_map.get(column, "per_match")
            formatted_value = format_metric_value(value, column)

            # Tooltip content
            tooltip_lines = [f"{title}"]

            # Related raw metric values
            for extra_field in metric_tooltip_fields.get(column, []):
                if extra_field in delta_df.columns:
                    raw_val = delta_df[extra_field].sum()
                    label = metric_labels.get(extra_field, extra_field.replace("_", " ").title())
                    tooltip_lines.append(f"{label} (Delta): {int(raw_val)}")

            # Avg Reference vs Delta logic
            if metric_type == "percentage" and column in percentage_formula_map:
                num_col, denom_col = percentage_formula_map[column]
                ref_num = pd.to_numeric(reference_df.get(num_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                ref_denom = pd.to_numeric(reference_df.get(denom_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                delta_num = pd.to_numeric(delta_df.get(num_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                delta_denom = pd.to_numeric(delta_df.get(denom_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

                reference_avg = (ref_num / ref_denom) * 100 if ref_denom != 0 else 0
                delta_avg = (delta_num / delta_denom) * 100 if delta_denom != 0 else 0

                tooltip_lines.append(f"Avg Reference: {reference_avg:.1f}%")
                tooltip_lines.append(f"Avg Delta: {delta_avg:.1f}%")

            else:
                reference_avg = compute_metric_average(reference_df, column)
                delta_avg = compute_metric_average(delta_df, column)
                suffix = "%" if metric_type == "percentage" else ""
                tooltip_lines.append(f"Avg Reference: {reference_avg:.1f}{suffix}")
                tooltip_lines.append(f"Avg Delta: {delta_avg:.1f}{suffix}")

            # Tooltip as hover title
            tooltip_html = "&#013;".join(tooltip_lines)

            # Display scorecard
            st.markdown(
                f"""
                <div title="{tooltip_html}" style='
                    text-align: center;
                    margin-bottom: 1.5rem;
                    padding: 0.5rem 0;
                '>
                    <div style='font-weight: bold; color: #333; margin-bottom: 0.3rem;'>{title}</div>
                    <div style='font-size: 1.6rem; font-weight: normal; color: #333;'>{formatted_value}</div>
                    <div style='color: {"green" if delta > 0 else "red" if delta < 0 else "#888"}; font-size: 0.9rem; margin-top: 0.2rem;'>
                        {arrow} {delta:+.1f} ({delta_percent:+.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ---------------------------
# WIRE DATES + SIDEBAR FILTER
# ---------------------------
def to_int64_id(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")

if "matchId" not in metrics_summary.columns:
    st.error(f"'matchId' missing in metrics_summary. Columns: {list(metrics_summary.columns)}"); st.stop()
if "matchId" not in match_data.columns:
    st.error(f"'matchId' missing in match_data. Columns: {list(match_data.columns)}"); st.stop()

metrics_summary = metrics_summary.copy()
match_data = match_data.copy()

metrics_summary["matchId"] = to_int64_id(metrics_summary["matchId"])
match_data["matchId"]      = to_int64_id(match_data["matchId"])

if "startDate" not in match_data.columns:
    st.error("match_data is missing 'startDate'."); st.stop()
match_data["startDate"] = pd.to_datetime(match_data["startDate"], errors="coerce")

# --- NEW: include 'season' while we dedupe match_data ---
needed_cols = ["matchId", "startDate"]
if "season" in match_data.columns:
    needed_cols.append("season")
else:
    st.warning("⚠️ 'season' column not found in match_data; Season filter will be hidden.")

md_meta = (
    match_data[needed_cols]
    .dropna(subset=["matchId"])
    .sort_values("startDate")
    .drop_duplicates(subset=["matchId"], keep="last")
    .rename(columns={"startDate": "matchDate"})
)

# Merge date (and season if present)
metrics_summary = metrics_summary.merge(md_meta, on="matchId", how="left")

# Validate match coverage
non_null_rate = metrics_summary["matchDate"].notna().mean() if "matchDate" in metrics_summary else 0.0

if non_null_rate == 0:
    st.error("No match dates matched. Likely mismatch in matchId values between frames.")
    st.write("metrics_summary sample:", metrics_summary[["matchId"]].head())
    st.write("match_data sample:", match_data[["matchId", "startDate"]].head())
    st.stop()

# ------------------------------
# Main area: Season + Time + fixed comparison + KPI selection
# ------------------------------
st.markdown("### Filters")

# 1) Season selector (only if season exists)
if "season" in metrics_summary.columns:
    # normalize to string for UI
    seasons = (
        metrics_summary["season"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    # Prefer most recent as default if you like; here we keep "All seasons".
    season_options = ["All seasons"] + seasons
    season_choice = st.selectbox("Season", options=season_options, index=0)
    if season_choice == "All seasons":
        st.session_state.selected_season = None   # means no explicit season filter
    else:
        st.session_state.selected_season = season_choice

    if season_choice == "All seasons":
        season_filtered = metrics_summary
    else:
        season_filtered = metrics_summary[metrics_summary["season"].astype(str) == season_choice]
        if season_filtered.empty:
            st.warning("No matches for the selected season. Showing all seasons instead.")
            season_filtered = metrics_summary
else:
    # Fallback: no season column
    season_filtered = metrics_summary

# 2) Time ranges (constrained by the season-filtered rows)
valid_dates = season_filtered["matchDate"].dropna()

def normalize_date_range(value, default_start, default_end):
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
    else:
        start, end = default_start, default_end
    return (start, end) if start <= end else (end, start)

def clamp_date_range(value, min_bound, max_bound):
    start, end = normalize_date_range(value, min_bound, max_bound)
    start = max(min_bound, min(start, max_bound))
    end = max(min_bound, min(end, max_bound))
    return (start, end) if start <= end else (min_bound, max_bound)

delta_match_ids = set()
reference_match_ids = set()
comparison_mode_label = "Previous month"
comparison_value_label = "1 month"

if valid_dates.empty:
    st.warning("No dated matches available to build time filters. Showing all data.")
    today = pd.Timestamp.today().date()
    analysis_start_date, analysis_end_date = today, today
    delta_start_date, delta_end_date = today, today
    reference_start_date, reference_end_date = today, today
    filtered_df = season_filtered.copy()
    reference_filtered_df = season_filtered.copy()
else:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    st.markdown("**1) Analysis Window**")
    analysis_default = clamp_date_range(
        st.session_state.get("analysis_date_range", (min_date, max_date)),
        min_date,
        max_date,
    )
    picked_analysis = st.date_input(
        "Analysis date range",
        value=analysis_default,
        min_value=min_date,
        max_value=max_date,
        key="analysis_date_range",
    )
    analysis_start_date, analysis_end_date = clamp_date_range(picked_analysis, min_date, max_date)

    analysis_mask = (
        season_filtered["matchDate"].dt.date >= analysis_start_date
    ) & (
        season_filtered["matchDate"].dt.date <= analysis_end_date
    )
    analysis_filtered = season_filtered.loc[analysis_mask].copy()

    st.caption(
        f"Analysis matches: {int(analysis_filtered['matchId'].nunique()) if 'matchId' in analysis_filtered.columns else 0}"
    )

    if analysis_filtered.empty:
        st.warning("No matches in Analysis range.")
        delta_start_date, delta_end_date = analysis_start_date, analysis_end_date
        reference_start_date, reference_end_date = analysis_start_date, analysis_end_date
        filtered_df = analysis_filtered.copy()
        reference_filtered_df = analysis_filtered.copy()
    else:
        st.markdown("**2) Comparison period**")
        st.caption("Comparison is always against the previous calendar month.")
        # Main stats use the full selected analysis window.
        filtered_df = analysis_filtered.sort_values("matchDate").copy()
        delta_start_date = analysis_start_date
        delta_end_date = analysis_end_date
        delta_match_ids = set(filtered_df["matchId"].dropna().tolist())

        analysis_end_ts = pd.Timestamp(analysis_end_date)
        reference_filtered_df = season_filtered.iloc[0:0].copy()
        comparison_mode_label = "Previous month"
        comparison_value_label = "1 month"
        current_month_start = analysis_end_ts.to_period("M").start_time.normalize()
        ref_start_ts = (current_month_start - pd.DateOffset(months=1)).normalize()
        ref_end_ts = (current_month_start - pd.Timedelta(days=1)).normalize()
        reference_filtered_df = season_filtered[
            (season_filtered["matchDate"] >= ref_start_ts) &
            (season_filtered["matchDate"] <= ref_end_ts)
        ].copy()

        reference_match_ids = set(reference_filtered_df["matchId"].dropna().tolist())
        if reference_filtered_df.empty:
            reference_start_date, reference_end_date = ref_start_ts.date(), ref_end_ts.date()
            st.warning("No matches found in the selected reference. Delta values will show 0.")
        else:
            reference_start_date = reference_filtered_df["matchDate"].min().date()
            reference_end_date = reference_filtered_df["matchDate"].max().date()

        st.caption(
            f"Main range: {delta_start_date.strftime('%d/%m/%Y')} -> {delta_end_date.strftime('%d/%m/%Y')}"
        )
        st.caption(
            "Comparison period (previous month): "
            f"{reference_start_date.strftime('%d/%m/%Y')} -> {reference_end_date.strftime('%d/%m/%Y')}"
        )
        st.caption(f"Selected matches: {int(filtered_df['matchId'].nunique()) if 'matchId' in filtered_df.columns else 0}")
        st.caption(f"Comparison matches: {int(reference_filtered_df['matchId'].nunique()) if 'matchId' in reference_filtered_df.columns else 0}")

st.markdown("**3) Data to display**")
glossary_sorted = sorted(
    all_kpi_options,
    key=lambda k: metric_labels.get(k, k.replace("_", " ").title()).lower(),
)

def _build_kpi_glossary_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "KPI": metric_labels.get(kpi, kpi.replace("_", " ").title()),
            "Definition": metric_definitions.get(kpi, "Definition not available."),
        }
        for kpi in glossary_sorted
    ])

if hasattr(st, "dialog"):
    @st.dialog("KPI Dictionary")
    def show_kpi_dictionary():
        glossary_df = _build_kpi_glossary_df()
        search_text = st.text_input(
            "Search KPI or definition",
            placeholder="e.g. xG, passes, aerial",
            key="kpi_dictionary_search",
        ).strip()
        if search_text:
            mask = (
                glossary_df["KPI"].str.contains(search_text, case=False, na=False) |
                glossary_df["Definition"].str.contains(search_text, case=False, na=False)
            )
            glossary_df = glossary_df[mask].copy()

        st.dataframe(
            glossary_df,
            use_container_width=True,
            hide_index=True,
            height=520,
        )
        st.caption(f"Showing {len(glossary_df)} KPI definition(s).")

    _, info_col_right = st.columns([5, 2])
    with info_col_right:
        if st.button("ℹ️ Open KPI Dictionary", use_container_width=True):
            show_kpi_dictionary()
else:
    with st.expander("ℹ️ Open KPI Dictionary"):
        glossary_df = _build_kpi_glossary_df()
        search_text = st.text_input(
            "Search KPI or definition",
            placeholder="e.g. xG, passes, aerial",
            key="kpi_dictionary_search_fallback",
        ).strip()
        if search_text:
            mask = (
                glossary_df["KPI"].str.contains(search_text, case=False, na=False) |
                glossary_df["Definition"].str.contains(search_text, case=False, na=False)
            )
            glossary_df = glossary_df[mask].copy()

        st.dataframe(glossary_df, use_container_width=True, hide_index=True, height=520)
        st.caption(f"Showing {len(glossary_df)} KPI definition(s).")

selected_kpis = st.multiselect(
    "KPIs",
    options=all_kpi_options,
    default=position_default_kpis,
    format_func=lambda k: metric_labels.get(k, k.replace("_", " ").title()),
    key="selected_kpis_main",
)
st.caption("Default KPIs for the player position are preselected. You can add or remove any available KPI.")
if not selected_kpis:
    st.warning("Select at least one KPI to display.")

if "selected_kpis" not in locals() or not selected_kpis:
    selected_kpis = all_kpi_options[:1] if all_kpi_options else []

# Keep current global date window for other sections as the Analysis window
start_date, end_date = analysis_start_date, analysis_end_date

# -- Dynamically assigned KPIs
metric_keys = selected_kpis

# --- Section Selector (both roles) ---
st.sidebar.header("Select Visualization")
section = st.sidebar.radio(
    "Go to section:",
    options=["Overview Stats", "Trends Stats", "Player Comparison"],
    index=0,
    key="selected_section"
)

# --- Player Info Section (always at top)

# Normalize IDs to avoid empty filters due to dtype mismatches
player_data = player_data.copy()
player_data["playerId"] = pd.to_numeric(player_data["playerId"], errors="coerce").astype("Int64")
_pid = pd.to_numeric(player_id, errors="coerce")

# Extract player static details (only once)
filtered_player_data = player_data[player_data["playerId"] == _pid]

if filtered_player_data.empty:
    st.warning(
        "No rows matched for this player in player_data after ID normalization. "
        "Showing the first available row as a fallback."
    )
    # Helpful debug info:
    st.caption(f"player_id (incoming) → {_pid} (type: {type(_pid)})")
    st.caption(f"Unique playerIds in player_data (head): {list(player_data['playerId'].dropna().unique()[:5])}")
    # Fallback to first row to avoid crash; adjust if you prefer to stop instead.
    player_info = player_data.iloc[0]
else:
    player_info = filtered_player_data.iloc[0]

age          = player_info.get("age", None)
shirt_number = player_info.get("shirtNo", None)
height       = player_info.get("height", None)
weight       = player_info.get("weight", None)
team_name    = player_info.get("teamName", None)

# --- Now dynamic stats based on time filter (start_date, end_date)

# 1. Merge player_data with match_data (bring in dates)
md = match_data.copy()
if "matchId" in md.columns:
    md["matchId"] = pd.to_numeric(md["matchId"], errors="coerce").astype("Int64")
if "startDate" in md.columns:
    md["startDate"] = pd.to_datetime(md["startDate"], errors="coerce")

filtered_logged_player_info = (
    player_data.merge(md[["matchId", "startDate"]], on="matchId", how="left")
)

# 2. Apply player ID and date filter (guard for missing dates)
has_dates = "startDate" in filtered_logged_player_info.columns
if has_dates:
    filtered_logged_player_info = filtered_logged_player_info[
        (filtered_logged_player_info["playerId"] == _pid) &
        (filtered_logged_player_info["startDate"].notna()) &
        (filtered_logged_player_info["startDate"].dt.date >= start_date) &
        (filtered_logged_player_info["startDate"].dt.date <= end_date)
    ]
else:
    st.warning("No startDate column available after merge; using all rows for this player.")
    filtered_logged_player_info = filtered_logged_player_info[
        (filtered_logged_player_info["playerId"] == _pid)
    ]

# 3. Totals within the selected window
games_played     = int(filtered_logged_player_info.shape[0])
games_as_starter = int(pd.to_numeric(filtered_logged_player_info.get("isFirstEleven", 0), errors="coerce").fillna(0).sum())
total_minutes    = float(pd.to_numeric(filtered_logged_player_info.get("minutesPlayed", 0), errors="coerce").fillna(0).sum())

# --- Labels and Values for display
labels = [
    "Age", "Shirt Number", "Height", "Weight", "Games Played",
    "Games as Starter", "Minutes Played"
]
values = [
    age, shirt_number, height, weight,
    games_played, games_as_starter, int(round(total_minutes))
]

# --- Styled Player Info Cards
with st.container():
    info_cols = st.columns(len(labels))
    for i, (label, value) in enumerate(zip(labels, values)):
        with info_cols[i]:
            st.markdown(
                f"""
                <div style="
                    background-color: #ffe6e6;
                    padding: 0.6rem;
                    border-radius: 12px;
                    text-align: center;
                    margin-bottom: 1rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 0.85rem; color: #444; font-weight: 500;">{label}</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #555;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


if section == "Overview Stats":
    st.markdown("""
    <div style="
        background-color: #fff9cc;
        padding: 1rem;
        border-left: 5px solid #fcec03;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 1.5rem;
        margin-top: 1.5rem;
    ">
    Main Performance Stats
    </div>
    """, unsafe_allow_html=True)

    # --- Merge team names into metrics_summary ---
    teams_info = (
        event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
        .first()
        .reset_index()
    )
    teams_info.to_excel("teams_info_lucas.xlsx")
    summary_df = metrics_summary.copy()
    summary_df = summary_df.merge(teams_info, on="matchId", how="left")

    # Ensure we don't duplicate matchDate before wiring dates
    if "matchDate" in summary_df.columns:
        summary_df = summary_df.drop(columns=["matchDate"])

    # Add match dates ONCE (expects a datetime column after this)
    summary_df, _, _ = add_match_dates(summary_df, match_data)

    # Make sure matchDate is datetime
    summary_df["matchDate"] = pd.to_datetime(summary_df["matchDate"], errors="coerce")

    # Sort by date
    summary_df = summary_df.sort_values("matchDate")

    # --- Apply Main + Reference period filters ---
    summary_df["matchDate"] = pd.to_datetime(summary_df["matchDate"], errors="coerce")
    summary_df["match_date_only"] = summary_df["matchDate"].dt.date
    summary_df["is_main_period"] = summary_df["matchId"].isin(delta_match_ids)
    summary_df["is_reference_period"] = summary_df["matchId"].isin(reference_match_ids)

    period_filtered_df = summary_df[
        summary_df["is_main_period"] | summary_df["is_reference_period"]
    ].copy()
    period_filtered_df = period_filtered_df.drop_duplicates(subset="matchId", keep="last")

    # Only fill NaNs for numeric columns.
    num_cols = period_filtered_df.select_dtypes(include=["number"]).columns
    period_filtered_df[num_cols] = period_filtered_df[num_cols].fillna(0)

    # Ensure team names are present (merge again if needed)
    if "oppositionTeamName" not in period_filtered_df.columns:
        teams_info = (
            event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
            .first()
            .reset_index()
        )
        period_filtered_df = period_filtered_df.merge(teams_info, on="matchId", how="left")
    else:
        period_filtered_df = period_filtered_df.copy()

    if period_filtered_df.empty:
        st.info("No matches in the selected Main/Reference ranges.")
        st.stop()

    # --- Create readable match labels and usage tags ---
    period_filtered_df["period_tag"] = np.select(
        [
            period_filtered_df["is_main_period"] & period_filtered_df["is_reference_period"],
            period_filtered_df["is_main_period"],
            period_filtered_df["is_reference_period"],
        ],
        ["Selected + Comparison", "Selected period", "Comparison period"],
        default="OUT",
    )
    date_str = period_filtered_df["matchDate"].dt.strftime("%Y-%m-%d").fillna("Unknown date")
    opp_str = period_filtered_df["oppositionTeamName"].astype("string").fillna("Unknown")
    period_filtered_df["match_label"] = (
        date_str + " vs " + opp_str + " - " + period_filtered_df["period_tag"]
    )

    # --- Match Filter Styled Like Excel ---
    with st.expander("Filter by Match (click to hide)", expanded=True):
        st.caption("Filter matches used in the selected period and in the comparison period.")
        match_options = (
            period_filtered_df[["matchId", "match_label", "matchDate", "period_tag"]]
            .drop_duplicates()
            .sort_values("matchDate")
        )
        all_match_ids = list(match_options["matchId"])
        state_key = "selected_match_ids_kpi"

        # Search box
        search_text = st.text_input("🔍 Search match:", "", key="match_search_kpi")
        filtered_options = match_options[
            match_options["match_label"].str.contains(search_text, case=False, na=False)
        ].copy()
        visible_match_ids = list(filtered_options["matchId"])

        # Initialize and normalize selected IDs in session state
        if state_key not in st.session_state:
            st.session_state[state_key] = all_match_ids
        else:
            st.session_state[state_key] = [mid for mid in st.session_state[state_key] if mid in all_match_ids]

        # Select all / clear buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Select All Matches", key="select_all_matches_kpi"):
                st.session_state[state_key] = all_match_ids
                for mid in all_match_ids:
                    st.session_state[f"match_kpi_{mid}"] = True
        with col2:
            if st.button("Clear Matches", key="clear_matches_kpi"):
                st.session_state[state_key] = []
                for mid in all_match_ids:
                    st.session_state[f"match_kpi_{mid}"] = False

        selected_ids = set(st.session_state[state_key])
        selected_visible_ids = []
        visible_set = set(visible_match_ids)

        st.markdown("<div style='max-height: 250px; overflow-y: auto; padding: 0 10px;'>", unsafe_allow_html=True)
        for _, row in filtered_options.iterrows():
            mid = row["matchId"]
            label = row["match_label"]
            cb_key = f"match_kpi_{mid}"
            if cb_key not in st.session_state:
                st.session_state[cb_key] = mid in selected_ids
            checked = st.checkbox(label, key=cb_key)
            if checked:
                selected_visible_ids.append(mid)
        st.markdown("</div>", unsafe_allow_html=True)

        hidden_selected = [mid for mid in st.session_state[state_key] if mid not in visible_set]
        updated_set = set(hidden_selected) | set(selected_visible_ids)
        st.session_state[state_key] = [mid for mid in all_match_ids if mid in updated_set]

    # Apply match filter (guard empty state)
    current_selected = st.session_state.get("selected_match_ids_kpi", [])
    if current_selected:
        selected_matches_df = period_filtered_df[period_filtered_df["matchId"].isin(current_selected)].copy()
    else:
        selected_matches_df = period_filtered_df.iloc[0:0].copy()

    # Early exit if no rows after filters
    if selected_matches_df.empty:
        st.info("No matches in the selected filters.")
        st.stop()

    main_df = selected_matches_df[selected_matches_df["is_main_period"]].copy()
    reference_df = selected_matches_df[selected_matches_df["is_reference_period"]].copy()

    if main_df.empty:
        st.info("No selected-period matches chosen. Select at least one match.")
        st.stop()
    if reference_df.empty:
        st.warning("No comparison-period matches selected. Delta values are shown as 0.")

    st.caption(f"Showing Metrics for position: **{player_position}**")
    st.caption(
        f"Selected matches: {int(main_df['matchId'].nunique())} | "
        f"Comparison matches: {int(reference_df['matchId'].nunique())}"
    )

    # --- Set metric_keys dynamically by position ---
    metric_keys = selected_kpis

    # --- Aggregate Metrics ---
    def compute_weighted_percentage(df, numerator_col, denominator_col):
        num = df.get(numerator_col, pd.Series(dtype=float)).sum()
        denom = df.get(denominator_col, pd.Series(dtype=float)).sum()
        return round((num / denom) * 100, 1) if denom and denom != 0 else 0

    aggregated_metrics = {}

    for key in metric_keys:
        if key not in main_df.columns:
            aggregated_metrics[key] = 0
            continue

        metric_type = metric_type_map.get(key, "per_match")

        # Custom logic for % metrics based on true numerators/denominators
        if key == "pass_completion_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "passesAccurate", "passesTotal")
        elif key == "aerial_duel_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "aerialsWon", "aerialsTotal")
        elif key == "take_on_success_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "dribblesWon", "dribblesAttempted")
        elif key == "shots_on_target_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "shotsOnTarget", "shotsTotal")
        elif key == "tackle_success_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "tackleSuccessful", "tacklesTotal")
        elif key == "throwin_accuracy_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "throwInsAccurate", "throwInsTotal")
        elif key == "long_pass_pct":
            aggregated_metrics[key] = compute_weighted_percentage(main_df, "long_passes_success", "long_passes_total")
        else:
            # Sum or mean depending on type
            if metric_type == "percentage":
                aggregated_metrics[key] = round(pd.to_numeric(main_df[key], errors="coerce").mean(), 1)
            else:
                aggregated_metrics[key] = round(pd.to_numeric(main_df[key], errors="coerce").sum(), 2)

    # --- Scorecards (4 per row) ---
    metrics_per_row = 4
    metric_chunks = [metric_keys[i:i + metrics_per_row] for i in range(0, len(metric_keys), metrics_per_row)]

    for chunk in metric_chunks:
        cols = st.columns(len(chunk))
        for i, key in enumerate(chunk):
            label = metric_labels.get(key, key.replace("_", " ").title())
            value = aggregated_metrics.get(key, "N/A")
            display_metric_card(cols[i], label, value, main_df, reference_df, key, color="#fcec03")

    with st.expander("Export PDF Report", expanded=False):
        season_value = st.session_state.get("selected_season") or "All seasons"
        selected_match_map = dict(zip(match_options["matchId"], match_options["match_label"]))
        selected_match_labels = [selected_match_map.get(mid, str(mid)) for mid in current_selected]
        cover_player_key = f"{player_id}_{player_name}"
        cover_session_key = f"player_report_cover_photo_path_{player_id}"

        if cover_session_key not in st.session_state:
            history = list_cover_photos(base_dir=BASE_DIR, player_key=cover_player_key)
            st.session_state[cover_session_key] = history[0]["path"] if history else None

        if st.button("Manage player cover photo", key="manage_player_report_cover_photo"):
            _pdf_cover_photo_dialog(
                player_key=cover_player_key,
                player_label=player_name,
                session_key=cover_session_key,
                key_prefix=f"player_pdf_cover_{player_id}",
            )

        selected_cover_photo_path = st.session_state.get(cover_session_key)
        if selected_cover_photo_path and os.path.exists(selected_cover_photo_path):
            st.caption("Active player cover photo")
            st.image(selected_cover_photo_path, width=140)
        else:
            st.caption("No player cover photo selected.")

        cover_photo_signature = "none"
        if selected_cover_photo_path and os.path.exists(selected_cover_photo_path):
            cover_photo_signature = f"{selected_cover_photo_path}:{int(os.path.getmtime(selected_cover_photo_path))}"

        report_signature = (
            f"{player_id}|{player_name}|{season_value}|"
            f"{delta_start_date}|{delta_end_date}|{reference_start_date}|{reference_end_date}|"
            f"{comparison_mode_label}|{comparison_value_label}|"
            f"{','.join(str(mid) for mid in current_selected)}|"
            f"{cover_photo_signature}"
        )
        if st.session_state.get("player_report_pdf_signature") != report_signature:
            st.session_state["player_report_pdf_signature"] = report_signature
            st.session_state.pop("player_report_pdf_bytes", None)
            st.session_state.pop("player_report_pdf_name", None)

        generating_key = "player_report_pdf_generating"
        auto_download_key = "player_report_pdf_auto_download_pending"
        if generating_key not in st.session_state:
            st.session_state[generating_key] = False
        if auto_download_key not in st.session_state:
            st.session_state[auto_download_key] = False

        if st.button(
            "Generate & Download PDF report",
            key="generate_player_report_pdf",
            type="primary",
            disabled=st.session_state.get(generating_key, False),
        ):
            st.session_state[generating_key] = True
            st.session_state[auto_download_key] = True
            st.rerun()

        if st.session_state.get(generating_key, False):
            overlay_placeholder = st.empty()
            overlay_placeholder.markdown(
                """
                <style>
                  .pdf-generation-overlay {
                    position: fixed;
                    inset: 0;
                    background: rgba(0, 0, 0, 0.45);
                    z-index: 2147483000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    pointer-events: all;
                  }
                  .pdf-generation-overlay-card {
                    background: #111;
                    color: #fff;
                    padding: 1rem 1.25rem;
                    border-radius: 12px;
                    font-weight: 600;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                  }
                </style>
                <div class="pdf-generation-overlay">
                  <div class="pdf-generation-overlay-card">Generating PDF report... Please wait.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                try:
                    from utils.pdf_generator import generate_player_report
                except ModuleNotFoundError:
                    st.error("Missing dependency for PDF generation. Install `fpdf2` and restart the app.")
                except Exception as e:
                    st.error(f"Could not load PDF generator: {e}")
                else:
                    try:
                        games_played_pdf = int(main_df["matchId"].nunique()) if "matchId" in main_df.columns else int(games_played)
                        if "isFirstEleven" in main_df.columns:
                            games_starter_pdf = int(pd.to_numeric(main_df["isFirstEleven"], errors="coerce").fillna(0).sum())
                        else:
                            games_starter_pdf = int(games_as_starter)
                        if "minutesPlayed" in main_df.columns:
                            minutes_pdf = float(pd.to_numeric(main_df["minutesPlayed"], errors="coerce").fillna(0).sum())
                        else:
                            minutes_pdf = float(total_minutes)

                        player_info_payload = {
                            "age": age if pd.notna(age) else "N/A",
                            "shirtNo": shirt_number if pd.notna(shirt_number) else "N/A",
                            "height": height if pd.notna(height) else "N/A",
                            "weight": weight if pd.notna(weight) else "N/A",
                            "gamesPlayed": games_played_pdf,
                            "gamesStarter": games_starter_pdf,
                            "minutesPlayed": minutes_pdf,
                        }

                        filters_data = {
                            "season": season_value,
                            "start_date": start_date,
                            "end_date": end_date,
                            "delta_start_date": delta_start_date,
                            "delta_end_date": delta_end_date,
                            "reference_start_date": reference_start_date,
                            "reference_end_date": reference_end_date,
                            "comparison_mode": comparison_mode_label,
                            "comparison_value": comparison_value_label,
                            "selected_matches": selected_match_labels,
                        }

                        # Build Trends charts for PDF (same player, selected main matches).
                        trends_data_pdf = []
                        trends_source_df = main_df.copy()
                        if not trends_source_df.empty and "matchDate" in trends_source_df.columns:
                            trends_source_df["matchDate"] = pd.to_datetime(trends_source_df["matchDate"], errors="coerce")
                            trends_source_df = trends_source_df[trends_source_df["matchDate"].notna()].sort_values("matchDate")
                            if "oppositionTeamName" not in trends_source_df.columns:
                                trends_source_df["oppositionTeamName"] = "Unknown"
                            trends_source_df["oppositionTeamName"] = trends_source_df["oppositionTeamName"].astype("string").fillna("Unknown")
                            trends_source_df["opponent_label"] = (
                                trends_source_df["matchDate"].dt.strftime("%b %d") + " - " + trends_source_df["oppositionTeamName"]
                            )

                            for key in metric_keys:
                                if key not in trends_source_df.columns:
                                    continue

                                hover_cols = [c for c in metric_tooltip_fields.get(key, []) if c in trends_source_df.columns]
                                chart_cols = ["opponent_label", "matchDate", key] + hover_cols
                                chart_data = trends_source_df[chart_cols].dropna(subset=["opponent_label", "matchDate", key]).copy()
                                if chart_data.empty:
                                    continue

                                reference_avg = compute_metric_average(reference_df, key)
                                fig = px.bar(
                                    chart_data,
                                    x="opponent_label",
                                    y=key,
                                    title=metric_labels.get(key, key),
                                    color_discrete_sequence=["#fcec03"],
                                    hover_data=hover_cols,
                                    labels={key: metric_labels.get(key, key)},
                                    height=300,
                                )
                                fig.add_hline(
                                    y=reference_avg,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Reference Avg: {reference_avg:.1f}",
                                    annotation_position="top right",
                                )
                                fig.update_layout(
                                    xaxis_title="Match",
                                    yaxis_title=metric_labels.get(key, key),
                                    showlegend=False,
                                    xaxis_tickangle=-45,
                                )
                                trends_data_pdf.append({"kpi_name": metric_labels.get(key, key), "fig": fig})

                        safe_name = _safe_pdf_filename(player_name)
                        file_name = f"{safe_name}_player_report.pdf"

                        with st.spinner("Generating PDF report..."):
                            pdf_bytes = generate_player_report(
                                player_name=player_name,
                                player_info=player_info_payload,
                                player_position=player_position,
                                aggregated_metrics=aggregated_metrics,
                                filtered_df=main_df,
                                filters_data=filters_data,
                                logo_path=LOGO_PATH,
                                calculate_delta_func=calculate_delta,
                                full_df=metrics_summary,
                                reference_df=reference_df,
                                position_kpi_map=position_kpi_map,
                                trends_data=trends_data_pdf,
                                comparison_data=None,
                                comparison_kpi_table=None,
                                comparison_charts=None,
                                background_image_path=BACKGROUND_COVER_PATH if os.path.exists(BACKGROUND_COVER_PATH) else None,
                                player_photo_path=selected_cover_photo_path if (selected_cover_photo_path and os.path.exists(selected_cover_photo_path)) else None,
                            )

                        if not pdf_bytes:
                            raise ValueError("Generated PDF is empty.")

                        st.session_state["player_report_pdf_bytes"] = pdf_bytes
                        st.session_state["player_report_pdf_name"] = file_name
                        st.success("PDF generated. Download should start automatically.")
                    except Exception as e:
                        st.session_state[auto_download_key] = False
                        st.error(f"Failed to generate PDF: {e}")
            finally:
                st.session_state[generating_key] = False
                overlay_placeholder.empty()

        if st.session_state.get(auto_download_key, False) and st.session_state.get("player_report_pdf_bytes"):
            pdf_b64 = base64.b64encode(st.session_state["player_report_pdf_bytes"]).decode("utf-8")
            file_name = st.session_state.get("player_report_pdf_name", "player_report.pdf")
            components.html(
                f"""
                <script>
                  (function() {{
                    const a = document.createElement('a');
                    a.href = 'data:application/pdf;base64,{pdf_b64}';
                    a.download = '{file_name}';
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                  }})();
                </script>
                """,
                height=0
            )
            st.session_state[auto_download_key] = False
            st.caption("If download does not start automatically, use the fallback button below.")

        if st.session_state.get("player_report_pdf_bytes"):
            st.download_button(
                "Download PDF report (fallback)",
                data=st.session_state["player_report_pdf_bytes"],
                file_name=st.session_state.get("player_report_pdf_name", "player_report.pdf"),
                mime="application/pdf",
                key="download_player_report_pdf",
            )

    # --- Full Stats Table ---
    display_df = selected_matches_df.copy()
    display_df = display_df.drop(columns=["match_date_only", "is_main_period", "is_reference_period"], errors="ignore")
    display_df = display_df.rename(columns={"period_tag": "Usage"})

    # Format KPI columns like the scorecards
    for col in selected_kpis:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: format_metric_value(x, col))

    # Optional: sort and reset
    display_df = display_df.sort_values("matchDate").reset_index(drop=True)

    # Style configuration for table
    def format_stat_table(df: pd.DataFrame) -> Styler:
        return df.style.set_properties(**{
            'text-align': 'center',
            'background-color': '#f9f9f9',
            'color': '#000000',
            'font-size': '14px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('color', '#fcec03'), ('font-weight', 'bold'), ('text-align', 'center')]}
        ])

    # Custom header style
    st.markdown("""
        <style>
        .streamlit-expanderHeader {
            font-size: 1.7rem !important;
            font-weight: bold;
            color: #fcec03 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display table
    with st.expander("Player Stats"):
        st.dataframe(format_stat_table(display_df), use_container_width=True)



elif section == "Trends Stats":
    st.info("Performance Trends Over Time")

    # Ensure team names are present
    if "oppositionTeamName" not in filtered_df.columns:
        teams_info = (
            event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
            .first()
            .reset_index()
        )
        trends_df = filtered_df.merge(teams_info, on="matchId", how="left")
    else:
        trends_df = filtered_df.copy()

    # Sort by matchDate
    trends_df = trends_df.sort_values("matchDate").copy()

    # Create opponent_label once
    trends_df["opponent_label"] = trends_df["matchDate"].dt.strftime("%b %d") + " - " + trends_df["oppositionTeamName"]

    # Create match_order
    trends_df["match_order"] = range(len(trends_df))

    # Save order dictionary
    match_order_dict = dict(zip(trends_df["opponent_label"], trends_df["match_order"]))

    # --- Match Filter ---
    with st.expander("Filter by Match (click to hide)", expanded=True):
        match_options = trends_df[["matchId", "opponent_label", "matchDate"]].drop_duplicates().sort_values("matchDate")
        all_match_ids = list(match_options["matchId"])
        state_key = "selected_match_ids_trends"

        search_text = st.text_input("🔍 Search match:", "", key="match_search_trends")
        filtered_options = match_options[
            match_options["opponent_label"].str.contains(search_text, case=False, na=False)
        ].copy()
        visible_match_ids = list(filtered_options["matchId"])

        if state_key not in st.session_state:
            st.session_state[state_key] = all_match_ids
        else:
            st.session_state[state_key] = [mid for mid in st.session_state[state_key] if mid in all_match_ids]

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Select All Matches", key="select_all_matches_trends"):
                st.session_state[state_key] = all_match_ids
                for mid in all_match_ids:
                    st.session_state[f"trends_match_{mid}"] = True
        with col2:
            if st.button("Clear Matches", key="clear_matches_trends"):
                st.session_state[state_key] = []
                for mid in all_match_ids:
                    st.session_state[f"trends_match_{mid}"] = False

        selected_ids = set(st.session_state[state_key])
        selected_visible_ids = []
        visible_set = set(visible_match_ids)

        st.markdown("<div style='max-height: 250px; overflow-y: auto; padding: 0 10px;'>", unsafe_allow_html=True)
        for _, row in filtered_options.iterrows():
            mid = row["matchId"]
            label = row["opponent_label"]
            cb_key = f"trends_match_{mid}"
            if cb_key not in st.session_state:
                st.session_state[cb_key] = mid in selected_ids
            checked = st.checkbox(label, key=cb_key)
            if checked:
                selected_visible_ids.append(mid)
        st.markdown("</div>", unsafe_allow_html=True)

        hidden_selected = [mid for mid in st.session_state[state_key] if mid not in visible_set]
        updated_set = set(hidden_selected) | set(selected_visible_ids)
        st.session_state[state_key] = [mid for mid in all_match_ids if mid in updated_set]

    # Apply match filter (but don't re-create opponent_label!)
    trends_df = trends_df[trends_df["matchId"].isin(st.session_state.get("selected_match_ids_trends", []))]

    # --- Now Plot ---
    for key in metric_keys:
        chart_data = trends_df[["opponent_label", "matchDate", key] + metric_tooltip_fields.get(key, [])].dropna().copy()

        if chart_data.empty:
            continue

        season_avg = metrics_summary[key].mean()

        # Create the bar chart
        fig = px.bar(
            chart_data,
            x="opponent_label",
            y=key,
            title=metric_labels[key],
            color_discrete_sequence=["#fcec03"],  # Yellow color
            hover_data=metric_tooltip_fields.get(key, []),
            labels={key: metric_labels[key]},
            height=300
        )
            
        # Add season average line
        fig.add_hline(
            y=season_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {season_avg:.1f}",
            annotation_position="top right"
        )
            
        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Match",
            yaxis_title=metric_labels[key],
            showlegend=False,
            xaxis_tickangle=-45
        )
            
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Player Comparison (season-aware)
# -----------------------------
elif section == "Player Comparison":

    # --- Season from sidebar (Option A) ---
    selected_season = st.session_state.get("selected_season", None)
    season_label = selected_season if selected_season else "All seasons"

    # Keep this consistent with what's actually stored in DB (already lower-cased in WHERE)
    COMPETITION = "championship"

    st.info(f"Top Players in the Competition – **{player_position}** (Season: {season_label})")

    # ---------- Normalize team_data schema ----------
    team_data_cmp = team_data.rename(columns={
        "match_id": "matchId",
        "team_id": "teamId",
        "team_name": "teamName",
    }).copy()

    def to_int64_id(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")

    if "matchId" not in team_data_cmp.columns or "teamId" not in team_data_cmp.columns:
        st.error("team_data missing matchId/teamId (even after renaming).")
        st.stop()

    team_data_cmp["matchId"] = to_int64_id(team_data_cmp["matchId"])
    team_data_cmp["teamId"]  = to_int64_id(team_data_cmp["teamId"])

    # ---------- Load matches (season-aware) ----------
    engine = connect_to_db()

    if selected_season:
        season_matches = pd.read_sql(
            """
            SELECT matchId, startDate, score, ftScore, season, competition
            FROM match_data
            WHERE season = %s AND LOWER(competition) = %s
            """,
            con=engine, params=(selected_season, COMPETITION)
        )
        if season_matches.empty:
            st.error(f"No matches found for season '{selected_season}' in competition '{COMPETITION}'.")
            st.stop()
    else:
        # All seasons: load by competition only; date range still applied later
        season_matches = pd.read_sql(
            """
            SELECT matchId, startDate, score, ftScore, season, competition
            FROM match_data
            WHERE LOWER(competition) = %s
            """,
            con=engine, params=(COMPETITION,)
        )
        if season_matches.empty:
            st.error(f"No matches found in competition '{COMPETITION}'.")
            st.stop()

    if "matchId" not in season_matches.columns:
        st.error("match_data has no matchId column.")
        st.stop()

    season_matches["matchId"]   = to_int64_id(season_matches["matchId"])
    season_matches["startDate"] = pd.to_datetime(season_matches["startDate"], errors="coerce")

    # ---------- Extract goals (prefer ftScore, then score) ----------
    def attach_goals(df):
        df = df.copy()
        df["home_goals"] = pd.NA
        df["away_goals"] = pd.NA

        def try_parse(col):
            if col not in df.columns:
                return False
            s = df[col].astype(str).str.strip().str.replace(" ", "", regex=False)
            ok = s.str.contains(r"^\d+:\d+$", na=False)
            if not ok.any():
                return False
            split = s.where(ok).str.split(":", expand=True)
            if split.shape[1] < 2:
                return False
            df.loc[ok, "home_goals"] = pd.to_numeric(split[0], errors="coerce")
            df.loc[ok, "away_goals"] = pd.to_numeric(split[1], errors="coerce")
            return True

        parsed = try_parse("ftScore")
        if not parsed:
            parsed = try_parse("score")

        if not parsed:
            st.warning("No valid scores parsed from ftScore/score — points may be partial/zero.")
        return df

    season_matches = attach_goals(season_matches)

    # ---------- Filter team_data by the (season’s) matchIds ----------
    filtered_match_ids = season_matches["matchId"].dropna().unique().tolist()
    if not filtered_match_ids:
        st.error("No season match IDs available after normalization.")
        st.stop()

    team_data_filtered = team_data_cmp[team_data_cmp["matchId"].isin(filtered_match_ids)].copy()
    if team_data_filtered.empty:
        st.error("team_data has no rows for these matches.")
        st.stop()

    # Merge scores into team_data
    team_scores_df = pd.merge(
        team_data_filtered,
        season_matches[["matchId", "home_goals", "away_goals", "score", "ftScore"]],
        on="matchId",
        how="inner"
    )

    # Infer home/away by row order within matchId (assumes exactly 2 rows per match)
    team_scores_df["team_order"] = team_scores_df.groupby("matchId").cumcount()
    team_scores_df["home_away"] = team_scores_df["team_order"].map({0: "home", 1: "away"})

    # Points (skip rows with missing goals)
    def assign_points(row):
        hg, ag = row["home_goals"], row["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            return 0
        if row["home_away"] == "home":
            return 3 if hg > ag else 1 if hg == ag else 0
        else:
            return 3 if ag > hg else 1 if ag == hg else 0

    team_scores_df["points"] = team_scores_df.apply(assign_points, axis=1)

    team_points = (
        team_scores_df
        .groupby(["teamId", "teamName"], as_index=False)["points"]
        .sum()
        .sort_values("points", ascending=False)
    )

    if team_points.empty:
        st.error("Could not compute team points for the selected filters.")
        st.stop()

    # Top-5 defaults (season-scoped if season set)
    top_5_teams      = team_points.head(5)
    top_team_ids     = top_5_teams["teamId"].tolist()
    top_5_team_names = top_5_teams["teamName"].tolist()

    # ---------- Real Position codes ----------
    reverse_position_map = {
        "Goalkeeper": ["GK"],
        "Center Back": ["DC", "CB"],
        "Left Back": ["DL", "DML", "LB", "LWB"],
        "Right Back": ["DR", "DMR", "RB", "RWB"],
        "Defensive Midfielder": ["DMC", "DM"],
        "Midfielder": ["MC", "ML", "MR", "CM", "LM", "RM"],
        "Attacking Midfielder": ["AMC", "AM"],
        "Left Winger": ["AML", "FWL", "LW"],
        "Right Winger": ["AMR", "FWR", "RW"],
        "Striker": ["FW", "ST", "CF"]
    }

    position_codes = reverse_position_map.get(player_position, [])
    if not position_codes:
        st.error(f"❌ No position codes found for player position: {player_position}")
        st.stop()

    st.write(f"Position codes resolved for **{player_position}** → {position_codes}")

    # ---------- Team Filter UI (ordered by points, top 5 pre-selected) ----------
    with st.expander("Filter by Teams", expanded=False):
        all_team_names = team_points["teamName"].tolist()  # already points-ordered

        search_team = st.text_input("🔍 Search team:", "", key="search_team")
        filtered_team_options = [t for t in all_team_names if search_team.lower() in t.lower()]

        if "selected_team_names" not in st.session_state:
            st.session_state.selected_team_names = top_5_team_names or all_team_names[:5]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All Teams"):
                st.session_state.selected_team_names = all_team_names
        with col2:
            if st.button("Clear Teams"):
                st.session_state.selected_team_names = []

        selected_team_names = st.session_state.get("selected_team_names", top_5_team_names)
        new_selection = []

        st.markdown("<div style='max-height: 250px; overflow-y: auto;'>", unsafe_allow_html=True)
        for i, team in enumerate(filtered_team_options):
            checked = team in selected_team_names
            if st.checkbox(team, value=checked, key=f"team_{team}_{i}"):
                new_selection.append(team)
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.selected_team_names = new_selection

        # Match back to IDs
        team_points["teamName_clean"] = team_points["teamName"].str.strip().str.lower()
        selected_team_names_clean = [name.strip().lower() for name in new_selection]
        selected_team_ids = team_points[team_points["teamName_clean"].isin(selected_team_names_clean)]["teamId"].tolist()

    # ---------- Player Filter (season + position + selected teams) ----------
    if not selected_team_ids:
        st.warning("No teams selected.")
        st.stop()

    selected_team_ids_str = ",".join(str(int(tid)) for tid in selected_team_ids if pd.notna(tid))

    # Build season-aware players query
  # --- Query players in scope (season + competition + position + teams + date window)
    # ---------------------------
    # 1) LOAD RAW TABLES (no joins)
# ---------------------------
# PLAYER COMPARISON — LOAD & SUMMARIZE DATA
# (Mirrors load_player_data logic exactly)
# ---------------------------

    # ---------------------------
# PLAYER COMPARISON — LOAD & SUMMARIZE DATA
# ---------------------------

    # ---------------------------
# PLAYER COMPARISON — LOAD & SUMMARIZE DATA
# ---------------------------

    conn = connect_to_db()

    # --- 1) LOAD RAW TABLES ---
    pd_cols = [
        "playerId","playerName","age","shirtNo","height","weight",
        "teamId","isFirstEleven","matchId","position",
        "subbedInExpandedMinute","subbedOutExpandedMinute"
    ]
    md_cols = ["matchId","startDate","competition","season"]
    td_cols = ["team_id","team_name","match_id"]

    match_data_raw  = pd.read_sql(f"SELECT {', '.join(md_cols)} FROM match_data", conn)
    player_data_raw = pd.read_sql(f"SELECT {', '.join(pd_cols)} FROM player_data", conn)
    team_data_raw   = pd.read_sql(f"SELECT {', '.join(td_cols)} FROM team_data", conn)

    # --- Normalize team_data column names ---
    team_data_raw = team_data_raw.rename(columns={
        "team_id": "teamId",
        "team_name": "teamName",
        "match_id": "matchId"
    })

    # ---------------------------
    # 2) NORMALIZE DTYPES
    # ---------------------------
    def to_int64(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    to_int64(player_data_raw, ["playerId","teamId","matchId"])
    to_int64(match_data_raw,  ["matchId"])
    to_int64(team_data_raw,   ["teamId","matchId"])

    # Dates
    if "startDate" in match_data_raw.columns:
        match_data_raw["startDate"] = pd.to_datetime(match_data_raw["startDate"], errors="coerce")

    # Competition lowercase
    if "competition" in match_data_raw.columns:
        match_data_raw["competition_l"] = match_data_raw["competition"].astype(str).str.lower()
    COMPETITION_l = str(COMPETITION).lower()

    # Normalize isFirstEleven
    player_data_raw["isFirstEleven"] = (
        player_data_raw["isFirstEleven"]
        .replace({"True": 1, "False": 0, True: 1, False: 0})
    )
    player_data_raw["isFirstEleven"] = pd.to_numeric(player_data_raw["isFirstEleven"], errors="coerce").fillna(0).astype(int)

    # ---------------------------
    # 3) COMPUTE MINUTES PLAYED (same helper as logged-in player)
    # ---------------------------
    from db_utils import prepare_player_data_with_minutes

    player_data_raw = prepare_player_data_with_minutes(player_data_raw)

    if "minutesPlayed" not in player_data_raw.columns:
        st.error("❌ prepare_player_data_with_minutes() did not add 'minutesPlayed'. Check helper function.")

    players_full = player_data_raw.merge(
        match_data_raw[match_data_raw["competition_l"] == COMPETITION_l][["matchId", "startDate", "season"]],
        on="matchId",
        how="left",
        validate="m:1"
    )

    if selected_season:
        players_full = players_full[players_full["season"] == selected_season]

    players_full = players_full[
        players_full["startDate"].notna() &
        (players_full["startDate"].dt.date >= start_date) &
        (players_full["startDate"].dt.date <= end_date)
    ].copy()

    # Merge team names
    td_names = team_data_raw[["teamId", "teamName"]].drop_duplicates(subset=["teamId"])
    players_full = players_full.merge(td_names, on="teamId", how="left")

    if players_full["startDate"].isna().all():
        st.warning("⚠️ No startDate merged from match_data — check matchId alignment or competition filter.")

    # Apply selected teams and selected position (same role as logged-in player).
    selected_team_ids_set = set(pd.to_numeric(pd.Series(selected_team_ids), errors="coerce").dropna().astype("Int64").tolist())
    players_scope_all = players_full[players_full["teamId"].isin(selected_team_ids_set)].copy()

    position_codes_set = {str(code).strip().upper() for code in position_codes}
    players_scope_all["position"] = players_scope_all["position"].astype("string").str.strip().str.upper()
    players_full = players_scope_all[players_scope_all["position"].isin(position_codes_set)].copy()

    if players_full.empty:
        st.warning("No players found for the selected teams and position.")
        st.stop()

    # Team-level denominator for minutes % (within current season/date/team filters).
    team_match_counts = (
        players_scope_all[["teamId", "matchId"]]
        .dropna(subset=["teamId", "matchId"])
        .drop_duplicates()
        .groupby("teamId", as_index=False)["matchId"]
        .nunique()
        .rename(columns={"matchId": "team_matches_in_scope"})
    )

    # ---------------------------
    # 5) DEDUPE TO ONE ROW PER PLAYER-MATCH
    # ---------------------------
    players_full = players_full.groupby(["playerId","playerName","teamId","teamName","matchId","startDate"], as_index=False).agg(
            age=("age","first"),
            shirtNo=("shirtNo","first"),
            height=("height","first"),
            weight=("weight","first"),
            isFirstEleven=("isFirstEleven","max"),
            minutesPlayed=("minutesPlayed","max")
        ).reset_index(drop=True)

    players_full = players_full.sort_values(["playerId","startDate"]).reset_index(drop=True)

    # ---------------------------
    # 6) PLAYER PICKER
    # ---------------------------
    player_index = (
        players_full.groupby(["playerId","playerName","teamId","teamName"], as_index=False)
        .agg(minutes=("minutesPlayed","sum"))
        .sort_values("minutes", ascending=False)
    )

    player_index = player_index.merge(team_match_counts, on="teamId", how="left")
    player_index["team_matches_in_scope"] = pd.to_numeric(
        player_index["team_matches_in_scope"], errors="coerce"
    ).fillna(0)
    player_index["minutes_pct_total"] = np.where(
        player_index["team_matches_in_scope"] > 0,
        (player_index["minutes"] / (player_index["team_matches_in_scope"] * 90.0)) * 100.0,
        np.nan,
    )
    player_index["minutes_pct_total"] = (
        pd.to_numeric(player_index["minutes_pct_total"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
        .round(1)
    )

    logged_player_id = str(player_id)

    with st.expander("Player Selector", expanded=True):
        max_players_to_show = st.number_input(
            "Max players to show",
            min_value=2,
            max_value=60,
            value=20,
            step=1,
            key="comparison_max_players_to_show",
            help="Limits the player list after applying team, position and minutes filters.",
        )
        min_minutes_pct = st.number_input(
            "Minimum minutes played (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            key="comparison_min_minutes_pct",
            help="Exclude players with low participation in the selected date range.",
        )
        use_per90_for_non_pct = st.toggle(
            "Use per 90 for non-% KPIs",
            value=True,
            key="comparison_use_per90_non_pct",
            help="When enabled, all non-percentage KPIs in comparison are normalized per 90 minutes.",
        )

        eligible_player_index = player_index[
            (player_index["minutes_pct_total"] >= float(min_minutes_pct))
            | (player_index["playerId"].astype(str) == logged_player_id)
        ].copy()
        eligible_player_index = eligible_player_index.sort_values(
            by=["minutes_pct_total", "minutes"],
            ascending=[False, False],
        )

        if eligible_player_index.empty:
            st.warning("No players match the selected minimum minutes percentage.")
            st.stop()

        limited_player_index = eligible_player_index.head(int(max_players_to_show)).copy()
        if (logged_player_id in eligible_player_index["playerId"].astype(str).values) and (
            logged_player_id not in limited_player_index["playerId"].astype(str).values
        ):
            logged_row = eligible_player_index[eligible_player_index["playerId"].astype(str) == logged_player_id].head(1)
            limited_player_index = pd.concat(
                [logged_row, limited_player_index.head(max(0, int(max_players_to_show) - 1))],
                ignore_index=True,
            ).drop_duplicates(subset=["playerId"])

        options_ids = limited_player_index["playerId"].astype(str).tolist()
        default_ids = limited_player_index.head(min(5, int(max_players_to_show)))["playerId"].astype(str).tolist()

        if (logged_player_id in options_ids) and (logged_player_id not in default_ids):
            default_ids = ([logged_player_id] + [pid for pid in default_ids if pid != logged_player_id])[:5]

        label_map = {
            str(r.playerId): (
                f"{r.playerName} ({r.teamName}) - {float(r.minutes_pct_total):.1f}% min"
                if pd.notna(r.teamName) and r.teamName != ""
                else f"{r.playerName} - {float(r.minutes_pct_total):.1f}% min"
            )
            for _, r in limited_player_index.iterrows()
        }

        selected_player_ids = st.multiselect(
            "Select players to compare",
            options=options_ids,
            default=default_ids,
            format_func=lambda pid: label_map.get(str(pid), str(pid))
        )

    filtered_players = players_full[players_full["playerId"].astype(str).isin([str(x) for x in selected_player_ids])].copy()
    if filtered_players.empty:
        st.warning("No player/match rows left after filters.")
        st.stop()

    # ---------------------------
    # 7) SUMMARY (same as logged-in)
    # ---------------------------
    summary_comparison_df = (
        filtered_players.groupby(["playerId","playerName","teamId","teamName"], as_index=False)
        .agg(
            age=("age","first"),
            shirtNo=("shirtNo","first"),
            height=("height","first"),
            weight=("weight","first"),
            matches_played=("matchId","nunique"),
            games_as_starter=("isFirstEleven","sum"),
            total_minutes=("minutesPlayed","sum"),
        )
    )

    summary_comparison_df = summary_comparison_df.merge(team_match_counts, on="teamId", how="left")
    summary_comparison_df["team_matches_in_scope"] = pd.to_numeric(
        summary_comparison_df["team_matches_in_scope"], errors="coerce"
    ).fillna(0)
    summary_comparison_df["minutes_pct_total"] = np.where(
        summary_comparison_df["team_matches_in_scope"] > 0,
        (summary_comparison_df["total_minutes"] / (summary_comparison_df["team_matches_in_scope"] * 90.0)) * 100.0,
        np.nan,
    )
    summary_comparison_df["minutes_pct_total"] = (
        pd.to_numeric(summary_comparison_df["minutes_pct_total"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
        .round(1)
    )

    summary_display = summary_comparison_df.rename(columns={
        "playerName":"Player",
        "teamName":"Team",
        "age":"Age",
        "shirtNo":"Shirt No",
        "height":"Height",
        "weight":"Weight",
        "matches_played":"Games Played",
        "games_as_starter":"Games as Starter",
        "total_minutes":"Minutes Played",
        "minutes_pct_total":"% Total Minutes"
    }).sort_values(by="Games Played", ascending=False)

    summary_table_columns = [
        "Player",
        "Team",
        "Age",
        "Shirt No",
        "Height",
        "Weight",
        "Games Played",
        "Games as Starter",
        "Minutes Played",
        "% Total Minutes",
    ]
    player_list_table_placeholder = st.empty()
    player_list_table_placeholder.dataframe(
        summary_display[summary_table_columns],
        use_container_width=True,
    )


    # --- Prepare IDs for SQL queries
    player_ids = filtered_players["playerId"].unique().tolist()
    match_ids = filtered_players["matchId"].unique().tolist()
    player_placeholders = ",".join(["%s"] * len(player_ids))
    match_placeholders = ",".join(["%s"] * len(match_ids))

    # --- fetch stats & events
    query_stats = f"""
        SELECT * FROM player_stats
        WHERE playerId IN ({player_placeholders})
        AND matchId  IN ({match_placeholders})
    """
    stats_df = pd.read_sql(query_stats, con=engine, params=tuple(player_ids + match_ids))

    query_events = f"""
        SELECT * FROM event_data
        WHERE playerId IN ({player_placeholders})
        AND matchId  IN ({match_placeholders})
    """
    events_df = pd.read_sql(query_events, con=engine, params=tuple(player_ids + match_ids))

    # Normalize IDs
    for df_ in (stats_df, events_df):
        df_["playerId"] = df_["playerId"].astype(str)
        df_["matchId"]  = df_["matchId"].astype(str)

    # Logged-in player's metrics (already built earlier in your app)
    logged_player_metrics = metrics_summary[
        (metrics_summary["matchDate"] >= pd.to_datetime(start_date)) &
        (metrics_summary["matchDate"] <= pd.to_datetime(end_date))
    ].copy()
    logged_player_id = str(player_id)
    logged_player_name = player_name
    logged_player_metrics["playerId"] = logged_player_id
    logged_player_metrics["playerId"] = logged_player_metrics["playerId"].astype(str)

    # Comparison players (exclude logged-in)
    comparison_stats  = stats_df[stats_df["playerId"] != logged_player_id].copy()
    comparison_events = events_df[events_df["playerId"] != logged_player_id].copy()

    comparison_metrics = process_player_comparison_metrics(
        comparison_stats, comparison_events, player_position
    )

    # ---- Combine + enrich labels
    all_metrics_df = pd.concat([logged_player_metrics, comparison_metrics], ignore_index=True)
    all_metrics_df["playerId"] = all_metrics_df["playerId"].astype(str)
    if "matchId" in all_metrics_df.columns:
        all_metrics_df["matchId"] = all_metrics_df["matchId"].astype(str)

    # Build a name/team lookup (prefer summary_comparison_df; else fallback)
    if {"playerId","playerName","teamName"}.issubset(summary_comparison_df.columns):
        name_lookup = (
            summary_comparison_df[["playerId","playerName","teamName"]]
            .drop_duplicates(subset=["playerId"])
            .copy()
        )
    else:
        name_lookup = (
            players_full[["playerId","playerName","teamId"]]
            .drop_duplicates(subset=["playerId"])
            .merge(
                team_data_filtered[["teamId","teamName"]].drop_duplicates(),
                on="teamId", how="left"
            )[["playerId","playerName","teamName"]]
            .copy()
        )
    name_lookup["playerId"] = name_lookup["playerId"].astype(str)

    # Drop any existing name/team cols to avoid suffixes, then merge
    for col in ("playerName","teamName"):
        if col in all_metrics_df.columns:
            all_metrics_df.drop(columns=[col], inplace=True)

    all_metrics_df = all_metrics_df.merge(name_lookup, on="playerId", how="left")

    # Attach minutesPlayed per (playerId, matchId) so per-90 calculations are reliable.
    minutes_sources = []
    if {"playerId", "matchId", "minutesPlayed"}.issubset(filtered_players.columns):
        minutes_sources.append(filtered_players[["playerId", "matchId", "minutesPlayed"]].copy())
    if {"playerId", "matchId", "minutesPlayed"}.issubset(filtered_logged_player_info.columns):
        minutes_sources.append(filtered_logged_player_info[["playerId", "matchId", "minutesPlayed"]].copy())

    if minutes_sources:
        minutes_lookup = pd.concat(minutes_sources, ignore_index=True)
        minutes_lookup["playerId"] = minutes_lookup["playerId"].astype(str)
        minutes_lookup["matchId"] = minutes_lookup["matchId"].astype(str)
        minutes_lookup["minutesPlayed"] = pd.to_numeric(minutes_lookup["minutesPlayed"], errors="coerce").fillna(0.0)
        minutes_lookup = (
            minutes_lookup.groupby(["playerId", "matchId"], as_index=False)["minutesPlayed"]
            .max()
        )

        all_metrics_df = all_metrics_df.merge(
            minutes_lookup,
            on=["playerId", "matchId"],
            how="left",
            suffixes=("", "_lookup"),
        )
        if "minutesPlayed_lookup" in all_metrics_df.columns:
            base_minutes = pd.to_numeric(all_metrics_df.get("minutesPlayed"), errors="coerce")
            lookup_minutes = pd.to_numeric(all_metrics_df["minutesPlayed_lookup"], errors="coerce")
            all_metrics_df["minutesPlayed"] = base_minutes.where(base_minutes.notna(), lookup_minutes)
            all_metrics_df.drop(columns=["minutesPlayed_lookup"], inplace=True)

    # ---- Ensure the logged-in player is present
    mask_logged = all_metrics_df["playerId"] == logged_player_id
    if not mask_logged.any():
        pass
        # Add a zero row so the logged-in player appears in comparisons
        zero_row = {c: 0 for c in all_metrics_df.columns if c not in ("playerId","playerName","teamName")}
        zero_row.update({"playerId": logged_player_id, "playerName": logged_player_name})
        all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([zero_row])], ignore_index=True)
        mask_logged = all_metrics_df["playerId"] == logged_player_id

    # Fill missing name/team for logged player
    all_metrics_df.loc[mask_logged, "playerName"] = all_metrics_df.loc[mask_logged, "playerName"].fillna(logged_player_name)
    if all_metrics_df.loc[mask_logged, "teamName"].isna().any():
        try:
            logged_team_name = (
                player_data[player_data["playerId"].astype(str) == logged_player_id]["teamName"].iloc[0]
                if "teamName" in player_data.columns and not player_data.empty else ""
            )
        except Exception:
            logged_team_name = ""
        all_metrics_df.loc[mask_logged, "teamName"] = all_metrics_df.loc[mask_logged, "teamName"].fillna(logged_team_name)

    # ---- Aggregate KPIs per player
    grouped = all_metrics_df.groupby("playerId")
    summary_metrics_df = grouped[["playerName","teamName"]].first().reset_index()
    all_metrics_df["_minutesPlayed_numeric"] = pd.to_numeric(
        all_metrics_df.get("minutesPlayed", pd.Series(index=all_metrics_df.index, dtype=float)),
        errors="coerce",
    )
    # Fallback to full-match minutes if a row still has no minutesPlayed.
    fallback_minutes = all_metrics_df.groupby("playerId")["matchId"].transform("nunique").astype(float) * 90.0
    all_metrics_df["_minutesPlayed_numeric"] = (
        all_metrics_df["_minutesPlayed_numeric"]
        .fillna(fallback_minutes)
        .clip(lower=0.0)
    )
    player_total_minutes = (
        all_metrics_df.groupby("playerId")["_minutesPlayed_numeric"].sum().replace(0, np.nan)
    )

    for kpi in selected_kpis:
        metric_type = metric_type_map.get(kpi, "aggregate")
        if metric_type == "percentage":
            numerator, denominator = percentage_formula_map.get(kpi, (None, None))
            if numerator in all_metrics_df.columns and denominator in all_metrics_df.columns:
                sum_num = grouped[numerator].sum()
                sum_den = grouped[denominator].sum().replace(0, np.nan)
                weighted_avg = (sum_num / sum_den) * 100
                summary_metrics_df[kpi] = summary_metrics_df["playerId"].map(weighted_avg.round(1))
            else:
                summary_metrics_df[kpi] = np.nan
        else:
            if kpi in all_metrics_df.columns:
                total = pd.to_numeric(grouped[kpi].sum(), errors="coerce")
                if use_per90_for_non_pct:
                    per90 = (total / player_total_minutes) * 90.0
                    summary_metrics_df[kpi] = summary_metrics_df["playerId"].map(per90.round(2))
                else:
                    summary_metrics_df[kpi] = summary_metrics_df["playerId"].map(total.round(1))
            else:
                summary_metrics_df[kpi] = np.nan

    # Tooltips
    all_tooltip_fields = set()
    for fields in metric_tooltip_fields.values():
        all_tooltip_fields.update(fields)

    for tooltip_col in all_tooltip_fields:
        if tooltip_col in all_metrics_df.columns:
            summary_metrics_df[tooltip_col] = summary_metrics_df["playerId"].map(grouped[tooltip_col].sum().round(1))
        else:
            summary_metrics_df[tooltip_col] = np.nan

    summary_metrics_df = summary_metrics_df.fillna(0)

    # Equal-weight performance score based on selected KPIs (min-max normalized per KPI).
    perf_kpis = [k for k in selected_kpis if k in summary_metrics_df.columns]
    if perf_kpis:
        perf_base = summary_metrics_df[perf_kpis].apply(pd.to_numeric, errors="coerce")
        perf_norm = pd.DataFrame(index=perf_base.index)
        for col in perf_kpis:
            col_vals = perf_base[col]
            col_min = col_vals.min(skipna=True)
            col_max = col_vals.max(skipna=True)
            if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
                perf_norm[col] = 50.0
            else:
                perf_norm[col] = ((col_vals - col_min) / (col_max - col_min)) * 100.0

        summary_metrics_df["Performance Score"] = (
            perf_norm.mean(axis=1).fillna(0.0).clip(lower=0.0, upper=100.0).round(1)
        )
    else:
        summary_metrics_df["Performance Score"] = 0.0

    # Update the players list table with the computed Performance KPI.
    if "playerId" in summary_display.columns:
        performance_lookup = summary_metrics_df[["playerId", "Performance Score"]].copy()
        performance_lookup["playerId"] = performance_lookup["playerId"].astype(str)
        summary_display["playerId"] = summary_display["playerId"].astype(str)
        summary_display = summary_display.merge(performance_lookup, on="playerId", how="left")
        summary_display["Performance Score"] = pd.to_numeric(
            summary_display["Performance Score"], errors="coerce"
        ).fillna(0.0).round(1)
        player_list_table_placeholder.dataframe(
            summary_display[summary_table_columns + ["Performance Score"]],
            use_container_width=True,
        )

    st.info("Performance Score (Equal-Weight KPI Mean)")
    if use_per90_for_non_pct:
        st.caption(
            "Performance Score is the mean of all selected KPIs with equal weight, using per-90 values for non-% KPIs "
            "and weighted ratios for % KPIs. The dashed red line shows the mean of selected players."
        )
    else:
        st.caption(
            "Performance Score is the mean of all selected KPIs with equal weight, using totals for non-% KPIs "
            "and weighted ratios for % KPIs. The dashed red line shows the mean of selected players."
        )
    performance_chart_data = summary_metrics_df[["playerName", "Performance Score"]].copy().sort_values(
        by="Performance Score",
        ascending=False,
    )
    performance_chart_data["color"] = performance_chart_data["playerName"].apply(
        lambda name: "#FFD700" if name == player_name else "#d3d3d3"
    )
    fig_perf = px.bar(
        performance_chart_data,
        x="playerName",
        y="Performance Score",
        title="Performance Score",
        color="color",
        color_discrete_map="identity",
        hover_data=["playerName", "Performance Score"],
        labels={"Performance Score": "Performance Score"},
        height=320,
    )
    performance_mean = performance_chart_data["Performance Score"].mean()
    fig_perf.add_hline(
        y=performance_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean of selected players: {performance_mean:.2f}",
        annotation_position="top right",
    )
    fig_perf.update_yaxes(range=[0, 100])
    fig_perf.update_layout(xaxis_title="Player", yaxis_title="Performance Score", showlegend=False)
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- Charts ---
    if use_per90_for_non_pct:
        st.info("Comparison by KPI (Per 90 for non-% metrics)")
    else:
        st.info("Comparison by KPI")

    for kpi in selected_kpis:
        if kpi not in summary_metrics_df.columns:
            continue

        is_percentage_metric = metric_type_map.get(kpi) == "percentage"
        kpi_display_label = metric_labels.get(kpi, kpi)
        if use_per90_for_non_pct and not is_percentage_metric:
            kpi_display_label = f"{kpi_display_label} (per 90)"

        chart_data = summary_metrics_df[["playerName", kpi]].copy().sort_values(by=kpi, ascending=False)
        chart_data["color"] = chart_data["playerName"].apply(
            lambda name: "#FFD700" if name == player_name else "#d3d3d3"
        )

        tooltip_fields = ["playerName", kpi] + metric_tooltip_fields.get(kpi, [])
        for col in tooltip_fields:
            if col not in chart_data.columns:
                chart_data[col] = np.nan

        fig = px.bar(
            chart_data,
            x="playerName",
            y=kpi,
            title=kpi_display_label,
            color="color",
            color_discrete_map="identity",
            hover_data=tooltip_fields,
            labels={kpi: kpi_display_label},
            height=300
        )

        season_avg = chart_data[kpi].mean()
        fig.add_hline(
            y=season_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {season_avg:.2f}",
            annotation_position="top right"
        )

        if is_percentage_metric:
            fig.update_yaxes(range=[0, 100])

        fig.update_layout(xaxis_title="Player", yaxis_title=kpi_display_label, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.expander("### Players Stats KPI Comparison")
    st.dataframe(summary_metrics_df, use_container_width=True)
