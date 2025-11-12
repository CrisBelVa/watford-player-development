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
from datetime import timedelta
from PIL import Image
from db_utils import connect_to_db, load_player_data, load_event_data_for_matches, get_player_position, process_player_metrics
from db_utils import get_all_players, process_player_comparison_metrics
from math import ceil
from typing import Tuple, Dict, Any
from sqlalchemy import create_engine
from pandas.io.formats.style import Styler
import streamlit.components.v1 as components  # ✅ needed for working HTML injection

# Optional: helper to navigate between pages
try:
    from streamlit_extras.switch_page_button import switch_page
except ModuleNotFoundError:
    switch_page = None


# === PAGE CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')

st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    Load players from data/watford_players_login_info.[xlsx|csv].
    Normalizes playerId to STRING (or None) to avoid dtype issues in the UI.
    """
    try:
        base_csv = os.path.join('data', 'watford_players_login_info.csv')
        base_xlsx = os.path.join('data', 'watford_players_login_info.xlsx')

        if os.path.exists(base_csv):
            players_df = pd.read_csv(base_csv, dtype={"playerId": "string"})
        elif os.path.exists(base_xlsx):
            players_df = pd.read_excel(base_xlsx, converters={"playerId": lambda x: str(x).strip() if pd.notna(x) else None})
        else:
            st.warning("No players file found. Upload CSV/XLSX with columns: playerId, playerName, activo.")
            return {}

        players_df.columns = [str(c).strip() for c in players_df.columns]

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
        else:
            st.warning("⚠️ Column 'playerId' not found. Some features may fail.")

        players: Dict[str, Dict[str, Any]] = {}
        for _, row in players_df.iterrows():
            full_name = str(row.get('playerName', '')).strip()
            if not full_name:
                continue
            pid = row.get('playerId', None) if has_id else None
            activo = int(row.get('activo', 1))
            label = f"{full_name}{'' if activo == 1 else ' (Inactive)'}"
            # Handle pandas NA values properly
            if pd.isna(pid) or pid in (None, "", "nan"):
                pid = None
            players[label] = {
                "playerId": pid,
                "playerName": full_name,
                "activo": activo,
            }

        return players

    except Exception as e:
        st.error(f"❌ Error loading players list: {e}")
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

        if pid is None or str(pid).strip() == "":
            st.error("Selected player has no 'playerId'. Add a 'playerId' column to your players file.")
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
    "xA": "Expected Assists (xA)",
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

selected_kpis = position_kpi_map.get(player_position, [])
if not selected_kpis:
    st.error(f"⚠️ No KPIs found for position: {player_position}")


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

def calculate_delta(filtered_df: pd.DataFrame, full_df: pd.DataFrame, column: str) -> Tuple[float, float]:
    """
    Calculates delta between filtered data (e.g., last 3 matches) and full season.

    - For percentage metrics: calculate from numerator / denominator columns.
    - For count metrics: average per match.
    """
    metric_type = metric_type_map.get(column, "per_match")
    if filtered_df.empty or full_df.empty:
        return 0.0, 0.0

    # For percentage metrics where we know the base columns
    percentage_formula_map = {
        "pass_completion_pct": ("passesAccurate", "passesTotal"),
        "aerial_duel_pct": ("aerialsWon", "aerialsTotal"),
        "take_on_success_pct": ("dribblesWon", "dribblesAttempted"),
        "shots_on_target_pct": ("shotsOnTarget", "shotsTotal"),
        "tackle_success_pct": ("tackleSuccessful", "tacklesTotal"),
        "throwin_accuracy_pct": ("throwInsAccurate", "throwInsTotal"),
        "long_pass_pct": ("long_passes_success", "long_passes_total"),
    }

    if metric_type == "percentage" and column in percentage_formula_map:
        num_col, denom_col = percentage_formula_map[column]
        try:
            filtered_num = filtered_df[num_col].sum()
            filtered_denom = filtered_df[denom_col].sum()
            season_num = full_df[num_col].sum()
            season_denom = full_df[denom_col].sum()

            filtered_value = (filtered_num / filtered_denom) * 100 if filtered_denom != 0 else 0
            season_value = (season_num / season_denom) * 100 if season_denom != 0 else 0
        except Exception:
            filtered_value = filtered_df[column].mean()
            season_value = full_df[column].mean()

    elif metric_type == "percentage":
        # Fallback: average the values if we don't know the exact formula
        filtered_value = filtered_df[column].mean()
        season_value = full_df[column].mean()

    else:
        # For count metrics: average per match
        filtered_value = filtered_df[column].sum() / max(1, len(filtered_df))
        season_value  = full_df[column].sum() / max(1, len(full_df))

    delta = filtered_value - season_value
    delta_percent = (delta / season_value * 100) if season_value != 0 else 0

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


def display_metric_card(col, title, value, filtered_df, full_df, column, color=None):
    with col:
        with st.container(border=True):
            delta, delta_percent = calculate_delta(filtered_df, full_df, column)
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""

            # Format value
            metric_type = metric_type_map.get(column, "per_match")
            formatted_value = format_metric_value(value, column)

            # Tooltip content
            tooltip_lines = [f"{title}"]

            # Related raw metric values
            for extra_field in metric_tooltip_fields.get(column, []):
                if extra_field in filtered_df.columns:
                    raw_val = filtered_df[extra_field].sum()
                    label = metric_labels.get(extra_field, extra_field.replace("_", " ").title())
                    tooltip_lines.append(f"{label}: {int(raw_val)}")

            # Avg Season and Avg Match logic
            if metric_type == "percentage" and column in percentage_formula_map:
                num_col, denom_col = percentage_formula_map[column]
                season_num = full_df[num_col].sum()
                season_denom = full_df[denom_col].sum()
                match_num = filtered_df[num_col].sum()
                match_denom = filtered_df[denom_col].sum()

                season_avg = (season_num / season_denom) * 100 if season_denom != 0 else 0
                match_avg = (match_num / match_denom) * 100 if match_denom != 0 else 0

                tooltip_lines.append(f"Avg Season: {season_avg:.1f}%")
                tooltip_lines.append(f"Avg Match Filtered: {match_avg:.1f}%")

            elif column in full_df.columns:
                season_avg = full_df[column].sum() / max(1, len(full_df))
                match_avg = filtered_df[column].sum() / max(1, len(filtered_df))
                suffix = "%" if metric_type == "percentage" else ""
                tooltip_lines.append(f"Avg Season: {season_avg:.1f}{suffix}")
                tooltip_lines.append(f"Avg Match: {match_avg:.1f}{suffix}")

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
# Sidebar: Season (on top) + Time (both roles)
# ------------------------------
st.sidebar.header("Season & Time Filters")

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
    season_choice = st.sidebar.selectbox("Season", options=season_options, index=0)
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

# 2) Time range (constrained by the season-filtered rows)
valid_dates = season_filtered["matchDate"].dropna()
if valid_dates.empty:
    st.warning("No dated matches available to build a time filter. Showing all data.")
    filtered_df = season_filtered.copy()
else:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    picked = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    start_date, end_date = (
        picked if isinstance(picked, tuple) and len(picked) == 2 else (min_date, max_date)
    )

    # 3) Apply date filter
    mask = (
        season_filtered["matchDate"].dt.date >= start_date
    ) & (
        season_filtered["matchDate"].dt.date <= end_date
    )
    filtered_df = season_filtered.loc[mask].copy()
    if filtered_df.empty:
        st.warning("No matches in the selected date range. Showing season selection only.")
        filtered_df = season_filtered.copy()

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

    # --- Apply global date filter (set earlier) ---
    # IMPORTANT: ensure start_date/end_date are date objects (not strings)
    mask = (
        (summary_df["matchDate"].dt.date >= start_date) &
        (summary_df["matchDate"].dt.date <= end_date)
    )
    filtered_df = summary_df.loc[mask].copy()
    filtered_df.to_excel("filtered_df_lucas0.xlsx")
    # Drop duplicated matchId
    filtered_df = filtered_df.drop_duplicates(subset="matchId", keep="last")
    filtered_df.to_excel("filtered_df_lucas2.xlsx")
    # ⚠️ DO NOT blanket fillna(0) on the whole DF – it corrupts strings into ints (root cause of your error).
    # Instead, only fill NaNs on numeric columns.
    num_cols = filtered_df.select_dtypes(include=["number"]).columns
    filtered_df[num_cols] = filtered_df[num_cols].fillna(0)
    filtered_df.to_excel("filtered_df_lucas2.xlsx")
    # Ensure team names are present (merge again if needed)
    if "oppositionTeamName" not in filtered_df.columns:
        teams_info = (
            event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
            .first()
            .reset_index()
        )
        filtered_df = filtered_df.merge(teams_info, on="matchId", how="left")
    else:
        filtered_df = filtered_df.copy()

    # if "oppositionTeamName" not in filtered_df.columns:
    #     teams_info = (
    #         event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
    #         .first()
    #         .reset_index()
    #     )
    #     filtered_df = filtered_df.merge(teams_info, on="matchId", how="left")

    # --- Create readable match labels (robust casting to string) ---
    date_str = filtered_df["matchDate"].dt.strftime("%Y-%m-%d").fillna("Unknown date")
      
    opp_str = filtered_df["oppositionTeamName"].astype("string").fillna("Unknown")
    #opp_str = filtered_df["oppositionTeamName"]
    filtered_df["match_label"] = date_str + " vs " + opp_str
    filtered_df.to_excel("filtered_df_lucas3.xlsx")
    # --- Match Filter Styled Like Excel ---
    with st.expander("Filter by Match (click to hide)", expanded=True):
        match_options = (
            filtered_df[["matchId", "match_label", "matchDate"]]
            .drop_duplicates()
            .sort_values("matchDate")
        )

        # Search box
        search_text = st.text_input("🔍 Search match:", "")

        filtered_options = match_options[
            match_options["match_label"].str.contains(search_text, case=False, na=False)
        ]

        # Select all / clear buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Select All Matches"):
                st.session_state.selected_match_ids = list(filtered_options["matchId"])
        with col2:
            if st.button("Clear Matches"):
                st.session_state.selected_match_ids = []

        # Maintain session state (default = all currently visible)
        default_ids = list(filtered_options["matchId"])
        selected_ids = st.session_state.get("selected_match_ids", default_ids)
        selected_match_ids = []

        # Scrollable checkbox list
        st.markdown("<div style='max-height: 250px; overflow-y: auto; padding: 0 10px;'>", unsafe_allow_html=True)
        for _, row in filtered_options.iterrows():
            checked = row["matchId"] in selected_ids
            checkbox = st.checkbox(row["match_label"], value=checked, key=f"match_{row['matchId']}")
            if checkbox:
                selected_match_ids.append(row["matchId"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.selected_match_ids = selected_match_ids

    st.caption(f"Showing Metrics for position: **{player_position}**")

    # Apply match filter (guard empty state)
    current_selected = st.session_state.get("selected_match_ids", [])
    if current_selected:
        filtered_df = filtered_df[filtered_df["matchId"].isin(current_selected)]
    else:
        # If nothing selected, show nothing but avoid crashes
        filtered_df = filtered_df.iloc[0:0]

    # Early exit if no rows after filters
    if filtered_df.empty:
        st.info("No matches in the selected filters.")
        st.stop()

    # --- Set metric_keys dynamically by position ---
    metric_keys = selected_kpis

    # --- Aggregate Metrics ---
    def compute_weighted_percentage(df, numerator_col, denominator_col):
        num = df.get(numerator_col, pd.Series(dtype=float)).sum()
        denom = df.get(denominator_col, pd.Series(dtype=float)).sum()
        return round((num / denom) * 100, 1) if denom and denom != 0 else 0

    aggregated_metrics = {}

    for key in metric_keys:
        if key not in filtered_df.columns:
            aggregated_metrics[key] = 0
            continue

        metric_type = metric_type_map.get(key, "per_match")

        # Custom logic for % metrics based on true numerators/denominators
        if key == "pass_completion_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "passesAccurate", "passesTotal")
        elif key == "aerial_duel_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "aerialsWon", "aerialsTotal")
        elif key == "take_on_success_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "dribblesWon", "dribblesAttempted")
        elif key == "shots_on_target_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "shotsOnTarget", "shotsTotal")
        elif key == "tackle_success_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "tackleSuccessful", "tacklesTotal")
        elif key == "throwin_accuracy_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "throwInsAccurate", "throwInsTotal")
        elif key == "long_pass_pct":
            aggregated_metrics[key] = compute_weighted_percentage(filtered_df, "long_passes_success", "long_passes_total")
        else:
            # Sum or mean depending on type
            if metric_type == "percentage":
                aggregated_metrics[key] = round(pd.to_numeric(filtered_df[key], errors="coerce").mean(), 1)
            else:
                aggregated_metrics[key] = round(pd.to_numeric(filtered_df[key], errors="coerce").sum(), 2)

    # --- Scorecards (4 per row) ---
    metrics_per_row = 4
    metric_chunks = [metric_keys[i:i + metrics_per_row] for i in range(0, len(metric_keys), metrics_per_row)]

    for chunk in metric_chunks:
        cols = st.columns(len(chunk))
        for i, key in enumerate(chunk):
            label = metric_labels.get(key, key.replace("_", " ").title())
            value = aggregated_metrics.get(key, "N/A")
            display_metric_card(cols[i], label, value, filtered_df, metrics_summary, key, color="#fcec03")

    # --- Full Stats Table ---
    display_df = filtered_df.copy()

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
        match_labels_dict = dict(zip(match_options["matchId"], match_options["opponent_label"]))

        search_text = st.text_input("🔍 Search match:", "")
        filtered_options = match_options[match_options["opponent_label"].str.contains(search_text, case=False, na=False)]

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Select All Matches"):
                st.session_state.selected_match_ids = list(filtered_options["matchId"])
        with col2:
            if st.button("Clear Matches"):
                st.session_state.selected_match_ids = []

        selected_ids = st.session_state.get("selected_match_ids", list(filtered_options["matchId"]))
        selected_match_ids = []

        st.markdown("<div style='max-height: 250px; overflow-y: auto; padding: 0 10px;'>", unsafe_allow_html=True)
        for _, row in filtered_options.iterrows():
            checked = row["matchId"] in selected_ids
            checkbox = st.checkbox(row["opponent_label"], value=checked, key=f"trends_match_{row['matchId']}")
            if checkbox:
                selected_match_ids.append(row["matchId"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.selected_match_ids = selected_match_ids

    # Apply match filter (but don't re-create opponent_label!)
    trends_df = trends_df[trends_df["matchId"].isin(st.session_state.selected_match_ids)]

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
        "Center Back": ["DC"],
        "Left Back": ["DL", "DML"],
        "Right Back": ["DR", "DMR"],
        "Defensive Midfielder": ["DMC"],
        "Midfielder": ["MC", "ML", "MR"],
        "Attacking Midfielder": ["AMC"],
        "Left Winger": ["AML", "FWL"],
        "Right Winger": ["AMR", "FWR"],
        "Striker": ["FW"]
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

    # ---------------------------
    # 5) DEDUPE TO ONE ROW PER PLAYER-MATCH
    # ---------------------------
    players_full.groupby(["playerId","playerName","teamId","teamName","matchId","startDate"], as_index=False).agg(
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
        players_full.groupby(["playerId","playerName","teamName"], as_index=False)
        .agg(minutes=("minutesPlayed","sum"))
        .sort_values("minutes", ascending=False)
    )

    options_ids = player_index["playerId"].astype(str).tolist()
    logged_player_id = str(player_id)
    default_ids = player_index.head(5)["playerId"].astype(str).tolist()

    if (logged_player_id in options_ids) and (logged_player_id not in default_ids):
        default_ids = ([logged_player_id] + [pid for pid in default_ids if pid != logged_player_id])[:5]

    label_map = {
        str(r.playerId): (f"{r.playerName} ({r.teamName})" if pd.notna(r.teamName) and r.teamName != "" else f"{r.playerName}")
        for _, r in player_index.iterrows()
    }

    with st.expander("Filter Players by Position", expanded=False):
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

    summary_display = summary_comparison_df.rename(columns={
        "playerName":"Player",
        "teamName":"Team",
        "age":"Age",
        "shirtNo":"Shirt No",
        "height":"Height",
        "weight":"Weight",
        "matches_played":"Games Played",
        "games_as_starter":"Games as Starter",
        "total_minutes":"Minutes Played"
    }).sort_values(by="Games Played", ascending=False)

    st.dataframe(
        summary_display[["Player","Team","Age","Shirt No","Height","Weight","Games Played","Games as Starter","Minutes Played"]],
        use_container_width=True
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

    for kpi in position_kpi_map.get(player_position, []):
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
                total = grouped[kpi].sum()
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

    # --- Charts ---
    st.info("Comparison by KPI")

    for kpi in position_kpi_map.get(player_position, []):
        if kpi not in summary_metrics_df.columns:
            continue

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
            title=metric_labels.get(kpi, kpi),
            color="color",
            color_discrete_map="identity",
            hover_data=tooltip_fields,
            labels={kpi: metric_labels.get(kpi, kpi)},
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

        if metric_type_map.get(kpi) == "percentage":
            fig.update_yaxes(range=[0, 100])

        fig.update_layout(xaxis_title="Player", yaxis_title=metric_labels.get(kpi, kpi), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.expander("### Players Stats KPI Comparison")
    st.dataframe(summary_metrics_df, use_container_width=True)


