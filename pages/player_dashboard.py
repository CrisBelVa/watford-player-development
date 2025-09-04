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
# Optional: helper to navigate between pages
try:
    from streamlit_extras.switch_page_button import switch_page
except ModuleNotFoundError:
    switch_page = None

# Configuración de la página - DEBE SER EL PRIMER COMANDO DE STREAMLIT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')



st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verificar autenticación
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be logged in to view this page.")
    st.stop()

# Verificar si el usuario es staff o jugador
is_staff = st.session_state.user_type == "staff"

# ---- Resolve current player (works for staff and players) ----

def load_players_list() -> Dict[str, Dict[str, Any]]:
    """
    Load players from data/watford_players_login_info.[xlsx|csv].
    Expected columns (preferred): playerId, playerName, activo
    Falls back gracefully if playerId is missing.
    Returns a dict keyed by a staff-facing label, with values containing id and name.
    """
    try:
        base_csv = os.path.join('data', 'watford_players_login_info.csv')
        base_xlsx = os.path.join('data', 'watford_players_login_info.xlsx')

        if os.path.exists(base_csv):
            players_df = pd.read_csv(base_csv)
        elif os.path.exists(base_xlsx):
            players_df = pd.read_excel(base_xlsx)
        else:
            st.warning("No players file found. Please upload a file (CSV or XLSX) with columns like playerId, playerName, activo.")
            return {}

        # Normalize column names
        players_df.columns = [c.strip() for c in players_df.columns]

        # Validate columns
        if 'playerName' not in players_df.columns:
            st.error("❌ Column 'playerName' is required in the players file.")
            return {}

        # activo handling
        if 'activo' in players_df.columns:
            players_df['activo'] = pd.to_numeric(players_df['activo'], errors='coerce').fillna(1).astype(int)
        else:
            players_df['activo'] = 1

        # Allow missing playerId, but warn
        has_id = 'playerId' in players_df.columns
        if not has_id:
            st.warning("⚠️ Column 'playerId' not found. Staff selection will not provide IDs; some features may fail.")

        players = {}
        for _, row in players_df.iterrows():
            full_name = str(row.get('playerName', '')).strip()
            if not full_name:
                continue
            pid = row.get('playerId', None) if has_id else None
            activo = int(row.get('activo', 1))

            # Staff-visible label
            status = "" if activo == 1 else " (Inactive)"
            label = f"{full_name}{status}" if has_id else f"{full_name}{status}"

            players[label] = {
                "playerId": None if pd.isna(pid) else str(pid) if pid is not None else None,
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
    Returns (player_id, player_name) for the current session.
    - Staff: from sidebar selector (requires players file; prefers playerId).
    - Player: from session_state set at login (expects player_id & player_name OR player_info dict).
    Raises Streamlit stop if missing info.
    """
    if is_staff:
        st.subheader("Player Dashboard")
        # Show staff info if available
        st.write(f"Staff: {st.session_state.staff_info['full_name']}" if 'staff_info' in st.session_state else "Staff User")

        # Manage players button (optional)
        if st.button("⚙️ Manage players list"):
            if switch_page:
                switch_page("manage_players")
            else:
                st.info("Use the page menu to open 'Manage Players'.")

        # Load players and build selector
        players = load_players_list()
        if not players:
            st.stop()

        # Filter by status
        player_status_filter = st.selectbox(
            "Filter Players by Status",
            options=["All Players", "Active Players", "Inactive Players"],
            key="player_status_filter"
        )

        filtered = {}
        for label, data in players.items():
            if player_status_filter == "Active Players" and data["activo"] != 1:
                continue
            if player_status_filter == "Inactive Players" and data["activo"] == 1:
                continue
            filtered[label] = data

        options = list(filtered.keys())
        if player_status_filter == "All Players":
            options = [""] + options

        selected = st.selectbox("Select Player", options=options, key="player_selector",
                                index=0 if player_status_filter == "All Players" else None)

        # File tools
        st.markdown("---")
        with st.expander("Player File Management", expanded=False):
            uploaded_file = st.file_uploader("Upload Players File", type=['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        up_df = pd.read_csv(uploaded_file)
                        out_path = os.path.join('data', 'watford_players_login_info.csv')
                        up_df.to_csv(out_path, index=False)
                    else:
                        up_df = pd.read_excel(uploaded_file)
                        out_path = os.path.join('data', 'watford_players_login_info.xlsx')
                        up_df.to_excel(out_path, index=False)
                    st.success(f"Players file saved to {out_path}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Error saving file: {e}")

        # Logout
        st.sidebar.markdown("---")
        st.sidebar.subheader("User Info")
        if "staff_info" in st.session_state:
            st.sidebar.write(f"**Name:** {st.session_state.staff_info.get('full_name','')}")
            st.sidebar.write(f"**Role:** {st.session_state.staff_info.get('role','')}")

        if st.sidebar.button("Logout", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if not selected:
            st.warning("Please select a player")
            st.stop()

        sel_data = filtered[selected]
        pid = sel_data.get("playerId")
        pname = sel_data.get("playerName")

        if pid is None:
            st.error("Selected player has no 'playerId'. Add a 'playerId' column to your players file.")
            st.stop()

        return str(pid), pname

    # Player (non-staff) flow: get from session_state (set during login)
    # Try common keys used in the app
    pid = (st.session_state.get("player_id") or
           st.session_state.get("playerInfo", {}).get("player_id") or
           st.session_state.get("player_info", {}).get("player_id") or
           st.session_state.get("player_info", {}).get("playerId"))
    pname = (st.session_state.get("player_name") or
             st.session_state.get("playerInfo", {}).get("player_name") or
             st.session_state.get("player_info", {}).get("player_name") or
             st.session_state.get("player_info", {}).get("full_name") or
             st.session_state.get("player_info", {}).get("playerName"))

    if not pid or not pname:
        st.error("Could not resolve player from session. Ensure login sets 'player_id' and 'player_name' (or player_info dict).")
        st.stop()

    return str(pid), str(pname)


# >>> Call the resolver BEFORE using player_name <<<
player_id, player_name = resolve_current_player(is_staff)

# Page title

# Your dashboard
try:
    logo = Image.open(LOGO_PATH)
    st.image(logo, width=100)
except FileNotFoundError:
    st.error("Logo image not found. Please check the image path.")
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


# SIDE BAR
st.sidebar.header("Time Filters")

# ---------------------------
# DATE JOIN + DELTAS + CARDS
# ---------------------------

def add_match_dates(df, match_data_df):
    """
    Adds 'matchDate' to df by joining with match_data_df on matchId.
    Robust to dtype mismatches and pre-existing matchDate columns.
    Returns (df_with_dates, min_date, max_date) with Python date objects.
    """
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

    # ❕Drop any pre-existing matchDate on the left to avoid duplicate labels later
    if "matchDate" in left.columns:
        left = left.drop(columns=["matchDate"])

    # Pick a usable date column
    date_col = None
    for c in ["startDate", "date", "kickoff", "matchDate"]:
        if c in right.columns:
            date_col = c
            break
    if date_col is None:
        out = left.copy()
        out["matchDate"] = pd.NaT
        return out, today, today

    # Parse dates
    right[date_col] = pd.to_datetime(right[date_col], errors="coerce")

    # De-dupe match_data (keep latest non-null date per match)
    if "matchId" in right.columns:
        right = (
            right.sort_values(date_col)
                 .drop_duplicates(subset=["matchId"], keep="last")
        )

    # ---- Normalize matchId on both sides ----
    def normalize_match_id(s: pd.Series) -> pd.Series:
        s = s.copy()
        num = pd.to_numeric(s, errors="coerce")
        if (num.notna().mean() >= 0.8):
            return num.astype("Int64")
        return s.astype(str).str.strip()

    if "matchId" not in left.columns or "matchId" not in right.columns:
        out = left.copy()
        out["matchDate"] = pd.NaT
        return out, today, today

    left["_matchIdKey"] = normalize_match_id(left["matchId"])
    right["_matchIdKey"] = normalize_match_id(right["matchId"])

    # Merge right date into left
    out = left.merge(
        right[["_matchIdKey", date_col]],
        on="_matchIdKey",
        how="left"
    ).rename(columns={date_col: "matchDate"})

    # If duplicates somehow appeared, collapse to a single 'matchDate'
    if (out.columns == "matchDate").sum() > 1:
        md = out.loc[:, out.columns == "matchDate"]
        # Keep the last one (usually the freshly merged right-hand date)
        out["matchDate"] = md.iloc[:, -1]
        # Drop the others
        keep_mask = np.ones(len(out.columns), dtype=bool)
        seen = 0
        for i, col in enumerate(out.columns):
            if col == "matchDate":
                seen += 1
                if seen < (md.shape[1] + 1):  # drop prior duplicates
                    keep_mask[i] = (seen == (md.shape[1] + 1))
        out = out.loc[:, keep_mask]

    # Diagnostics (safe)
    try:
        null_rate = out["matchDate"].isna().mean()
        st.caption(f"🔎 matchId join coverage: {(1 - null_rate):.0%} matched")
    except Exception:
        null_rate = 1.0

    # Compute min/max over valid dates
    valid_dates = out["matchDate"].dropna()
    if valid_dates.empty:
        return out, today, today

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

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

# 0) Normalize 'matchId' to nullable Int64 on BOTH frames
def to_int64_id(s):
    # convert strings/floats like "12345" / 12345.0 → 12345 (Int64), keep NaN as <NA>
    return pd.to_numeric(s, errors="coerce").astype("Int64")

# ensure column exists
if "matchId" not in metrics_summary.columns:
    st.error(f"'matchId' missing in metrics_summary. Columns: {list(metrics_summary.columns)}")
    st.stop()
if "matchId" not in match_data.columns:
    st.error(f"'matchId' missing in match_data. Columns: {list(match_data.columns)}")
    st.stop()

metrics_summary = metrics_summary.copy()
match_data = match_data.copy()

metrics_summary["matchId"] = to_int64_id(metrics_summary["matchId"])
match_data["matchId"]      = to_int64_id(match_data["matchId"])

# 1) Parse date column once
if "startDate" not in match_data.columns:
    st.error("match_data is missing 'startDate'.")
    st.stop()
match_data["startDate"] = pd.to_datetime(match_data["startDate"], errors="coerce")

# 2) De-dupe match_data (just in case) and attach matchDate via merge
md_dates = (
    match_data[["matchId", "startDate"]]
    .dropna(subset=["matchId"])
    .sort_values("startDate")
    .drop_duplicates(subset=["matchId"], keep="last")
    .rename(columns={"startDate": "matchDate"})
)

metrics_summary = (
    metrics_summary
    .merge(md_dates, on="matchId", how="left")
)

# 3) Validate we actually matched
non_null_rate = metrics_summary["matchDate"].notna().mean() if "matchDate" in metrics_summary else 0.0
st.caption(f"🔎 matchId→date match coverage: {non_null_rate:.0%}")

if non_null_rate == 0:
    st.error("No match dates matched. Likely mismatch in matchId values between frames.")
    st.write("metrics_summary sample:", metrics_summary[["matchId"]].head())
    st.write("match_data sample:", match_data[["matchId", "startDate"]].head())
    st.stop()

# 4) Sidebar range
valid_dates = metrics_summary["matchDate"].dropna()
min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

st.sidebar.header("Time Filters")
picked = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = picked if isinstance(picked, tuple) and len(picked) == 2 else (min_date, max_date)

# 5) Filter
mask = (metrics_summary["matchDate"].dt.date >= start_date) & (metrics_summary["matchDate"].dt.date <= end_date)
filtered_df = metrics_summary.loc[mask].copy()
if filtered_df.empty:
    st.warning("No matches in the selected date range. Showing season totals instead.")
    filtered_df = metrics_summary.copy()



# -- Dynamically assigned KPIs
metric_keys = selected_kpis

# --- Native Watford-Styled Section Selector ---
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
   
    # Merge team names into metrics_summary
    teams_info = (
        event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
        .first()
        .reset_index()
    )

    summary_df = metrics_summary.copy()
    summary_df = summary_df.merge(teams_info, on="matchId", how="left")

    # Prevent duplicate 'matchDate' before wiring dates
    if "matchDate" in summary_df.columns:
        summary_df = summary_df.drop(columns=["matchDate"])

    # Add match dates ONCE
    summary_df, _, _ = add_match_dates(summary_df, match_data)

    # Sort by date
    summary_df = summary_df.sort_values("matchDate")

    # Apply global date filter (set earlier)
    mask = (
        (summary_df["matchDate"].dt.date >= start_date) &
        (summary_df["matchDate"].dt.date <= end_date)
    )
    filtered_df = summary_df.loc[mask].copy()

    # Drop duplicated matchId and fill NaNs
    filtered_df = filtered_df.drop_duplicates(subset="matchId", keep="last").fillna(0)

    
    # Ensure team names are merged
    if "oppositionTeamName" not in filtered_df.columns:
        teams_info = (
            event_data.groupby("matchId")[["teamName", "oppositionTeamName"]]
            .first()
            .reset_index()
        )
        filtered_df = filtered_df.merge(teams_info, on="matchId", how="left")

    # Create readable match labels (after team name merge)
    filtered_df["match_label"] = filtered_df["matchDate"].dt.strftime("%Y-%m-%d") + " vs " + filtered_df["oppositionTeamName"]

    # --- Match Filter Styled Like Excel ---
    with st.expander("Filter by Match (click to hide)", expanded=True):
        match_options = filtered_df[["matchId", "match_label", "matchDate"]].drop_duplicates().sort_values("matchDate")
        match_labels_dict = dict(zip(match_options["matchId"], match_options["match_label"]))

        # Search box
        search_text = st.text_input("🔍 Search match:", "")

        filtered_options = match_options[match_options["match_label"].str.contains(search_text, case=False, na=False)]

        # Select all / clear buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Select All Matches"):
                st.session_state.selected_match_ids = list(filtered_options["matchId"])
        with col2:
            if st.button("Clear Matches"):
                st.session_state.selected_match_ids = []

        # Maintain session state
        selected_ids = st.session_state.get("selected_match_ids", list(filtered_options["matchId"]))
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

    # Apply match filter
    filtered_df = filtered_df[filtered_df["matchId"].isin(st.session_state.selected_match_ids)]

    # Set metric_keys dynamically by position
    metric_keys = selected_kpis

    # --- Aggregate Metrics ---
    
    def compute_weighted_percentage(df, numerator_col, denominator_col):
        num = df[numerator_col].sum()
        denom = df[denominator_col].sum()
        return round((num / denom) * 100, 1) if denom != 0 else 0

    # --- Aggregated Metrics for Scorecards ---

    aggregated_metrics = {}

    for key in metric_keys:
        if key not in filtered_df.columns:
            aggregated_metrics[key] = 0
            continue

        metric_type = metric_type_map.get(key, "per_match")

        # Custom logic for % metrics based on actual numerators/denominators
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
            # For all others, sum or average depending on type
            if metric_type == "percentage":
                aggregated_metrics[key] = round(filtered_df[key].mean(), 1)
            else:
                aggregated_metrics[key] = round(filtered_df[key].sum(), 2)

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

    # Format KPI columns like the scorecards
    display_df = filtered_df.copy()
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


elif section == "Player Comparison":

    st.info(f"Top Players in the Competition – **{player_position}**")

    # ---------------------------
    # Step 0: normalize team_data schema (snake_case → camelCase)
    # ---------------------------
    team_data_cmp = team_data.copy()
    team_data_cmp = team_data_cmp.rename(columns={
        "match_id": "matchId",
        "team_id": "teamId",
        "team_name": "teamName"
    })

    # helpful dtype normalizer for ids
    def to_int64_id(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")

    if "matchId" not in team_data_cmp.columns or "teamId" not in team_data_cmp.columns:
        st.error("team_data is missing matchId/teamId (after renaming). Check TEAM_DATA schema load.")
        st.stop()

    team_data_cmp["matchId"] = to_int64_id(team_data_cmp["matchId"])
    team_data_cmp["teamId"]  = to_int64_id(team_data_cmp["teamId"])

    # ---------------------------
    # Step 1: load season matches
    # ---------------------------
    selected_season = "2024-2025"
    engine = connect_to_db()
    season_matches = pd.read_sql(
        """
        SELECT * FROM match_data
        WHERE season = %s AND competition = 'eng-championship'
        """,
        con=engine,
        params=(selected_season,)
    )

    # Make sure matchId is comparable with team_data_cmp
    if "matchId" not in season_matches.columns:
        st.error("match_data returned without matchId. Check match_data schema.")
        st.stop()
    season_matches["matchId"] = to_int64_id(season_matches["matchId"])

    # ---------------------------
    # Step 2: score parsing (robust + non-blocking)
    # ---------------------------
    season_matches["score_clean"] = (
        season_matches["score"].astype(str).str.strip().str.replace(" ", "", regex=False)
    )
    valid_scores = season_matches["score_clean"].str.contains(r"^\d+:\d+$", na=False)
    clean_scores = season_matches.loc[valid_scores, "score_clean"]

    if clean_scores.empty:
        st.warning("No valid scores in season_matches — points table may be empty.")
        season_matches["home_goals"] = pd.NA
        season_matches["away_goals"] = pd.NA
    else:
        score_split = clean_scores.str.split(":", expand=True)
        if score_split.shape[1] < 2:
            st.error("❌ Score splitting failed — format not consistent with '1:0'")
            season_matches["home_goals"] = pd.NA
            season_matches["away_goals"] = pd.NA
        else:
            season_matches.loc[clean_scores.index, "home_goals"] = pd.to_numeric(score_split.iloc[:, 0], errors="coerce")
            season_matches.loc[clean_scores.index, "away_goals"] = pd.to_numeric(score_split.iloc[:, 1], errors="coerce")

    # ---------------------------
    # Step 3: filter team_data by matchIds (now using normalized columns)
    # ---------------------------
    filtered_match_ids = season_matches["matchId"].dropna().unique().tolist()
    if not filtered_match_ids:
        st.warning("No season match IDs found for the selected season.")
        st.stop()

    team_data_filtered = team_data_cmp[team_data_cmp["matchId"].isin(filtered_match_ids)].copy()
    if team_data_filtered.empty:
        st.warning("team_data has no rows for these matches.")
        st.stop()

    # ---------------------------
    # Step 4: merge scores into team_data
    # ---------------------------
    team_scores_df = pd.merge(
        team_data_filtered,
        season_matches[["matchId", "home_goals", "away_goals", "score"]],
        on="matchId",
        how="inner"
    )

    # ---------------------------
    # Step 5: infer home/away by order within matchId
    # ---------------------------
    team_scores_df["team_order"] = team_scores_df.groupby("matchId").cumcount()
    team_scores_df["home_away"] = team_scores_df["team_order"].map({0: "home", 1: "away"})

    # ---------------------------
    # Step 6: assign points
    # ---------------------------
    def assign_points(row):
        hg, ag = row["home_goals"], row["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            return 0  # or pd.NA if you prefer
        if row["home_away"] == "home":
            return 3 if hg > ag else 1 if hg == ag else 0
        else:
            return 3 if ag > hg else 1 if ag == hg else 0

    team_scores_df["points"] = team_scores_df.apply(assign_points, axis=1)

    # ---------------------------
    # Step 7: aggregate total points per team
    # ---------------------------
    team_points = (
        team_scores_df
        .groupby(["teamId", "teamName"], as_index=False)["points"]
        .sum()
        .sort_values(by="points", ascending=False)
    )

    # ---------------------------
    # Step 8: select top 5 teams
    # ---------------------------
    top_5_teams = team_points.head(5)
    top_team_ids = top_5_teams["teamId"].tolist()
    top_5_team_names = top_5_teams["teamName"].tolist()

    # Step 3: Get Real Position of Logged-in Player
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

        # Step 4: Team Filter UI (ordered by points, top 5 pre-selected)
    with st.expander("Filter by Teams", expanded=False):
        all_team_names = team_points["teamName"].tolist()  # already sorted by points descending

        # Search box
        search_team = st.text_input("🔍 Search team:", "", key="search_team")
        filtered_team_options = [t for t in all_team_names if search_team.lower() in t.lower()]

        # Set default selection only once
        if "selected_team_names" not in st.session_state:
            st.session_state.selected_team_names = top_5_team_names

        # Quick-select buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All Teams"):
                st.session_state.selected_team_names = all_team_names
        with col2:
            if st.button("Clear Teams"):
                st.session_state.selected_team_names = []

        # Show checkboxes
        selected_team_names = st.session_state.get("selected_team_names", top_5_team_names)
        new_selection = []

        st.markdown("<div style='max-height: 250px; overflow-y: auto;'>", unsafe_allow_html=True)
        for i, team in enumerate(filtered_team_options):
            checked = team in selected_team_names
            if st.checkbox(team, value=checked, key=f"team_{team}_{i}"):
                new_selection.append(team)

        st.markdown("</div>", unsafe_allow_html=True)

        # Update session state
        st.session_state.selected_team_names = new_selection

        # Clean and match team names
        team_points["teamName_clean"] = team_points["teamName"].str.strip().str.lower()
        selected_team_names_clean = [name.strip().lower() for name in new_selection]
        selected_team_ids = team_points[team_points["teamName_clean"].isin(selected_team_names_clean)]["teamId"].tolist()

    # Step 5: Player Filter UI (same position + selected teams)
    if not selected_team_ids:
        st.warning("No teams selected.")
    else:
        selected_team_ids_str = ",".join(str(tid) for tid in selected_team_ids)

        query = f"""
            SELECT 
                ps.playerId,
                ps.playerName,
                pd.age,
                pd.shirtNo,
                pd.height,
                pd.weight,
                ps.teamId,
                td.teamName,
                pd.isFirstEleven,

                ps.matchId,
                md.startDate
            FROM player_stats ps
            JOIN player_data pd 
                ON ps.playerId = pd.playerId AND ps.matchId = pd.matchId
            JOIN match_data md 
                ON ps.matchId = md.matchId
            JOIN team_data td 
                ON ps.teamId = td.teamId AND ps.matchId = td.matchId
            WHERE pd.position IN ({','.join([f"'{pos}'" for pos in position_codes])})
            AND ps.teamId IN ({selected_team_ids_str})
            AND md.startDate BETWEEN %s AND %s
        """
        players_full = pd.read_sql(query, connect_to_db(), params=(start_date, end_date))

        # 🛠️ Add minutesPlayed manually from your prepared player_data
        minutes_df = player_data[["playerId", "matchId", "minutesPlayed"]].copy()

        # --- Fix dtypes just in case ---
        players_full["playerId"] = players_full["playerId"].astype(str)
        players_full["matchId"] = players_full["matchId"].astype(str)
        minutes_df["playerId"] = minutes_df["playerId"].astype(str)
        minutes_df["matchId"] = minutes_df["matchId"].astype(str)

        # 🔁 Merge minutesPlayed into the player stats
        players_full = pd.merge(players_full, minutes_df, on=["playerId", "matchId"], how="left")

        # Build player selector BEFORE filtering
        all_player_names = players_full[["playerId", "playerName"]].drop_duplicates().sort_values("playerName")
        player_options = all_player_names["playerName"].tolist()
        top_players = (
            players_full.groupby("playerName")["minutesPlayed"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        with st.expander("Filter Players by Position", expanded=False):
            selected_players = st.multiselect("Select players to compare", player_options, default=top_players)

        # Now filter selected players
        selected_player_ids = all_player_names[all_player_names["playerName"].isin(selected_players)]["playerId"].tolist()
        filtered_players = players_full[players_full["playerId"].isin(selected_player_ids)].copy()

        # Now apply date filtering safely
        # Ensure startDate is in datetime format
        filtered_players["startDate"] = pd.to_datetime(filtered_players["startDate"], errors="coerce")

        # Now safely apply date filter
        filtered_players = filtered_players[
            (filtered_players["startDate"] >= pd.to_datetime(start_date)) &
            (filtered_players["startDate"] <= pd.to_datetime(end_date))
].copy()

        # Group and aggregate player summary
        summary_comparison_df = (
            filtered_players.groupby(["playerId", "playerName", "teamId"], as_index=False)
                .agg(
                    age=("age", "first"),
                    shirtNo=("shirtNo", "first"),
                    height=("height", "first"),
                    weight=("weight", "first"),
                    matches_played=("matchId", "nunique"),
                    games_as_starter=("isFirstEleven", "sum"),
                    total_minutes=("minutesPlayed", "sum"),
                )
            )

        # Convert teamId to str before merging
        summary_comparison_df["teamId"] = summary_comparison_df["teamId"].astype(str)
        team_data_filtered["teamId"] = team_data_filtered["teamId"].astype(str)

        # Merge to get team name
        summary_comparison_df = pd.merge(summary_comparison_df, team_data_filtered[["teamId", "teamName"]].drop_duplicates(), on="teamId", how="left")

        # Rename for display
        summary_display = summary_comparison_df.rename(columns={
            "playerName": "Player",
            "teamName": "Team",
            "age": "Age",
            "shirtNo": "Shirt No",
            "height": "Height",
                "weight": "Weight",
                "matches_played": "Games Played",
                "games_as_starter": "Games as Starter",
                "total_minutes": "Minutes Played"
            })

        # Order and show
        columns_to_display = [
            "Player", "Team", "Age", "Shirt No", "Height", "Weight",
            "Games Played", "Games as Starter", "Minutes Played"
        ]
        summary_display = summary_display.sort_values(by="Games Played", ascending=False)
        st.dataframe(summary_display[columns_to_display], use_container_width=True)

        # --- STEP 6: KPI Calculation for Selected Players ---

        # Ensure filtered_players has string playerId and matchId for safe joins
        filtered_players["playerId"] = filtered_players["playerId"].astype(str)
        filtered_players["matchId"] = filtered_players["matchId"].astype(str)

        # Ensure logged-in player is included with full metadata
        logged_player_id = str(player_id)
        logged_player_name = player_name

        # Get all matches for the logged-in player
        logged_matches_full = players_full[players_full["playerId"] == logged_player_id]
        
        # Add logged-in player's data to filtered players
        filtered_players = pd.concat([filtered_players, logged_matches_full]).drop_duplicates(
            subset=["playerId", "matchId"]
        ).reset_index(drop=True)

        # Ensure logged-in player's matches are included in match_ids
        match_ids = filtered_players["matchId"].unique().tolist()

        # Proceed to load player stats and events
        player_ids = filtered_players["playerId"].unique().tolist()
        match_ids = filtered_players["matchId"].unique().tolist()

        # --- Load player_stats and event_data for selected players ---
        player_placeholders = ','.join(['%s'] * len(player_ids))
        match_placeholders = ','.join(['%s'] * len(match_ids))

        query_stats = f"""
        SELECT * FROM player_stats
        WHERE playerId IN ({player_placeholders})
        AND matchId IN ({match_placeholders})
    """
    stats_df = pd.read_sql(query_stats, con=connect_to_db(), params=tuple(player_ids + match_ids))

    query_events = f"""
        SELECT * FROM event_data
        WHERE playerId IN ({player_placeholders})
        AND matchId IN ({match_placeholders})
    """
    events_df = pd.read_sql(query_events, con=connect_to_db(), params=tuple(player_ids + match_ids))

    # --- Convert IDs to string for safe joins ---
    stats_df["playerId"] = stats_df["playerId"].astype(str)
    stats_df["matchId"] = stats_df["matchId"].astype(str)
    events_df["playerId"] = events_df["playerId"].astype(str)
    events_df["matchId"] = events_df["matchId"].astype(str)

    # Combine logged player's stats with other players' metrics

    # --- STEP 6: KPI Calculation for Selected Players (logged-in + comparison) ---

    # Prepare logged-in player metrics from Section 1
    logged_player_metrics = metrics_summary[
    (metrics_summary["matchDate"] >= pd.to_datetime(start_date)) &
    (metrics_summary["matchDate"] <= pd.to_datetime(end_date))
    ].copy()
    logged_player_metrics["playerId"] = str(player_id)

    # Filter stats/events for comparison players
    comparison_stats = stats_df[stats_df["playerId"] != str(player_id)].copy()
    comparison_events = events_df[events_df["playerId"] != str(player_id)].copy()

    # Compute match-level metrics for comparison players
    comparison_metrics = process_player_comparison_metrics(
        comparison_stats, comparison_events, player_position
    )

    # Combine both
    all_metrics_df = pd.concat([logged_player_metrics, comparison_metrics], ignore_index=True)

    # Add playerName and teamName from summary_comparison_df
    player_info_map = summary_comparison_df.set_index("playerId")[["playerName", "teamName"]].to_dict(orient="index")
    if logged_player_id not in player_info_map:
        logged_team_name = player_data[player_data["playerId"] == logged_player_id]["teamName"].values[0]
        player_info_map[logged_player_id] = {
            "playerName": logged_player_name,
            "teamName": logged_team_name
        }
    all_metrics_df["playerName"] = all_metrics_df["playerId"].map(lambda x: player_info_map.get(x, {}).get("playerName", ""))
    all_metrics_df["teamName"] = all_metrics_df["playerId"].map(lambda x: player_info_map.get(x, {}).get("teamName", ""))

    # --- Group by player and aggregate metrics ---
    grouped = all_metrics_df.groupby("playerId")

    # Initialize summary DataFrame
    summary_metrics_df = grouped[["playerName", "teamName"]].first().reset_index()

    # Loop through position-based KPIs and aggregate
    for kpi in position_kpi_map.get(player_position, []):
        metric_type = metric_type_map.get(kpi, "aggregate")

        if metric_type == "percentage":
            numerator, denominator = percentage_formula_map.get(kpi, (None, None))
            if numerator in all_metrics_df.columns and denominator in all_metrics_df.columns:
                sum_num = grouped[numerator].sum()
                sum_den = grouped[denominator].sum()
                weighted_avg = (sum_num / sum_den.replace(0, np.nan)) * 100
                summary_metrics_df[kpi] = summary_metrics_df["playerId"].map(weighted_avg.round(1))
            else:
                summary_metrics_df[kpi] = np.nan
        else:
            if kpi in all_metrics_df.columns:
                total = grouped[kpi].sum()
                summary_metrics_df[kpi] = summary_metrics_df["playerId"].map(total.round(1))
            else:
                summary_metrics_df[kpi] = np.nan

    # Add tooltip support
    all_tooltip_fields = set()
    for fields in metric_tooltip_fields.values():
        all_tooltip_fields.update(fields)

    for tooltip_col in all_tooltip_fields:
        if tooltip_col in all_metrics_df.columns:
            summary_metrics_df[tooltip_col] = summary_metrics_df["playerId"].map(grouped[tooltip_col].sum().round(1))
        else:
            summary_metrics_df[tooltip_col] = np.nan

    # Final cleanup
    summary_metrics_df = summary_metrics_df.fillna(0)

    # --- Plot KPI Comparison Charts ---
    st.info("Comparison by KPI")

    for kpi in position_kpi_map.get(player_position, []):
        if kpi not in summary_metrics_df.columns:
            continue

        chart_data = summary_metrics_df[["playerName", kpi]].copy()
        chart_data = chart_data.sort_values(by=kpi, ascending=False)
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

        # Add season average line
        season_avg = chart_data[kpi].mean()
        fig.add_hline(
            y=season_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {season_avg:.2f}",
            annotation_position="top right"
        )

        # Clamp percentage metrics to 100%
        if metric_type_map.get(kpi) == "percentage":
            fig.update_yaxes(range=[0, 100])

        fig.update_layout(
            xaxis_title="Player",
            yaxis_title=metric_labels.get(kpi, kpi),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.expander("### Players Stats KPI Comparison")
    st.dataframe(summary_metrics_df, use_container_width=True)
