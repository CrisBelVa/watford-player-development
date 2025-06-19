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
from datetime import timedelta
from PIL import Image
from db_utils import connect_to_db
from db_utils import load_player_data
from db_utils import load_event_data_for_matches
from db_utils import get_player_position
from db_utils import process_player_metrics
from db_utils import get_minutes, get_active_players
from math import ceil
from typing import Tuple, Dict, Any
from sqlalchemy import create_engine
from pandas.io.formats.style import Styler

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

# Función para cargar la lista de jugadores activos
def load_players_list() -> Dict[int, Dict[str, Any]]:
    """Carga la lista de jugadores activos para el menú desplegable."""
    try:
        players_df = get_active_players()
        
        # Verificar si el DataFrame está vacío
        if players_df is None or players_df.empty:
            st.error("No se encontraron jugadores activos.")
            return {}
            
        # Convertir el DataFrame a un diccionario con el formato esperado
        players_dict = {}
        for _, row in players_df.iterrows():
            player_id = row.get('player_id') or row.get('id')  # Ajusta según el nombre de la columna
            if player_id is not None:
                players_dict[player_id] = {
                    'full_name': f"{row.get('nombre', '')} {row.get('apellido', '')}".strip(),
                    # Agrega aquí otros campos que necesites
                }
        
        if not players_dict:
            st.error("No se pudieron cargar los datos de los jugadores.")
            return {}
        return players_dict
        
    except Exception as e:
        st.error(f"Error al cargar la lista de jugadores: {str(e)}")
        import traceback
        st.error(f"Detalles del error: {traceback.format_exc()}")
        return {}

# Configuración de la barra lateral
with st.sidebar:
    st.subheader("User Info")
    
    if is_staff:
        # Mostrar información del staff
        st.write(f"Staff: {st.session_state.staff_info['full_name']}" if 'staff_info' in st.session_state else "Staff User")
        
        # Cargar lista de jugadores para el menú desplegable
        players = load_players_list()
        player_list = {f"{v['full_name']} (ID: {k})": k for k, v in players.items() if k is not None}
        
        # Seleccionar jugador
        selected_player = st.selectbox(
            "Select Player",
            options=[""] + list(player_list.keys()),
            index=0
        )
        
        if selected_player:
            player_id = player_list[selected_player]
            player_name = selected_player.split(" (ID:")[0]
        else:
            st.warning("Please select a player")
            st.stop()
    else:
        # Mostrar información del jugador
        st.write(f"Player: {st.session_state.player_name}")
        st.write(f"ID: {st.session_state.player_id}")
        player_id = st.session_state.player_id
        player_name = st.session_state.player_name
    
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Si es staff y no se ha seleccionado un jugador, mostrar mensaje
if is_staff and 'player_id' not in locals():
    st.warning("Please select a player from the sidebar to view their dashboard.")
    st.stop()

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

# After loading player_data as before
player_id = str(player_id)
match_ids = [str(m) for m in player_data["matchId"].unique()]
team_id = str(player_data["teamId"].iloc[0])  # assuming single team

event_data = load_event_data_for_matches(player_id, match_ids, team_id=team_id)

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

metrics_summary = process_player_metrics(
    player_stats=player_stats,
    event_data=event_data,
    player_id=player_id,
    player_name=player_name
)


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



def add_match_dates(df, match_data_df):
    """
    Adds a 'matchDate' column to df based on matchId using match_data_df,
    handles NaT and fallback for Streamlit compatibility.
    Returns updated df and valid min/max dates as Python date objects.
    """
    # Parse match start dates safely
    match_data_df["startDate"] = pd.to_datetime(match_data_df["startDate"], errors="coerce")

    # Map matchId to parsed startDate
    match_dates_dict = dict(zip(match_data_df["matchId"], match_data_df["startDate"]))
    
    df = df.copy()
    df["matchDate"] = df["matchId"].map(match_dates_dict)

    # Drop rows with missing matchDate
    df = df.dropna(subset=["matchDate"])

    if df.empty:
        today = pd.Timestamp.today().date()
        return df, today, today

    min_date = df["matchDate"].min()
    max_date = df["matchDate"].max()

    # Ensure Python date objects (not pandas Timestamp)
    min_date = min_date.date() if pd.notnull(min_date) else pd.Timestamp.today().date()
    max_date = max_date.date() if pd.notnull(max_date) else pd.Timestamp.today().date()

    return df, min_date, max_date


# DELTAS

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
        except:
            filtered_value = filtered_df[column].mean()
            season_value = full_df[column].mean()

    elif metric_type == "percentage":
        # Fallback: average the values
        filtered_value = filtered_df[column].mean()
        season_value = full_df[column].mean()

    else:
        # For count metrics: average per match
        filtered_value = filtered_df[column].sum() / len(filtered_df)
        season_value = full_df[column].sum() / len(full_df)

    delta = filtered_value - season_value
    delta_percent = (delta / season_value * 100) if season_value != 0 else 0

    return round(delta, 1), round(delta_percent, 1)


#Metric Card    


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
                season_avg = full_df[column].sum() / len(full_df)
                match_avg = filtered_df[column].sum() / len(filtered_df)
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

# -- Apply match dates to player metrics
metrics_summary, min_date, max_date = add_match_dates(metrics_summary, match_data)

# -- Get default range from last 5 games
if not metrics_summary.empty:
    last_5_games = metrics_summary.sort_values("matchDate").drop_duplicates("matchId").tail(5)
    default_start_date = last_5_games["matchDate"].min().date() if pd.notnull(last_5_games["matchDate"].min()) else min_date
    default_end_date = last_5_games["matchDate"].max().date() if pd.notnull(last_5_games["matchDate"].max()) else max_date
else:
    default_start_date = min_date
    default_end_date = max_date

# -- Date selectors with safe defaults
start_date = st.sidebar.date_input("Start date", default_start_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", default_end_date, min_value=min_date, max_value=max_date)

# -- Filter data based on selected date range
mask = (
    (metrics_summary["matchDate"].dt.date >= start_date) &
    (metrics_summary["matchDate"].dt.date <= end_date)
)
filtered_df = metrics_summary.loc[mask].sort_values("matchDate")

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

# Extract player static details (only once)
filtered_player_data = player_data[player_data["playerId"] == player_id]

if filtered_player_data.empty:
    st.error(f"🚨 No player data found for playerId: {player_id}")
    st.stop()
else:
    player_info = filtered_player_data.iloc[0]

age = player_info["age"]
shirt_number = player_info["shirtNo"]
height = player_info["height"]
weight = player_info["weight"]
team_name = player_info["teamName"]

# --- Now dynamic stats based on time filter (start_date, end_date)

# 1. Merge player_data with match_data
filtered_logged_player_info = pd.merge(
    player_data,
    match_data[["matchId", "startDate"]],
    on="matchId",
    how="left"
)

filtered_logged_player_info["startDate"] = pd.to_datetime(filtered_logged_player_info["startDate"], errors="coerce")

# 2. Apply player ID and date filter
filtered_logged_player_info = filtered_logged_player_info[
    (filtered_logged_player_info["playerId"] == player_id) &
    (filtered_logged_player_info["startDate"].dt.date >= start_date) &
    (filtered_logged_player_info["startDate"].dt.date <= end_date)
]

# 3. Now calculate correctly
games_played = filtered_logged_player_info.shape[0]
games_as_starter = filtered_logged_player_info["isFirstEleven"].sum()
total_minutes = filtered_logged_player_info["minutesPlayed"].sum()

# --- Labels and Values for display
labels = [
    "Age", "Shirt Number", "Height", "Weight", "Games Played",
    "Games as Starter", "Minutes Played"
]
values = [
    age, shirt_number, height, weight,
    games_played, int(games_as_starter), int(total_minutes)
]

# --- Styled Player Info Cards (your custom HTML)
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

    # Add match dates
    summary_df, _, _ = add_match_dates(summary_df, match_data)
    summary_df = summary_df.sort_values("matchDate")

    # Apply global date filter (set earlier)
    mask = (summary_df["matchDate"].dt.date >= start_date) & (summary_df["matchDate"].dt.date <= end_date)
    filtered_df = summary_df[mask].copy()

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

    # --- Now Plot with Plotly ---
    for key in metric_keys:
        chart_data = trends_df[["opponent_label", "matchDate", key] + metric_tooltip_fields.get(key, [])].dropna().copy()

        if chart_data.empty:
            continue

        season_avg = metrics_summary[key].mean()
        
        with st.container(border=True):
            st.markdown(f"**{metric_labels[key]}**")
            
            # Ordenar por fecha
            chart_data = chart_data.sort_values("matchDate")
            
            # Crear gráfico de barras con Plotly
            fig = px.bar(
                chart_data,
                x='opponent_label',
                y=key,
                title=f"{metric_labels[key]} por Partido",
                labels={"opponent_label": "Partido", key: metric_labels[key]},
                color_discrete_sequence=['#fcec03']
            )
            
            # Agregar etiquetas de texto redondeadas
            fig.update_traces(
                text=chart_data[key].round(2),
                textposition='outside'
            )
            
            # Añadir línea de promedio
            fig.add_hline(
                y=season_avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f'Promedio: {season_avg:.2f}',
                annotation_position="top right"
            )
            
            # Mejorar el diseño
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False,
                margin=dict(t=50, b=100, l=50, r=50),
                xaxis=dict(tickmode='array', tickvals=chart_data['opponent_label']),
                hovermode='x unified'
            )
            
            # Mejorar tooltips
            hover_text = [f"<b>Partido:</b> {row['opponent_label']}<br>"
                        f"<b>{metric_labels[key]}:</b> {row[key]:.2f}<br>"
                        for _, row in chart_data.iterrows()]
            
            fig.update_traces(
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                textposition='outside'
            )
            
            # Mostrar el gráfico
            st.plotly_chart(fig, use_container_width=True)



elif section == "Player Comparison":

    st.info(f"Top 5 Players in the Competition – **{player_position}**")

    match_data["startDate"] = pd.to_datetime(match_data["startDate"], errors="coerce")
    filtered_matches = match_data[
        (match_data["startDate"].dt.date >= start_date) &
        (match_data["startDate"].dt.date <= end_date)
    ]

    # Extract goals
    score_split = filtered_matches["score"].str.replace(" ", "", regex=False).str.split(":", expand=True)
    filtered_matches["home_goals"] = pd.to_numeric(score_split[0], errors="coerce")
    filtered_matches["away_goals"] = pd.to_numeric(score_split[1], errors="coerce")

    # Merge with team_data
    team_scores_df = pd.merge(
        team_data,
        filtered_matches[["matchId", "home_goals", "away_goals"]],
        on="matchId", how="inner"
    )

    # Infer home/away
    team_scores_df["home_away"] = team_scores_df.apply(
        lambda row: "home" if row["scores.fulltime"] == row["home_goals"] else "away", axis=1
    )

    # Assign points
    def assign_points(row):
        if row["home_away"] == "home":
            return 3 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 0
        else:
            return 3 if row["away_goals"] > row["home_goals"] else 1 if row["away_goals"] == row["home_goals"] else 0

    team_scores_df["points"] = team_scores_df.apply(assign_points, axis=1)

    # Group to get top 5 teams
    team_points = (
        team_scores_df
        .groupby(["teamId", "teamName"], as_index=False)["points"]
        .sum()
        .sort_values(by="points", ascending=False)
        .head(5)
    )
    top_team_ids = team_points["teamId"].tolist()


    # --- Step 3: Filter Top 5 Players Matching Position ---

    reverse_position_map = {
        "Goalkeeper": ["GK"],
        "Center Back": ["DC"],
        "Left Back": ["DL", "DML"],
        "Right Back": ["DR", "DMR"],
        "Defensive Midfielder": ["DMC"],
        "Midfielder": ["MC", "ML", "MR"],
        "Attacking Midfielder": ["AMC", "AML", "AMR"],
        "Striker": ["FW", "FWL", "FWR"]
    }

    position_codes = reverse_position_map.get(player_position, [])

    top5_team_ids_str = ','.join(map(str, top_team_ids))

    # --- Query Top 5 Players and Player Info ---
    with st.spinner('🔎 Loading Top 5 players and stats...'):

        query_top5_players = f"""
            SELECT ps.* FROM player_stats ps
            JOIN player_data pd ON ps.playerId = pd.playerId AND ps.matchId = pd.matchId
            JOIN match_data md ON ps.matchId = md.matchId
            WHERE pd.position IN ({','.join([f"'{pos}'" for pos in position_codes])})
            AND ps.teamId IN ({top5_team_ids_str})
            AND md.startDate BETWEEN '{start_date}' AND '{end_date}'
        """
        all_player_stats = pd.read_sql(query_top5_players, connect_to_db())

        query_player_info = f"""
            SELECT * FROM player_data
            WHERE playerId IN (
                SELECT DISTINCT playerId FROM player_stats
                WHERE teamId IN ({top5_team_ids_str})
            )
        """
        player_info_df = pd.read_sql(query_player_info, connect_to_db())


    # Merge everything
    players_full = pd.merge(
        all_player_stats,
        player_info_df[['playerId', 'matchId', 'position', 'age', 'shirtNo', 'height', 'weight', 'isFirstEleven', 'subbedInExpandedMinute', 'subbedOutExpandedMinute']],
        on=['playerId', 'matchId'],
        how='left'
    )
    players_full = pd.merge(players_full, match_data[['matchId', 'startDate']], on='matchId', how='left')

    players_full['startDate'] = pd.to_datetime(players_full['startDate'])
    players_full['ratings_clean'] = pd.to_numeric(players_full['ratings'].astype(str).str.replace(",", "."), errors='coerce')
    players_full = players_full.drop_duplicates(subset=["playerId", "matchId"])

    # --- Step 4: Aggregate and Rank Top 5 Players ---

    players_full['subbedInExpandedMinute'] = players_full['subbedInExpandedMinute'].astype(str).str.replace(",", ".").replace("None", pd.NA)
    players_full['subbedOutExpandedMinute'] = players_full['subbedOutExpandedMinute'].astype(str).str.replace(",", ".").replace("None", pd.NA)
    players_full['subbedInExpandedMinute'] = pd.to_numeric(players_full['subbedInExpandedMinute'], errors='coerce')
    players_full['subbedOutExpandedMinute'] = pd.to_numeric(players_full['subbedOutExpandedMinute'], errors='coerce')

    def get_minutes(row):
        if row['isFirstEleven'] == 1:
            return row['subbedOutExpandedMinute'] if pd.notna(row['subbedOutExpandedMinute']) else 90.0
        elif pd.notna(row['subbedInExpandedMinute']):
            return row['subbedOutExpandedMinute'] - row['subbedInExpandedMinute'] if pd.notna(row['subbedOutExpandedMinute']) else 90.0 - row['subbedInExpandedMinute']
        else:
            return 0.0

    players_full['minutesPlayed'] = players_full.apply(get_minutes, axis=1)

    # --- Step 4: Aggregate and Rank Top 5 Players ---

    # --- Step 4: Aggregate and Rank Top 5 Players ---

    summary_df_top5 = (
        players_full.groupby(["playerId", "playerName"], as_index=False)
        .agg(
            teamId=("teamId", "first"),
            age=("age", "first"),
            shirtNo=("shirtNo", "first"),
            height=("height", "first"),
            weight=("weight", "first"),
            matches_played=("matchId", "nunique"),
            games_as_starter=("isFirstEleven", "sum"),
            total_minutes=("minutesPlayed", "sum"),
            total_rating=("ratings_clean", "sum")  # Needed for sorting but not to display
        )
    )

    # Merge teamName
    summary_df_top5 = pd.merge(
        summary_df_top5,
        team_data[["teamId", "teamName"]],
        on="teamId",
        how="left"
    )

    # Sort and select Top 5 by total_rating
    summary_df_top5 = (
        summary_df_top5
        .sort_values(by="total_rating", ascending=False)
        .drop_duplicates(subset="playerId", keep="first")
        .head(5)
    )

    # --- Step 4.5: Prepare Display Table ---

    # Rename columns for nice display
    summary_display = summary_df_top5.rename(columns={
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

    # Select only the columns we want to display
    columns_to_display = [
        "Player", "Team", "Age", "Shirt No", "Height", "Weight",
        "Games Played", "Games as Starter", "Minutes Played"
    ]

    # Optional: Sort the table by Matches Played or any field you prefer
    summary_display = summary_display.sort_values(by="Games Played", ascending=False)

    # --- Step 4.6: Show Table ---

    st.dataframe(summary_display[columns_to_display], use_container_width=True)



    # --- Step 4.5: Prepare Top 5 Stats and Events before KPI calculation ---
    top5_player_ids = summary_df_top5["playerId"].tolist()

    # ✅ Query filtered player_stats (already done above: all_player_stats)
    player_stats_top5 = all_player_stats[all_player_stats["playerId"].isin(top5_player_ids)].copy()

    # ✅ Now query filtered event_data
    query_top5_events = f"""
        SELECT e.* FROM event_data e
        JOIN match_data m ON e.matchId = m.matchId
        WHERE e.playerId IN ({','.join(map(str, top5_player_ids))})
        AND m.startDate BETWEEN '{start_date}' AND '{end_date}'
    """
    all_event_data_top5 = pd.read_sql(query_top5_events, connect_to_db())
    event_data_top5 = all_event_data_top5.copy()


    # --- Step 5: Calculate KPIs for Top 5 Players ---

    # 1. Get top 5 player IDs from previous summary
    top5_player_ids = summary_df_top5["playerId"].tolist()

    # 2. Filter player_stats by new SQL query
    query_top5_players = f"""
        SELECT ps.* FROM player_stats ps
        JOIN player_data pd ON ps.playerId = pd.playerId AND ps.matchId = pd.matchId
        JOIN match_data md ON ps.matchId = md.matchId
        WHERE pd.position IN ({','.join([f"'{pos}'" for pos in position_codes])})
        AND ps.teamId IN ({top5_team_ids_str})
        AND md.startDate BETWEEN '{start_date}' AND '{end_date}'
    """
    all_player_stats = pd.read_sql(query_top5_players, connect_to_db())

    player_stats_top5 = all_player_stats[all_player_stats["playerId"].isin(top5_player_ids)].copy()

    # 3. Clean player_stats_top5
    metric_cols = player_stats_top5.columns.drop(['playerId', 'playerName', 'matchId', 'field', 'teamId', 'teamName'], errors='ignore')
    for col in metric_cols:
        player_stats_top5[col] = player_stats_top5[col].astype(str).str.replace(',', '.', regex=False)
        player_stats_top5[col] = pd.to_numeric(player_stats_top5[col], errors='coerce')

    # 4. Filter event_data by new SQL query
    query_top5_events = f"""
        SELECT e.* FROM event_data e
        JOIN match_data m ON e.matchId = m.matchId
        WHERE e.playerId IN ({','.join(map(str, top5_player_ids))})
        AND m.startDate BETWEEN '{start_date}' AND '{end_date}'
    """
    all_event_data_top5 = pd.read_sql(query_top5_events, connect_to_db())
    event_data_top5 = all_event_data_top5.copy()

    # 5. Clean event_data_top5
    for col in ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY', 'value_Length', 'xG', 'xA', 'ps_xG']:
        if col in event_data_top5.columns:
            event_data_top5[col] = event_data_top5[col].astype(str).str.replace(',', '.', regex=False)
            event_data_top5[col] = pd.to_numeric(event_data_top5[col], errors='coerce')

    # 6. Compute inline event-based metrics
    passes_into_penalty_area = event_data_top5[
        (event_data_top5['value_PassEndX'] >= 94) &
        (event_data_top5['value_PassEndY'].between(21, 79))
    ].groupby(['playerId', 'matchId']).size().rename('passes_into_penalty_area')

    carries_into_final_third = event_data_top5[
        (event_data_top5['x'] < 66.7) & (event_data_top5['endX'] >= 66.7)
    ].groupby(['playerId', 'matchId']).size().rename('carries_into_final_third')

    carries_into_penalty_area = event_data_top5[
        (event_data_top5['endX'] >= 94) & (event_data_top5['endY'].between(21, 79))
    ].groupby(['playerId', 'matchId']).size().rename('carries_into_penalty_area')
    

    goals = event_data_top5[event_data_top5['type_displayName'] == 'Goal'].groupby(['playerId', 'matchId']).size().rename('goals')
    assists = event_data_top5[event_data_top5['value_IntentionalGoalAssist'] == 1].groupby(['playerId', 'matchId']).size().rename('assists')
    crosses = event_data_top5[event_data_top5['value_Cross'] == 1].groupby(['playerId', 'matchId']).size().rename('crosses')

    long_passes = event_data_top5[event_data_top5['value_Length'] >= 30]
    long_passes_total = long_passes.groupby(['playerId', 'matchId']).size().rename('long_passes_total')
    long_passes_success = long_passes[long_passes['outcomeType_displayName'] == 'Successful'].groupby(['playerId', 'matchId']).size().rename('long_passes_success')
    long_pass_pct = (long_passes_success / long_passes_total.replace(0, np.nan) * 100).rename('long_pass_pct')

    progressive_pass_distance = event_data_top5[
        (event_data_top5['type_displayName'] == 'Pass') & (event_data_top5['value_Length'] > 10)
    ].groupby(['playerId', 'matchId'])['value_Length'].sum().rename('progressive_passes')

    progressive_carry_distance = event_data_top5[
        (event_data_top5['type_displayName'] == 'Carry') & ((event_data_top5['endX'] - event_data_top5['x']) > 10)
    ].assign(distance=lambda d: d['endX'] - d['x']) \
    .groupby(['playerId', 'matchId'])['distance'].sum().rename('progressive_carry_distance')

    recoveries = event_data_top5[event_data_top5['type_displayName'] == 'Recovery'].groupby(['playerId', 'matchId']).size().rename('recoveries')
    interceptions = event_data_top5[event_data_top5['type_displayName'] == 'Interception'].groupby(['playerId', 'matchId']).size().rename('interceptions')
    clearances = event_data_top5[event_data_top5['type_displayName'] == 'Clearance'].groupby(['playerId', 'matchId']).size().rename('clearances')

    def_actions_outside_box = event_data_top5[
        (event_data_top5['x'] > 25) &
        (event_data_top5['type_displayName'].isin(['Tackle', 'Interception', 'Clearance']))
    ].groupby(['playerId', 'matchId']).size().rename('def_actions_outside_box')

    shot_creation_actions = event_data_top5[event_data_top5['value_ShotAssist'] > 0].groupby(['playerId', 'matchId']).size().rename('shot_creation_actions')

    xG = event_data_top5.groupby(['playerId', 'matchId'])['xG'].sum().rename('xG')
    xA = event_data_top5.groupby(['playerId', 'matchId'])['xA'].sum().rename('xA')
    ps_xG = event_data_top5.groupby(['playerId', 'matchId'])['ps_xG'].sum().rename('ps_xG')

    # 7. Combine all event metrics
    event_metrics_top5 = pd.concat([
        passes_into_penalty_area,
        carries_into_final_third,
        carries_into_penalty_area,
        goals, assists, crosses,
        long_pass_pct,
        progressive_pass_distance,
        progressive_carry_distance,
        recoveries, interceptions, clearances,
        def_actions_outside_box,
        shot_creation_actions,
        xG, xA, ps_xG
    ], axis=1).reset_index().fillna(0)

    # 8. Merge stats + events
    merged_top5 = pd.merge(
        player_stats_top5,
        event_metrics_top5,
        on=["playerId", "matchId"],
        how="left"
    ).fillna(0)

    # 9. Derived KPIs
    merged_top5["pass_completion_pct"] = (merged_top5["passesAccurate"] / merged_top5["passesTotal"].replace(0, np.nan)) * 100
    merged_top5["aerial_duel_pct"] = (merged_top5["aerialsWon"] / merged_top5["aerialsTotal"].replace(0, np.nan)) * 100
    merged_top5["take_on_success_pct"] = (merged_top5["dribblesWon"] / merged_top5["dribblesAttempted"].replace(0, np.nan)) * 100
    merged_top5["shots_on_target_pct"] = (merged_top5["shotsOnTarget"] / merged_top5["shotsTotal"].replace(0, np.nan)) * 100
    merged_top5["tackle_success_pct"] = (merged_top5["tackleSuccessful"] / merged_top5["tacklesTotal"].replace(0, np.nan)) * 100
    merged_top5["throwin_accuracy_pct"] = (merged_top5["throwInsAccurate"] / merged_top5["throwInsTotal"].replace(0, np.nan)) * 100
    merged_top5["key_passes"] = merged_top5["passesKey"]
    merged_top5["goal_creating_actions"] = merged_top5["passesKey"] + merged_top5["dribblesWon"]
    merged_top5["shot_creating_actions"] = merged_top5["shotsTotal"] + merged_top5["passesKey"]

    # 10. Round percentage metrics
    percent_cols = [
        'pass_completion_pct', 'aerial_duel_pct', 'take_on_success_pct',
        'shots_on_target_pct', 'tackle_success_pct', 'throwin_accuracy_pct', 'long_pass_pct'
    ]
    merged_top5[percent_cols] = merged_top5[percent_cols].round(1)

    # 11. Aggregate metrics per playerId
    all_metric_keys = list(set().union(*position_kpi_map.values()))
    available_keys = [k for k in all_metric_keys if k in merged_top5.columns]
    summary_df_top5_metrics = (
        merged_top5
        .groupby("playerId", as_index=False)
        .agg({k: "mean" for k in available_keys})
    )

    # 12. Merge playerName and teamName
    summary_df_top5_metrics = pd.merge(
        summary_df_top5[["playerId", "playerName", "teamName"]],
        summary_df_top5_metrics,
        on="playerId",
        how="left"
    )


    # Metrics that must be summed (count metrics like goals, assists, carries, etc.)
    sum_metrics = [
        "passes_into_penalty_area", "carries_into_final_third", "carries_into_penalty_area",
        "goals", "assists", "crosses", "recoveries", "interceptions", "clearances",
        "def_actions_outside_box", "shot_creation_actions", "xG", "xA", "ps_xG",
        "key_passes", "goal_creating_actions", "shot_creating_actions"
    ]

    # Metrics that must be averaged (percentage and efficiency metrics)
    mean_metrics = [
        "pass_completion_pct", "aerial_duel_pct", "take_on_success_pct",
        "shots_on_target_pct", "tackle_success_pct", "throwin_accuracy_pct", "long_pass_pct",
        "progressive_pass_distance", "progressive_carry_distance"
    ]

    # --- Step 5B: Calculate metrics for Top5 Players ---

    # First, generate derived KPIs
    merged_top5["key_passes"] = merged_top5["passesKey"]
    merged_top5["goal_creating_actions"] = merged_top5["passesKey"] + merged_top5["dribblesWon"]
    merged_top5["shot_creating_actions"] = merged_top5["shotsTotal"] + merged_top5["passesKey"]

    # Now aggregate properly
    aggregation_dict = {}
    for metric in sum_metrics:
        if metric in merged_top5.columns:
            aggregation_dict[metric] = "sum"
    for metric in mean_metrics:
        if metric in merged_top5.columns:
            aggregation_dict[metric] = "mean"

    summary_df_top5_metrics = (
        merged_top5
        .groupby("playerId", as_index=False)
        .agg(aggregation_dict)
    )

    # Merge player names and teams
    summary_df_top5_metrics = pd.merge(
        summary_df_top5[["playerId", "playerName", "teamName"]],
        summary_df_top5_metrics,
        on="playerId",
        how="left"
    )

    # --- Step 5C: Add Logged-in Player Metrics (metrics_summary) ---

   # --- Filter logged player metrics by date ---
    metrics_summary["matchDate"] = pd.to_datetime(metrics_summary["matchDate"], errors="coerce")
    metrics_summary_filtered = metrics_summary[
        (metrics_summary["matchDate"].dt.date >= start_date) &
        (metrics_summary["matchDate"].dt.date <= end_date)
    ].copy()

    # Fallback if no matches in range
    if metrics_summary_filtered.empty:
        metrics_summary_filtered = metrics_summary.copy()

    # --- Calculate KPIs ---
    metrics_summary_filtered["key_passes"] = metrics_summary_filtered["passesKey"]
    metrics_summary_filtered["goal_creating_actions"] = metrics_summary_filtered["passesKey"] + metrics_summary_filtered["dribblesWon"]
    metrics_summary_filtered["shot_creating_actions"] = metrics_summary_filtered["shotsTotal"] + metrics_summary_filtered["passesKey"]

    # --- Aggregate ---
    logged_player_agg = {}

    for metric in sum_metrics:
        if metric in metrics_summary_filtered.columns:
            logged_player_agg[metric] = metrics_summary_filtered[metric].sum()

    for metric in mean_metrics:
        if metric in metrics_summary_filtered.columns:
            logged_player_agg[metric] = metrics_summary_filtered[metric].mean()

    logged_player_df = pd.DataFrame({
        "playerId": [player_id],
        "playerName": [player_name],
        "teamName": [team_name],
        **logged_player_agg
    })

    # --- Step 5D: Combine Top5 + Logged-in Player ---
    combined_metrics_df = pd.concat([summary_df_top5_metrics, logged_player_df], ignore_index=True)

    # Remove duplicates (keep first instance of playerId)
    combined_metrics_df = combined_metrics_df.drop_duplicates(subset="playerId", keep="first").reset_index(drop=True)

    # --- Step 5E: Plot the bar plots for each metric ---

    metric_keys = position_kpi_map.get(player_position, [])

    if not metric_keys:
        st.warning(f"No KPIs defined for position: {player_position}")
    else:
        st.info("Player Comparison")

        for key in metric_keys:
            chart_data = combined_metrics_df[["playerName", "teamName", key]].dropna().copy()

            if chart_data.empty:
                continue

            season_avg = chart_data[key].mean()

            with st.container(border=True):
                st.markdown(f"**{metric_labels.get(key, key)}**")

                chart_data["color"] = chart_data["playerName"].apply(
                    lambda name: "#FFD700" if name == player_name else "#d3d3d3"
                )

                # Tooltip
                tooltip_fields = [
                    alt.Tooltip("playerName:N", title="Player"),
                    alt.Tooltip(f"{key}:Q", title=metric_labels.get(key, key), format=".2f")
                ]

                bar_chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("playerName:N", title="Player", sort="-y"),
                    y=alt.Y(f"{key}:Q", title=metric_labels.get(key, key)),
                    color=alt.Color("color:N", scale=None),
                    tooltip=tooltip_fields
                )

                avg_line = alt.Chart(pd.DataFrame({"y": [season_avg]})).mark_rule(
                    color="red", strokeDash=[4, 4]
                ).encode(y="y:Q")

                avg_text = alt.Chart(pd.DataFrame({
                    "x": [chart_data["playerName"].iloc[-1]],
                    "y": [season_avg],
                    "label": [f"Avg: {season_avg:.1f}"]
                })).mark_text(
                    align="left",
                    baseline="bottom",
                    dy=-5,
                    color="red"
                ).encode(
                    x=alt.X("x:N"),
                    y=alt.Y("y:Q"),
                    text="label:N"
                )

                chart = (bar_chart + avg_line + avg_text).properties(height=300)

                st.altair_chart(chart, use_container_width=True)
