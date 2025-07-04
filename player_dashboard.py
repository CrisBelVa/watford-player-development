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
from db_utils import connect_to_db, load_player_data, load_event_data_for_matches, get_player_position, process_player_metrics
from db_utils import process_player_comparison_metrics, get_active_players
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

    # --- Now Plot ---
    for key in metric_keys:
        chart_data = trends_df[["opponent_label", "matchDate", key] + metric_tooltip_fields.get(key, [])].dropna().copy()

        if chart_data.empty:
            continue

        season_avg = metrics_summary[key].mean()

        with st.container(border=True):
            st.markdown(f"**{metric_labels[key]}**")

            # Tooltip list
            tooltip_fields = [
                alt.Tooltip("opponent_label:N", title="Match"),
                alt.Tooltip(f"{key}:Q", title=metric_labels[key], format=".2f")
            ]
            for extra_field in metric_tooltip_fields.get(key, []):
                if extra_field in chart_data.columns:
                    tooltip_fields.append(
                        alt.Tooltip(f"{extra_field}:Q", title=extra_field.replace("_", " ").title())
                    )

            # NEW: Sort opponent_label based on matchDate!
            sort_order = list(chart_data.sort_values("matchDate")["opponent_label"])

            bar_chart = alt.Chart(chart_data).mark_bar(color="#fcec03").encode(
                x=alt.X("opponent_label:N", title="Match", sort=sort_order, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(f"{key}:Q", title=metric_labels[key]),
                tooltip=tooltip_fields
            )

            avg_line = alt.Chart(pd.DataFrame({"y": [season_avg]})).mark_rule(
                color="red", strokeDash=[4, 4]
            ).encode(y="y:Q")

            avg_text = alt.Chart(pd.DataFrame({
                "x": [sort_order[-1]],
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


elif section == "Player Comparison":

    st.info(f"Top Players in the Competition – **{player_position}**")

    # Step 1: Load All Match Data for Selected Season
    selected_season = "2024-2025"  # Customize as needed
    engine = connect_to_db()

    # Load all Championship matches for the selected season
    season_matches = pd.read_sql(
    "SELECT * FROM vw_championship_match_data",
    con=engine
)

    # Preprocess scores
    score_split = season_matches["score"].str.replace(" ", "", regex=False).str.split(":", expand=True)
    season_matches["home_goals"] = pd.to_numeric(score_split[0], errors="coerce")
    season_matches["away_goals"] = pd.to_numeric(score_split[1], errors="coerce")

    # Merge with team_data to get both teams per match
    team_scores_df = pd.merge(
        team_data,
        season_matches[["matchId", "home_goals", "away_goals", "score"]],
        on="matchId", how="inner"
    )

    # Infer home/away status
    team_scores_df["home_away"] = team_scores_df.apply(
        lambda row: "home" if row["scores.fulltime"] == row["home_goals"] else "away",
        axis=1
    )

    # Assign points
    def assign_points(row):
        if row["home_away"] == "home":
            return 3 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 0
        else:
            return 3 if row["away_goals"] > row["home_goals"] else 1 if row["away_goals"] == row["home_goals"] else 0

    team_scores_df["points"] = team_scores_df.apply(assign_points, axis=1)

    # Group and sort all teams by points
    team_points = (
        team_scores_df
        .groupby(["teamId", "teamName"], as_index=False)["points"]
        .sum()
        .sort_values(by="points", ascending=False)
    )

    # Save top 5 if needed
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
        for team in filtered_team_options:
            checked = team in selected_team_names
            if st.checkbox(team, value=checked, key=f"team_{team}"):
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
            FROM vw_championship_player_stats ps
            JOIN vw_championship_player_data pd 
                ON ps.playerId = pd.playerId AND ps.matchId = pd.matchId
            JOIN vw_championship_match_data md 
                ON ps.matchId = md.matchId
            JOIN vw_championship_team_data td 
                ON ps.teamId = td.teamId AND ps.matchId = td.matchId
            WHERE pd.position IN ({','.join([f"'{pos}'" for pos in position_codes])})
            AND ps.teamId IN ({selected_team_ids_str})
        """
        players_full = pd.read_sql(query, connect_to_db())


        if players_full.empty:
            st.warning("No players found for this position and selected teams.")
        else:
            # Calculate minutesPlayed from your function or external join
            minutes_df = player_data[["playerId", "matchId", "minutesPlayed"]].copy()
            # --- Fix dtypes ---
            players_full["playerId"] = players_full["playerId"].astype(str)
            players_full["matchId"] = players_full["matchId"].astype(str)
            minutes_df["playerId"] = minutes_df["playerId"].astype(str)
            minutes_df["matchId"] = minutes_df["matchId"].astype(str)

            players_full = pd.merge(players_full, minutes_df, on=["playerId", "matchId"], how="left")

            # Convert startDate before filtering
            players_full["startDate"] = pd.to_datetime(players_full["startDate"], errors="coerce")

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

            # ✅ Now apply date filtering safely
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
            team_data["teamId"] = team_data["teamId"].astype(str)

            # Merge to get team name
            summary_comparison_df = pd.merge(summary_comparison_df, team_data[["teamId", "teamName"]].drop_duplicates(), on="teamId", how="left")

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

    # ✅ FIX: Ensure logged-in player is included in filtered_players
    logged_player_id = str(player_id)
    logged_player_name = player_name
    logged_player_matches = player_data[
        (player_data["playerId"] == logged_player_id) &
        (player_data["matchId"].isin(filtered_players["matchId"].unique()))
    ][["playerId", "matchId"]].drop_duplicates()

    if not logged_player_matches.empty:
        for _, row in logged_player_matches.iterrows():
            if not ((filtered_players["playerId"] == row["playerId"]) & (filtered_players["matchId"] == row["matchId"])).any():
                filtered_players = pd.concat([
                    filtered_players,
                    pd.DataFrame([{"playerId": row["playerId"], "matchId": row["matchId"]}])
                ], ignore_index=True)

    # Proceed to load player stats and events
    player_ids = filtered_players["playerId"].unique().tolist()
    match_ids = filtered_players["matchId"].unique().tolist()

    # --- Load player_stats and event_data for selected players ---
    player_placeholders = ','.join(['%s'] * len(player_ids))
    match_placeholders = ','.join(['%s'] * len(match_ids))

    query_stats = f"""
        SELECT * FROM vw_championship_player_stats
        WHERE playerId IN ({player_placeholders})
        AND matchId IN ({match_placeholders})
    """
    stats_df = pd.read_sql(query_stats, con=connect_to_db(), params=tuple(player_ids + match_ids))

    query_events = f"""
        SELECT * FROM vw_championship_event_data
        WHERE playerId IN ({player_placeholders})
        AND matchId IN ({match_placeholders})
    """
    events_df = pd.read_sql(query_events, con=connect_to_db(), params=tuple(player_ids + match_ids))

    # --- Convert IDs to string for safe joins ---
    stats_df["playerId"] = stats_df["playerId"].astype(str)
    stats_df["matchId"] = stats_df["matchId"].astype(str)
    events_df["playerId"] = events_df["playerId"].astype(str)
    events_df["matchId"] = events_df["matchId"].astype(str)

    # --- Compute metrics using shared logic ---
    summary_metrics_df = process_player_comparison_metrics(stats_df, events_df, player_position)

    # --- Merge playerName and teamName for chart display ---
    summary_metrics_df = pd.merge(
        summary_metrics_df,
        summary_comparison_df[["playerId", "playerName", "teamName"]],
        on="playerId", how="left"
    )

    # --- KPI Comparison Charts ---
    st.info("📊 Comparison by KPI")

    for kpi in position_kpi_map.get(player_position, []):
        if kpi not in summary_metrics_df.columns:
            continue

        chart_data = summary_metrics_df[["playerName", "teamName", kpi]].dropna()
        chart_data["color"] = chart_data["playerName"].apply(lambda x: "#FFD700" if x == player_name else "#d3d3d3")
        season_avg = chart_data[kpi].mean()

        with st.container(border=True):
            st.markdown(f"**{metric_labels.get(kpi, kpi)}**")

            tooltip = [
                alt.Tooltip("playerName:N", title="Player"),
                alt.Tooltip(kpi + ":Q", title=metric_labels.get(kpi, kpi), format=".2f")
            ]

            bar = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("playerName:N", sort="-y", title="Player"),
                y=alt.Y(f"{kpi}:Q", title=metric_labels.get(kpi, kpi)),
                color=alt.Color("color:N", scale=None),
                tooltip=tooltip
            )

            avg_line = alt.Chart(pd.DataFrame({"y": [season_avg]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
            avg_text = alt.Chart(pd.DataFrame({
                "x": [chart_data["playerName"].iloc[-1]],
                "y": [season_avg],
                "label": [f"Avg: {season_avg:.1f}"]
            })).mark_text(align="left", baseline="bottom", dy=-5, color="red").encode(
                x=alt.X("x:N"),
                y=alt.Y("y:Q"),
                text="label:N"
            )

            chart = (bar + avg_line + avg_text).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
