import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
from datetime import timedelta
from PIL import Image
from db_utils import connect_to_db
from db_utils import load_player_data 
from db_utils import prepare_player_data_with_minutes
from db_utils import get_top5_aml_players, calculate_kpis_for_comparison
from math import ceil
from typing import Tuple
from sqlalchemy import create_engine


# Redirect if not logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be logged in to view this page.")
    st.stop()

# Page title
st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon="img/watford_logo.png",  # Favicon
    layout="wide"
)

# Get player info
player_id = st.session_state.player_id
player_name = st.session_state.player_name

# Your dashboard
logo = Image.open("img/watford_logo.png")
st.image(logo, width=100)
st.title(f"{player_name}")

# --- Load Data ---

event_data, match_data, player_data, team_data, player_stats, total_minutes, games_as_starter = load_player_data(player_id, player_name)


def process_player_metrics(player_stats, event_data, player_id, player_name):
    """
    Cleans and computes performance metrics for the logged-in player
    """
    # Filter data
    player_stats = player_stats[player_stats['playerId'] == player_id].copy()
    event_data = event_data[event_data['playerName'].str.lower().str.contains(player_name.lower(), na=False)].copy()

    # Clean numeric columns
    metric_cols = [
        'passesAccurate', 'passesTotal', 'passesKey',
        'aerialsWon', 'aerialsTotal', 'dribblesWon', 'dribblesAttempted',
        'shotsOnTarget', 'shotsTotal'
    ]
    for col in metric_cols:
        player_stats[col] = (
            player_stats[col].astype(str).str.replace(',', '.', regex=False)
        )
        player_stats[col] = pd.to_numeric(player_stats[col], errors='coerce')

    # Clean event_data columns
    event_data['playerName'] = event_data['playerName'].astype(str).str.lower()
    for col in ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY']:
        event_data[col] = event_data[col].astype(str).str.replace(',', '.', regex=False)
        event_data[col] = pd.to_numeric(event_data[col], errors='coerce')

    # Compute advanced metrics from event_data
    def calculate_event_metrics(df):
        df = df.copy()
        passes_into_penalty_area = df[
            (df['value_PassEndX'] >= 94) &
            (df['value_PassEndY'] >= 21) & (df['value_PassEndY'] <= 79)
        ].groupby('matchId').size().rename('passes_into_penalty_area')

        carries_into_final_third = df[
            (df['x'] < 66.7) & (df['endX'] >= 66.7)
        ].groupby('matchId').size().rename('carries_into_final_third')

        carries_into_penalty_area = df[
            (df['endX'] >= 94) & (df['endY'] >= 21) & (df['endY'] <= 79)
        ].groupby('matchId').size().rename('carries_into_penalty_area')

        goals = df[df['type_displayName'] == 'Goal'].groupby('matchId').size().rename('goals')

        shot_creation_actions = df[df['value_ShotAssist'] > 0].groupby('matchId').size().rename('shot_creation_actions')

        return pd.concat([
            passes_into_penalty_area,
            carries_into_final_third,
            carries_into_penalty_area,
            goals,
            shot_creation_actions
        ], axis=1).fillna(0).reset_index()

    event_metrics = calculate_event_metrics(event_data)

        # Merge and compute percentages
    metrics_summary = pd.merge(event_metrics, player_stats, on='matchId', how='left').fillna(0)
    metrics_summary['pass_completion_pct'] = (metrics_summary['passesAccurate'] / metrics_summary['passesTotal'].replace(0, np.nan)) * 100
    metrics_summary['aerial_duel_pct'] = (metrics_summary['aerialsWon'] / metrics_summary['aerialsTotal'].replace(0, np.nan)) * 100
    metrics_summary['take_on_success_pct'] = (metrics_summary['dribblesWon'] / metrics_summary['dribblesAttempted'].replace(0, np.nan)) * 100
    metrics_summary['shots_on_target_pct'] = (metrics_summary['shotsOnTarget'] / metrics_summary['shotsTotal'].replace(0, np.nan)) * 100

    # Add additional metrics used in the UI
    metrics_summary["key_passes"] = metrics_summary["passesKey"]
    metrics_summary["goal_creating_actions"] = metrics_summary["passesKey"] + metrics_summary["dribblesWon"]
    metrics_summary["shot_creating_actions"] = metrics_summary["shotsTotal"] + metrics_summary["passesKey"]

    # Round for clarity
    metrics_summary[['pass_completion_pct', 'aerial_duel_pct', 'take_on_success_pct', 'shots_on_target_pct']] = \
        metrics_summary[['pass_completion_pct', 'aerial_duel_pct', 'take_on_success_pct', 'shots_on_target_pct']].round(1)
    
    # Drop potential matchId duplicates caused by merges
    metrics_summary = metrics_summary.drop_duplicates(subset="matchId", keep="last").reset_index(drop=True)

    return metrics_summary

metrics_summary = process_player_metrics(player_stats, event_data, player_id, player_name)

# Step 6 – Extract player details
player_info = player_data[player_data["playerId"] == player_id].iloc[0]

# Extract and organize details
age = player_info["age"]
shirt_number = player_info["shirtNo"]
height = player_info["height"]
weight = player_info["weight"]
games_played = metrics_summary.shape[0]

# Labels and values for display
labels = [
    "Age", "Shirt Number", "Height", "Weight", "Games Played",
    "Games as Starter", "Minutes Played"
]
values = [
    age, shirt_number, height, weight, games_played,
    int(games_as_starter), int(total_minutes)
]

# Styled Player Info Cards
with st.container():
    st.markdown("""<style>
        .player-box {
            background-color: #1e1e1e;
            padding: 0.6rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.25);
        }
        .player-label {
            font-size: 0.9rem;
            color: #fcec03;
        }
        .player-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: white;
        }
    </style>""", unsafe_allow_html=True)

    cols = st.columns(len(labels))
    for i in range(len(labels)):
        with cols[i]:
            st.markdown(f"""
                <div class='player-box'>
                    <div class='player-label'>{labels[i]}</div>
                    <div class='player-value'>{values[i]}</div>
                </div>
            """, unsafe_allow_html=True)
    
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

def get_player_position(player_data_df, player_id):
    position_map = {
        "GK": "Goalkeeper", "CB": "Center Back", "LB": "Left Back", "RB": "Right Back",
        "CM": "Midfielder", "CDM": "Defensive Midfielder", "CAM": "Attacking Midfielder",
        "LW": "Left Winger", "RW": "Right Winger", "ST": "Striker", "Sub": "Substitute"
    }
    raw_position = player_data_df[player_data_df["playerId"] == player_id]["position"].values[0]
    return position_map.get(raw_position, "Unknown")

def get_player_stats(player_stats_df, player_id):
    """
    Retorna todos os jogos e estatísticas do jogador.
    """
    return player_stats_df[player_stats_df["playerId"] == player_id]

def get_player_match_stats(player_stats_df, player_id, match_id):
    """
    Retorna as estatísticas do jogador em um jogo específico.
    """
    return player_stats_df[
        (player_stats_df["playerId"] == player_id) &
        (player_stats_df["matchId"] == match_id)
    ]

def format_with_commas(value):
    if isinstance(value, float):
        return f"{value:,.1f}"
    elif isinstance(value, int):
        return f"{value:,}"
    else:
        return str(value)


st.sidebar.header("Time Filters")

def add_match_dates(df, match_data_df):
    """
    Adds a 'matchDate' column to the DataFrame based on matchId using match_data_df.
    Returns updated DataFrame and date range.
    """
    # Map matchId to startDate
    match_dates = dict(zip(match_data_df["matchId"], pd.to_datetime(match_data_df["startDate"], errors="coerce")))
    
    df = df.copy()
    df["matchDate"] = df["matchId"].map(match_dates)

    # Drop rows where matchDate could not be mapped
    df = df.dropna(subset=["matchDate"])

    # Get min/max dates
    min_date = df["matchDate"].min().date()
    max_date = df["matchDate"].max().date()

    return df, min_date, max_date

# Aplicar datas ao metrics_summary
metrics_summary, min_date, max_date = add_match_dates(metrics_summary, match_data)

# 4 Lat games filter
last_4_games = metrics_summary.sort_values("matchDate").drop_duplicates("matchId").tail(4)
default_start_date = last_4_games["matchDate"].min().date()
default_end_date = last_4_games["matchDate"].max().date()

start_date = st.sidebar.date_input("Start date", default_start_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", default_end_date, min_value=min_date, max_value=max_date)

# Aplicar filtro de intervalo de datas
mask = (metrics_summary["matchDate"].dt.date >= start_date) & (metrics_summary["matchDate"].dt.date <= end_date)
filtered_df = metrics_summary.loc[mask].sort_values("matchDate")

# Define once globally
metric_keys = [
    "pass_completion_pct", "key_passes", "aerial_duel_pct",
    "take_on_success_pct", "goal_creating_actions", "shot_creating_actions",
    "shots_on_target_pct", "passes_into_penalty_area", "carries_into_final_third",
    "carries_into_penalty_area", "goals"
]

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
    "goals": "Goals"
}

# Mapping of metrics to types
metric_type_map = {
    "pass_completion_pct": "percentage",
    "aerial_duel_pct": "percentage",
    "take_on_success_pct": "percentage",
    "shots_on_target_pct": "percentage",
    "key_passes": "per_match",
    "goal_creating_actions": "per_match",
    "shot_creating_actions": "per_match",
    "passes_into_penalty_area": "per_match",
    "carries_into_final_third": "per_match",
    "carries_into_penalty_area": "per_match",
    "goals": "per_match"
}

# Tooltip fields for each metric
metric_tooltip_fields = {
    "pass_completion_pct": ["passesTotal", "passesAccurate"],
    "aerial_duel_pct": ["aerialsTotal", "aerialsWon"],
    "take_on_success_pct": ["dribblesAttempted", "dribblesWon"],
    "shots_on_target_pct": ["shotsTotal", "shotsOnTarget"],
    "key_passes": ["passesKey"],
    "goal_creating_actions": ["passesKey", "dribblesWon"],
    "shot_creating_actions": ["shotsTotal", "passesKey"],
    "passes_into_penalty_area": [],
    "carries_into_final_third": [],
    "carries_into_penalty_area": [],
    "goals": []
}


def calculate_delta(filtered_df: pd.DataFrame, full_df: pd.DataFrame, column: str) -> Tuple[float, float]:
    """
    Calculates the delta value between the filtered matches (e.g., last 3) and the full season.
    For percentage metrics: uses mean.
    For count metrics: uses avg per match.
    """
    metric_type = metric_type_map.get(column, "per_match")  # default to per_match

    if filtered_df.empty or full_df.empty:
        return 0.0, 0.0

    if metric_type == "percentage":
        filtered_value = filtered_df[column].mean()
        season_value = full_df[column].mean()
    elif metric_type == "per_match":
        filtered_value = filtered_df[column].sum() / len(filtered_df)
        season_value = full_df[column].sum() / len(full_df)
    else:
        return 0.0, 0.0

    delta = filtered_value - season_value
    delta_percent = (delta / season_value * 100) if season_value != 0 else 0

    return round(delta, 1), round(delta_percent, 1)

#Metric Card    

def display_metric_card(col, title, value, filtered_df, full_df, column, color=None):
    with col:
        with st.container(border=True):
            delta, delta_percent = calculate_delta(filtered_df, full_df, column)
            delta_str = f"{delta:+.1f} ({delta_percent:+.1f}%)"
            st.metric(label=title, value=format_with_commas(value), delta=delta_str)

# --- Native Watford-Styled Section Selector ---

st.sidebar.header("Select Visualization")

section = st.sidebar.radio(
    "Go to section:",
    options=["Overview Stats", "Trends Stats", "Player Comparison"],
    index=0,
    key="selected_section"
)

#Logout Button
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

logout_container = st.sidebar.container()
with logout_container:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.pop("player_id", None)
        st.session_state.pop("player_name", None)
        st.rerun()

if section == "Overview Stats":
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.info("Main Performance Stats")

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


    # Create readable match labels
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

    # Apply match filter
    filtered_df = filtered_df[filtered_df["matchId"].isin(st.session_state.selected_match_ids)]

    # --- Aggregate Metrics ---
    aggregated_metrics = {}
    for key in metric_keys:
        if key in [
            "goals", "passes_into_penalty_area", "carries_into_final_third",
            "carries_into_penalty_area", "key_passes", "goal_creating_actions",
            "shot_creating_actions"
        ]:
            aggregated_metrics[key] = int(filtered_df[key].sum())
        else:
            aggregated_metrics[key] = round(filtered_df[key].mean(), 1)

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
    st.markdown("""
        <style>
        .streamlit-expanderHeader {
            font-size: 1.7rem !important;
            font-weight: bold;
            color: #fcec03 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("Player Stats"):
        st.dataframe(
            filtered_df.sort_values("matchDate").reset_index(drop=True).style.set_properties(**{
                'background-color': '#1e1e1e',
                'color': 'white',
                'text-align': 'center'
            }).set_table_styles([
                {'selector': 'th', 'props': [('color', '#fcec03'), ('font-weight', 'bold'), ('text-align', 'center')]}
            ]),
            use_container_width=True
        )

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

    # Sort by date
    trends_df = trends_df.sort_values("matchDate")

    # Create readable match labels
    trends_df["match_label"] = trends_df["matchDate"].dt.strftime("%Y-%m-%d") + " vs " + trends_df["oppositionTeamName"]

    # --- Match Filter Styled Like Excel ---
    with st.expander("Filter by Match (click to hide)", expanded=True):
        match_options = trends_df[["matchId", "match_label", "matchDate"]].drop_duplicates().sort_values("matchDate")
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
            checkbox = st.checkbox(row["match_label"], value=checked, key=f"trends_match_{row['matchId']}")
            if checkbox:
                selected_match_ids.append(row["matchId"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.selected_match_ids = selected_match_ids

    # Apply match filter
    trends_df = trends_df[trends_df["matchId"].isin(st.session_state.selected_match_ids)]

    # Create label for X-axis
    trends_df["opponent_label"] = trends_df["matchDate"].dt.strftime("%b %d") + " - " + trends_df["oppositionTeamName"]

    # Loop through metrics and display chart in container
    for key in metric_keys:
        chart_data = trends_df[["opponent_label", "matchDate", key] + metric_tooltip_fields.get(key, [])].dropna().copy()

        if chart_data.empty:
            continue

        # Season average from full dataset (metrics_summary)
        season_avg = metrics_summary[key].mean()

        with st.container(border=True):
            st.markdown(f"**{metric_labels[key]}**")

            # Prepare tooltip list
            tooltip_fields = [
                alt.Tooltip("opponent_label:N", title="Match"),
                alt.Tooltip(f"{key}:Q", title=metric_labels[key], format=".2f")
            ]
            for extra_field in metric_tooltip_fields.get(key, []):
                if extra_field in chart_data.columns:
                    tooltip_fields.append(
                        alt.Tooltip(f"{extra_field}:Q", title=extra_field.replace("_", " ").title())
                    )

            # Build bar chart + average rule
            chart = alt.Chart(chart_data).mark_bar(color="#fcec03").encode(
                x=alt.X("opponent_label:N", title="Match", sort=chart_data["matchDate"].tolist()),
                y=alt.Y(f"{key}:Q", title=metric_labels[key]),
                tooltip=tooltip_fields
            ) + alt.Chart(pd.DataFrame({
                "y": [season_avg],
                "label": [f"Avg. Season {metric_labels[key]}: {season_avg:.1f}"]
            })).mark_rule(color="red", strokeDash=[4, 4]).encode(
                y="y:Q",
                tooltip=alt.Tooltip("label:N", title="")
            )

            st.altair_chart(chart, use_container_width=True)

elif section == "Player Comparison":
    st.info("Top 5 Players in the Competition – Left Wingers (AML)")

    # Conectar ao banco
    engine = connect_to_db()

    # --- Carregar dados necessários ---
    match_data = pd.read_sql("SELECT * FROM match_data", engine)
    team_data = pd.read_sql("SELECT * FROM team_data", engine)
    player_data = pd.read_sql("SELECT * FROM player_data", engine)

    match_data['startDate'] = pd.to_datetime(match_data['startDate'])
    score_split = match_data['score'].str.replace(" ", "", regex=False).str.split(":", expand=True)
    match_data['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
    match_data['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')

    team_data = team_data.merge(
        match_data[['matchId', 'home_goals', 'away_goals']],
        on='matchId', how='left'
    )

    team_data['home_away'] = team_data.apply(
        lambda row: 'home' if row['scores.fulltime'] == row['home_goals'] else 'away', axis=1
    )

    def assign_points(row):
        if row['home_away'] == 'home':
            return 3 if row['home_goals'] > row['away_goals'] else 1 if row['home_goals'] == row['away_goals'] else 0
        else:
            return 3 if row['away_goals'] > row['home_goals'] else 1 if row['away_goals'] == row['home_goals'] else 0

    team_data['points'] = team_data.apply(assign_points, axis=1)
    team_points = team_data.groupby(['teamId', 'teamName'], as_index=False)['points'].sum()
    top5_teams = team_points.sort_values(by='points', ascending=False).head(5)

    # --- Carregar apenas dados dos jogadores top 5 AML ---
    top5_team_ids = top5_teams['teamId'].unique().tolist()
    top5_team_ids_str = ','.join(map(str, top5_team_ids))

    query_top5_players = f"""
        SELECT ps.* FROM player_stats ps
        JOIN player_data pd ON ps.playerId = pd.playerId AND ps.matchId = pd.matchId
        JOIN match_data md ON ps.matchId = md.matchId
        WHERE pd.position = 'AML' AND ps.teamId IN ({top5_team_ids_str})
        AND md.startDate BETWEEN '{start_date}' AND '{end_date}'
    """
    all_player_stats = pd.read_sql(query_top5_players, engine)

    query_player_info = f"""
        SELECT * FROM player_data
        WHERE playerId IN (SELECT DISTINCT playerId FROM player_stats WHERE teamId IN ({top5_team_ids_str}))
    """
    player_data = pd.read_sql(query_player_info, engine)

    query_event_data = f"""
        SELECT * FROM event_data
        WHERE playerId IN (SELECT DISTINCT playerId FROM player_stats WHERE teamId IN ({top5_team_ids_str}))
    """
    event_data = pd.read_sql(query_event_data, engine)

    # --- Preparar base de jogadores ---
    players_full = pd.merge(
        all_player_stats,
        player_data[['playerId', 'matchId', 'position', 'age', 'shirtNo', 'height', 'weight', 'isFirstEleven', 'subbedInExpandedMinute', 'subbedOutExpandedMinute']],
        on=['playerId', 'matchId'], how='left'
    )
    players_full = pd.merge(players_full, match_data[['matchId', 'startDate']], on='matchId', how='left')

    players_full['ratings_clean'] = players_full['ratings'].astype(str).str.replace(',', '.', regex=False).astype(float)
    players_full['startDate'] = pd.to_datetime(players_full['startDate'])

    # Filter for AML and top5 team and date range
    aml_top5 = players_full[
        (players_full['position'] == 'AML') &
        (players_full['teamId'].isin(top5_teams['teamId'])) &
        (players_full['startDate'].dt.date >= start_date) &
        (players_full['startDate'].dt.date <= end_date)
    ].copy()

    aml_top5['subbedInExpandedMinute'] = aml_top5['subbedInExpandedMinute'].astype(str).str.replace(',', '.', regex=False).replace('None', pd.NA)
    aml_top5['subbedOutExpandedMinute'] = aml_top5['subbedOutExpandedMinute'].astype(str).str.replace(',', '.', regex=False).replace('None', pd.NA)
    aml_top5['subbedInExpandedMinute'] = pd.to_numeric(aml_top5['subbedInExpandedMinute'], errors='coerce')
    aml_top5['subbedOutExpandedMinute'] = pd.to_numeric(aml_top5['subbedOutExpandedMinute'], errors='coerce')

    def get_minutes(row):
        if row['isFirstEleven'] == 1:
            return row['subbedOutExpandedMinute'] if pd.notna(row['subbedOutExpandedMinute']) else 90.0
        elif pd.notna(row['subbedInExpandedMinute']):
            return row['subbedOutExpandedMinute'] - row['subbedInExpandedMinute'] if pd.notna(row['subbedOutExpandedMinute']) else 90.0 - row['subbedInExpandedMinute']
        else:
            return 0.0

    aml_top5['minutesPlayed'] = aml_top5.apply(get_minutes, axis=1)

    summary_df_top5 = (
        aml_top5.groupby(['playerId', 'playerName'], as_index=False)
        .agg(
            teamId=('teamId', 'first'),
            age=('age', 'first'),
            shirtNo=('shirtNo', 'first'),
            height=('height', 'first'),
            weight=('weight', 'first'),
            games_played=('matchId', 'nunique'),
            games_as_starter=('isFirstEleven', 'sum'),
            total_minutes=('minutesPlayed', 'sum'),
            total_rating=('ratings_clean', 'sum')
        )
    )
    summary_df_top5 = pd.merge(summary_df_top5, top5_teams[['teamId', 'teamName']], on='teamId', how='left')
    summary_df_top5 = summary_df_top5.sort_values(by='total_rating', ascending=False).head(5)

    if not summary_df_top5.empty:
        st.markdown("#### 🧽 Top 5 AML Players (Detailed)")
        st.dataframe(summary_df_top5.rename(columns={
            "playerName": "Player",
            "teamName": "Team",
            "age": "Age",
            "shirtNo": "Shirt Number",
            "height": "Height",
            "weight": "Weight",
            "games_played": "Games Played",
            "games_as_starter": "Games as Starter",
            "total_minutes": "Minutes Played"
        })[[
            "Player", "Team", "Age", "Shirt Number", "Height", "Weight",
            "Games Played", "Games as Starter", "Minutes Played"
        ]], use_container_width=True)
    else:
        st.warning("No AML players found in the selected time range.")


    # Prepare metric comparison manually
    top5_kpis = []
    for pid, pname in summary_df_top5[['playerId', 'playerName']].itertuples(index=False):
        player_event_data = event_data[event_data['playerName'].str.lower().str.contains(pname.lower(), na=False)].copy()
        player_stats = all_player_stats[all_player_stats['playerId'] == pid].copy()

        for col in ['passesAccurate', 'passesTotal', 'passesKey', 'aerialsWon', 'aerialsTotal', 'dribblesWon', 'dribblesAttempted', 'shotsOnTarget', 'shotsTotal']:
            player_stats[col] = pd.to_numeric(player_stats[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

        for col in ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY']:
            player_event_data[col] = pd.to_numeric(player_event_data[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

        player_event_data['playerName'] = player_event_data['playerName'].astype(str).str.lower()

        # --- Event-based metrics ---
        passes_into_penalty_area = player_event_data[
            (player_event_data['value_PassEndX'] >= 94) &
            (player_event_data['value_PassEndY'] >= 21) & (player_event_data['value_PassEndY'] <= 79)
        ].shape[0]

        carries_into_final_third = player_event_data[
            (player_event_data['x'] < 66.7) & (player_event_data['endX'] >= 66.7)
        ].shape[0]

        carries_into_penalty_area = player_event_data[
            (player_event_data['endX'] >= 94) & (player_event_data['endY'] >= 21) & (player_event_data['endY'] <= 79)
        ].shape[0]

        goals = player_event_data[player_event_data['type_displayName'] == 'Goal'].shape[0]

        # --- Stats-based metrics ---
        pass_completion_pct = (player_stats['passesAccurate'].sum() / player_stats['passesTotal'].sum()) * 100 if player_stats['passesTotal'].sum() > 0 else 0
        aerial_duel_pct = (player_stats['aerialsWon'].sum() / player_stats['aerialsTotal'].sum()) * 100 if player_stats['aerialsTotal'].sum() > 0 else 0
        take_on_success_pct = (player_stats['dribblesWon'].sum() / player_stats['dribblesAttempted'].sum()) * 100 if player_stats['dribblesAttempted'].sum() > 0 else 0
        shots_on_target_pct = (player_stats['shotsOnTarget'].sum() / player_stats['shotsTotal'].sum()) * 100 if player_stats['shotsTotal'].sum() > 0 else 0

        top5_kpis.append({
            "playerName": pname,
            "pass_completion_pct": round(pass_completion_pct, 1),
            "aerial_duel_pct": round(aerial_duel_pct, 1),
            "take_on_success_pct": round(take_on_success_pct, 1),
            "shots_on_target_pct": round(shots_on_target_pct, 1),
            "key_passes": player_stats['passesKey'].sum(),
            "goal_creating_actions": player_stats['passesKey'].sum() + player_stats['dribblesWon'].sum(),
            "shot_creating_actions": player_stats['shotsTotal'].sum() + player_stats['passesKey'].sum(),
            "passes_into_penalty_area": passes_into_penalty_area,
            "carries_into_final_third": carries_into_final_third,
            "carries_into_penalty_area": carries_into_penalty_area,
            "goals": goals,
            "is_logged_player": False
        })

    # Jogador logado
    logged_kpis = {}
    for kpi in metric_keys:
        if kpi in metric_type_map:
            if metric_type_map[kpi] == "percentage":
                logged_kpis[kpi] = round(filtered_df[kpi].mean(), 1)
            else:
                logged_kpis[kpi] = round(filtered_df[kpi].sum(), 1)

    logged_kpis["playerName"] = player_name
    logged_kpis["is_logged_player"] = True

    # Combine with top 5 KPIs
    comparison_df = pd.DataFrame(top5_kpis + [logged_kpis])

    # Tooltip fields for each metric
    metric_tooltip_fields = {
        "pass_completion_pct": ["passesTotal", "passesAccurate"],
        "aerial_duel_pct": ["aerialsTotal", "aerialsWon"],
        "take_on_success_pct": ["dribblesAttempted", "dribblesWon"],
        "shots_on_target_pct": ["shotsTotal", "shotsOnTarget"],
        "key_passes": ["passesKey"],
        "goal_creating_actions": ["passesKey", "dribblesWon"],
        "shot_creating_actions": ["shotsTotal", "passesKey"],
        "passes_into_penalty_area": [],
        "carries_into_final_third": [],
        "carries_into_penalty_area": [],
        "goals": []
    }

    # Sort comparison data for each KPI and render bar plots
    for kpi in metric_keys:
        if kpi in comparison_df.columns:
            label = metric_labels.get(kpi, kpi.replace("_", " ").title())
            chart_data = comparison_df.sort_values(by=kpi, ascending=False).copy()

            # Set bar colors
            chart_data["bar_color"] = chart_data["is_logged_player"].map(
                lambda x: "#FFD700" if x else "#BBBBBB"  # Yellow for logged-in player, grey for others
            )

            # Compute average for dashed line (excluding logged-in player)
            avg_value = chart_data[~chart_data["is_logged_player"]][kpi].mean()

            # Add hover data based on the metric_tooltip_fields
            tooltip_fields = metric_tooltip_fields.get(kpi, [])
            hover_data = {"playerName": True, kpi: True}
            for field in tooltip_fields:
                if field in chart_data.columns:
                    hover_data[field] = True

            with st.container(border=True):
                st.markdown(f"**{label}**")

                fig = px.bar(
                    chart_data,
                    x="playerName",
                    y=kpi,
                    labels={"playerName": "", kpi: label},
                    color="bar_color",
                    color_discrete_map="identity",
                    height=400,
                    hover_data=hover_data
                )

                # Add average line in red
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(chart_data)-0.5,
                    y0=avg_value,
                    y1=avg_value,
                    line=dict(color="red", width=2, dash="dash"),
                )
                fig.add_annotation(
                    x=len(chart_data)-1,
                    y=avg_value,
                    text=f"Avg: {avg_value:.1f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="red")
                )

                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title=None,
                    showlegend=False,
                    title_x=0.5
                )

                st.plotly_chart(fig, use_container_width=True)
