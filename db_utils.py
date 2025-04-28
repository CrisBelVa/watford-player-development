import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import List, Tuple
import streamlit as st
import altair as alt
from dotenv import load_dotenv
import os

def connect_to_db():
    load_dotenv()

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    if not all([user, password, host, port, database]):
        st.error("Database credentials are missing.")
        return None

    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
    return engine


def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except:
                continue
    return df

def get_player_position(player_data_df, event_data_df, player_id, player_name):
    """
    Returns the dominant position for a player. If 'Sub', finds the most frequent real position played using event_data.
    """
    position_map = {
    "GK": "Goalkeeper",

    "DC": "Center Back",
    "DMC": "Defensive Midfielder",
    "DL": "Left Back",
    "DML": "Left Back",
    "DR": "Right Back",
    "DMR": "Right Back",

    "MC": "Midfielder",
    "ML": "Midfielder",
    "MR": "Midfielder",

    "AMC": "Attacking Midfielder",
    "AML": "Left Winger",
    "FWL": "Left Winger",

    "AMR": "Right Winger",
    "FWR": "Right Winger",

    "FW": "Striker",

    "Sub": "Substitute"
}

    try:
        raw_position = player_data_df[player_data_df["playerId"] == player_id]["position"].values[0]
    except IndexError:
        return "Unknown"

    # If not Sub, return mapped position
    if raw_position != "Sub":
        return position_map.get(raw_position, "Unknown")

    # Sub: find most frequent real position, excluding 'Sub'
    df = event_data_df.copy()
    df = df[(df["playerId"] == player_id) & (df["position"].notnull()) & (df["position"] != "Sub")]

    if df.empty:
        return "Unknown"

    most_common = df["position"].value_counts().idxmax()
    return position_map.get(most_common, most_common)


def process_player_metrics(player_stats, event_data, player_id, player_name):
    """
    Cleans and computes performance metrics for the logged-in player, including extended KPIs
    for all positions.
    """
    # --- Safety Check ---
    if player_stats is None or player_stats.empty:
        st.error("🚨 No player stats available for this player.")
        return pd.DataFrame()

    if event_data is None or event_data.empty:
        st.error("🚨 No event data available for this player.")
        return pd.DataFrame()

    # --- Step 1: Filter by player ---
    player_stats = player_stats[player_stats['playerId'] == player_id].copy()
    event_data = event_data[
        (event_data['playerId'] == player_id) &
        (event_data['playerName'].str.lower().str.contains(player_name.lower(), na=False))
    ].copy()

    # --- Step 2: Clean numeric columns ---
    metric_cols = player_stats.columns.drop(['playerId', 'playerName', 'matchId', 'field', 'teamId', 'teamName'], errors='ignore')
    for col in metric_cols:
        player_stats[col] = player_stats[col].astype(str).str.replace(',', '.', regex=False)
        player_stats[col] = pd.to_numeric(player_stats[col], errors='coerce')

    for col in ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY', 'value_Length', 'xG', 'xA', 'ps_xG']:
        if col in event_data.columns:
            event_data[col] = event_data[col].astype(str).str.replace(',', '.', regex=False)
            event_data[col] = pd.to_numeric(event_data[col], errors='coerce')

    # --- Step 3: Calculate event-based metrics ---
    def calculate_event_metrics(df):
        df = df.copy()

        passes_into_penalty_area = df[
            (df['value_PassEndX'] >= 94) &
            (df['value_PassEndY'].between(21, 79))
        ].groupby(['playerId', 'matchId']).size().rename('passes_into_penalty_area')

        carries_into_final_third = df[
            (df['x'] < 66.7) & (df['endX'] >= 66.7)
        ].groupby(['playerId', 'matchId']).size().rename('carries_into_final_third')

        carries_into_penalty_area = df[
            (df['endX'] >= 94) & (df['endY'].between(21, 79))
        ].groupby(['playerId', 'matchId']).size().rename('carries_into_penalty_area')

        goals = df[df['type_displayName'] == 'Goal'].groupby(['playerId', 'matchId']).size().rename('goals')
        assists = df[df['value_IntentionalAssist'] == 1].groupby(['playerId', 'matchId']).size().rename('assists')
        crosses = df[df['value_Cross'] == 1].groupby(['playerId', 'matchId']).size().rename('crosses')

        long_passes = df[df['value_Length'] >= 30]
        long_passes_total = long_passes.groupby(['playerId', 'matchId']).size().rename('long_passes_total')
        long_passes_success = long_passes[long_passes['outcomeType_displayName'] == 'Successful'].groupby(['playerId', 'matchId']).size().rename('long_passes_success')
        long_pass_pct = (long_passes_success / long_passes_total.replace(0, np.nan) * 100).rename('long_pass_pct')

        progressive_pass_distance = df[
            (df['type_displayName'] == 'Pass') & (df['value_Length'] > 10)
        ].groupby(['playerId', 'matchId'])['value_Length'].sum().rename('progressive_pass_distance')

        progressive_carry_distance = df[
            (df['type_displayName'] == 'Carry') & ((df['endX'] - df['x']) > 10)
        ].assign(distance=lambda d: d['endX'] - d['x']) \
         .groupby(['playerId', 'matchId'])['distance'].sum().rename('progressive_carry_distance')

        recoveries = df[df['type_displayName'] == 'Recovery'].groupby(['playerId', 'matchId']).size().rename('recoveries')
        interceptions = df[df['type_displayName'] == 'Interception'].groupby(['playerId', 'matchId']).size().rename('interceptions')
        clearances = df[df['type_displayName'] == 'Clearance'].groupby(['playerId', 'matchId']).size().rename('clearances')

        defensive_actions_outside_box = df[
            (df['x'] > 25) &
            (df['type_displayName'].isin(['Tackle', 'Interception', 'Clearance']))
        ].groupby(['playerId', 'matchId']).size().rename('def_actions_outside_box')

        shot_creation_actions = df[df['value_ShotAssist'] > 0].groupby(['playerId', 'matchId']).size().rename('shot_creation_actions')

        xG = df.groupby(['playerId', 'matchId'])['xG'].sum().rename('xG')
        xA = df.groupby(['playerId', 'matchId'])['xA'].sum().rename('xA')
        ps_xG = df.groupby(['playerId', 'matchId'])['ps_xG'].sum().rename('ps_xG')

        return pd.concat([
            passes_into_penalty_area,
            carries_into_final_third,
            carries_into_penalty_area,
            goals, assists, crosses,
            long_pass_pct,
            progressive_pass_distance,
            progressive_carry_distance,
            recoveries, interceptions, clearances,
            defensive_actions_outside_box,
            shot_creation_actions,
            xG, xA, ps_xG
        ], axis=1).fillna(0).reset_index()

    event_metrics = calculate_event_metrics(event_data)

    # --- Step 4: Merge stats + events ---
    metrics_summary = pd.merge(
        player_stats,
        event_metrics,
        on=["playerId", "matchId"],
        how="left"
    ).fillna(0)

    # --- Step 5: Create Derived KPIs ---
    metrics_summary['pass_completion_pct'] = (metrics_summary['passesAccurate'] / metrics_summary['passesTotal'].replace(0, np.nan)) * 100
    metrics_summary['aerial_duel_pct'] = (metrics_summary['aerialsWon'] / metrics_summary['aerialsTotal'].replace(0, np.nan)) * 100
    metrics_summary['take_on_success_pct'] = (metrics_summary['dribblesWon'] / metrics_summary['dribblesAttempted'].replace(0, np.nan)) * 100
    metrics_summary['shots_on_target_pct'] = (metrics_summary['shotsOnTarget'] / metrics_summary['shotsTotal'].replace(0, np.nan)) * 100
    metrics_summary['tackle_success_pct'] = (metrics_summary['tackleSuccessful'] / metrics_summary['tacklesTotal'].replace(0, np.nan)) * 100
    metrics_summary['throwin_accuracy_pct'] = (metrics_summary['throwInsAccurate'] / metrics_summary['throwInsTotal'].replace(0, np.nan)) * 100

    metrics_summary["key_passes"] = metrics_summary["passesKey"]
    metrics_summary["goal_creating_actions"] = metrics_summary["passesKey"] + metrics_summary["dribblesWon"]
    metrics_summary["shot_creating_actions"] = metrics_summary["shotsTotal"] + metrics_summary["passesKey"]

    # --- Step 6: Final Cleanup ---
    percent_cols = [
        'pass_completion_pct', 'aerial_duel_pct', 'take_on_success_pct',
        'shots_on_target_pct', 'tackle_success_pct', 'throwin_accuracy_pct', 'long_pass_pct'
    ]
    metrics_summary[percent_cols] = metrics_summary[percent_cols].round(1)

    metrics_summary = metrics_summary.drop_duplicates(subset=["matchId"], keep="last").reset_index(drop=True)

    return metrics_summary



def prepare_player_data_with_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans subbedIn/subbedOut columns, converts to float,
    and calculates minutes played for each row.
    """
    df['subbedInExpandedMinute'] = (
        df['subbedInExpandedMinute']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .replace('None', pd.NA)
    )
    df['subbedOutExpandedMinute'] = (
        df['subbedOutExpandedMinute']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .replace('None', pd.NA)
    )

    df['subbedInExpandedMinute'] = pd.to_numeric(df['subbedInExpandedMinute'], errors='coerce')
    df['subbedOutExpandedMinute'] = pd.to_numeric(df['subbedOutExpandedMinute'], errors='coerce')

    def get_minutes(row):
        if row['isFirstEleven'] == 1:
            if pd.notna(row['subbedOutExpandedMinute']):
                return row['subbedOutExpandedMinute']
            else:
                return 90.0
        elif pd.notna(row['subbedInExpandedMinute']):
            if pd.notna(row['subbedOutExpandedMinute']):
                return row['subbedOutExpandedMinute'] - row['subbedInExpandedMinute']
            else:
                return 90.0 - row['subbedInExpandedMinute']
        else:
            return 0.0

    df['minutesPlayed'] = df.apply(get_minutes, axis=1)
    return df

@st.cache_data(ttl=3600)
def load_player_data(player_id, player_name):
    engine = connect_to_db()

    player_stats = pd.read_sql(
        "SELECT * FROM player_stats WHERE playerId = %s",
        con=engine,
        params=(player_id,)
    )

    match_data = pd.read_sql("SELECT * FROM match_data", engine)
    player_data = pd.read_sql(
        "SELECT * FROM player_data WHERE playerId = %s",
        con=engine,
        params=(player_id,)
    )
    team_data = pd.read_sql("SELECT * FROM team_data", engine)

    # Prepare minutes played
    player_data = prepare_player_data_with_minutes(player_data)

    # Aggregates
    total_minutes = player_data['minutesPlayed'].sum()
    games_as_starter = player_data['isFirstEleven'].fillna(0).sum()

    return match_data, player_data, team_data, player_stats, total_minutes, games_as_starter


@st.cache_data(ttl=300)
def load_event_data_for_matches(player_id, match_ids):
    """
    Loads event_data for the given player and match IDs.
    """
    engine = connect_to_db()

    if not match_ids:
        return pd.DataFrame()

    query = f"""
    SELECT * FROM event_data
    WHERE playerId = %s
    AND matchId IN ({','.join(['%s' for _ in match_ids])})
    """

    params = tuple([player_id] + match_ids)  # 🛠️ Fix here

    event_data = pd.read_sql(query, con=engine, params=params)

    return event_data


