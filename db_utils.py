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
    for all positions, with corrected Save % and Goals Conceded logic.
    """

    import numpy as np
    import pandas as pd
    import streamlit as st

    # --- Safety check ---
    if player_stats is None or player_stats.empty:
        st.error("🚨 No player stats available for this player.")
        return pd.DataFrame()

    if event_data is None or event_data.empty:
        st.error("🚨 No event data available for this player.")
        return pd.DataFrame()

    # --- Step 1: Filter ---
    player_stats = player_stats[player_stats["playerId"] == player_id].copy()
    event_data_player = event_data[
        (event_data["playerId"] == player_id) &
        (event_data["playerName"].str.lower().str.contains(player_name.lower(), na=False))
    ].copy()

    # --- Step 2: Clean numeric columns ---
    metric_cols = player_stats.columns.drop(
        ['playerId', 'playerName', 'matchId', 'field', 'teamId', 'teamName'],
        errors='ignore'
    )
    for col in metric_cols:
        player_stats[col] = player_stats[col].astype(str).str.replace(',', '.', regex=False)
        player_stats[col] = pd.to_numeric(player_stats[col], errors='coerce')

    for col in ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY', 'value_Length', 'xG', 'xA', 'ps_xG']:
        if col in event_data.columns:
            event_data[col] = event_data[col].astype(str).str.replace(',', '.', regex=False)
            event_data[col] = pd.to_numeric(event_data[col], errors='coerce')

    # --- Step 3: Event-based metrics ---
    def calculate_event_metrics(df):
        df = df.copy()

        passes_into_penalty_area = df[
            (df['value_PassEndX'] >= 94) & (df['value_PassEndY'].between(21, 79))
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

        recoveries = df[df['type_displayName'] == 'BallRecovery'].groupby(['playerId', 'matchId']).size().rename('recoveries')
        interceptions = df[df['type_displayName'] == 'Interception'].groupby(['playerId', 'matchId']).size().rename('interceptions')
        clearances = df[df['type_displayName'] == 'Clearance'].groupby(['playerId', 'matchId']).size().rename('clearances')

        def_actions_outside_box = df[
            (df['x'] > 25) & df['type_displayName'].isin(['Tackle', 'Interception', 'Clearance'])
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
            long_passes_total,
            long_passes_success,
            long_pass_pct,
            progressive_pass_distance,
            progressive_carry_distance,
            recoveries,                # ✅ explicitly included
            interceptions,             # ✅ explicitly included
            clearances,                 # ✅ explicitly included
            def_actions_outside_box,    # ✅ explicitly included
            shot_creation_actions,
            xG, xA, ps_xG
        ], axis=1).fillna(0).reset_index()

    event_metrics = calculate_event_metrics(event_data)

    # --- Step 4: Save % and Goals Conceded Corrected ---
    keeper_team_id = player_stats["teamId"].iloc[0]

    saves = event_data[
        (event_data["type_displayName"] == "SavedShot") &
        (event_data["outcomeType_displayName"] == "Successful") &
        (event_data["playerId"] == player_id)
    ].groupby("matchId").size().rename("saves").reset_index()

    shots_on_target_faced = event_data[
        (event_data["isShot"] == 1) &
        (event_data["type_displayName"].isin(["SavedShot", "Goal"]))
    ].groupby("matchId").size().rename("shots_on_target_faced").reset_index()
    shots_on_target_faced["playerId"] = player_id

    goals_conceded = event_data[
        (event_data["isShot"] == 1) &
        (event_data["type_displayName"] == "Goal") &
        (event_data["teamId"] != keeper_team_id)
    ].groupby("matchId").size().rename("goals_conceded").reset_index()
    goals_conceded["playerId"] = player_id

    save_pct = pd.merge(saves, shots_on_target_faced, on="matchId", how="outer").fillna(0)
    save_pct["playerId"] = player_id
    save_pct["save_pct"] = (save_pct["saves"] / save_pct["shots_on_target_faced"].replace(0, np.nan)) * 100
    save_pct["save_pct"] = save_pct["save_pct"].round(1)

    # --- Step 5: Merge all ---
    metrics_summary = pd.merge(player_stats, event_metrics, on=["playerId", "matchId"], how="left")
    metrics_summary = pd.merge(metrics_summary, save_pct[["playerId", "matchId", "save_pct"]], on=["playerId", "matchId"], how="left")
    metrics_summary = pd.merge(metrics_summary, shots_on_target_faced, on=["playerId", "matchId"], how="left")
    metrics_summary = pd.merge(metrics_summary, goals_conceded, on=["playerId", "matchId"], how="left")
    metrics_summary = metrics_summary.fillna(0)

    # --- Step 6: Derived KPIs ---
    metrics_summary["pass_completion_pct"] = (metrics_summary["passesAccurate"] / metrics_summary["passesTotal"].replace(0, np.nan)) * 100
    metrics_summary["aerial_duel_pct"] = (metrics_summary["aerialsWon"] / metrics_summary["aerialsTotal"].replace(0, np.nan)) * 100
    metrics_summary["take_on_success_pct"] = (metrics_summary["dribblesWon"] / metrics_summary["dribblesAttempted"].replace(0, np.nan)) * 100
    metrics_summary["shots_on_target_pct"] = (metrics_summary["shotsOnTarget"] / metrics_summary["shotsTotal"].replace(0, np.nan)) * 100
    metrics_summary["tackle_success_pct"] = (metrics_summary["tackleSuccessful"] / metrics_summary["tacklesTotal"].replace(0, np.nan)) * 100
    metrics_summary["throwin_accuracy_pct"] = (metrics_summary["throwInsAccurate"] / metrics_summary["throwInsTotal"].replace(0, np.nan)) * 100

    metrics_summary["key_passes"] = metrics_summary["passesKey"]
    metrics_summary["goal_creating_actions"] = metrics_summary["passesKey"] + metrics_summary["dribblesWon"]
    metrics_summary["shot_creating_actions"] = metrics_summary["shotsTotal"] + metrics_summary["passesKey"]

    metrics_summary["claimsHigh"] = player_stats["claimsHigh"]
    metrics_summary["collected"] = player_stats["collected"]
    metrics_summary["totalSaves"] = player_stats["totalSaves"]

    # --- Step 7: Clean up ---
    percent_cols = [
        "pass_completion_pct", "aerial_duel_pct", "take_on_success_pct",
        "shots_on_target_pct", "tackle_success_pct", "throwin_accuracy_pct",
        "long_pass_pct", "save_pct"
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

@st.cache_data(ttl=3600)
def get_top5_players_by_position(
    start_date: str,
    end_date: str,
    position: str,
    player_stats: pd.DataFrame,
    player_data: pd.DataFrame,
    match_data: pd.DataFrame,
    team_data: pd.DataFrame,
    event_data: pd.DataFrame,
):
    import pandas as pd
    import numpy as np

    # Position groups based on your DB values
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

    position_codes = reverse_position_map.get(position, [])
    if not position_codes:
        raise ValueError(f"Unsupported position: {position}")

    # Use full match_data to compute top teams
    match_data = match_data.copy()
    match_data["startDate"] = pd.to_datetime(match_data["startDate"])
    score_split = match_data["score"].str.replace(" ", "", regex=False).str.split(":", expand=True)
    match_data["home_goals"] = pd.to_numeric(score_split[0], errors="coerce")
    match_data["away_goals"] = pd.to_numeric(score_split[1], errors="coerce")

    team_data = pd.merge(team_data, match_data[["matchId", "home_goals", "away_goals"]], on="matchId", how="left")
    team_data["home_away"] = team_data.apply(
        lambda row: "home" if row["scores.fulltime"] == row["home_goals"] else "away", axis=1
    )

    def assign_points(row):
        if row["home_away"] == "home":
            return 3 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 0
        else:
            return 3 if row["away_goals"] > row["home_goals"] else 1 if row["away_goals"] == row["home_goals"] else 0

    team_data["points"] = team_data.apply(assign_points, axis=1)
    top_teams = team_data.groupby(["teamId", "teamName"], as_index=False)["points"].sum().sort_values("points", ascending=False).head(5)
    top_team_ids = top_teams["teamId"].tolist()

    # Merge player info
    full = pd.merge(
        player_stats,
        player_data[[
            "playerId", "matchId", "position", "age", "shirtNo",
            "height", "weight", "isFirstEleven", "subbedInExpandedMinute", "subbedOutExpandedMinute"
        ]],
        on=["playerId", "matchId"], how="left"
    )
    full = pd.merge(full, match_data[["matchId", "startDate"]], on="matchId", how="left")
    full["startDate"] = pd.to_datetime(full["startDate"])

    print("🎯 Checking position values before filtering:")
    print(full["position"].value_counts())
    print("🎯 Target position codes to match:", position_codes)

    # Filter by top 5 team and relevant position(s)
    filtered = full[
        (full["position"].isin(position_codes)) &
        (full["teamId"].isin(top_team_ids))
    ].copy()

    # Minutes played
    filtered["subbedInExpandedMinute"] = filtered["subbedInExpandedMinute"].astype(str).str.replace(",", ".").replace("None", pd.NA)
    filtered["subbedOutExpandedMinute"] = filtered["subbedOutExpandedMinute"].astype(str).str.replace(",", ".").replace("None", pd.NA)
    filtered["subbedInExpandedMinute"] = pd.to_numeric(filtered["subbedInExpandedMinute"], errors="coerce")
    filtered["subbedOutExpandedMinute"] = pd.to_numeric(filtered["subbedOutExpandedMinute"], errors="coerce")

    def get_minutes (row):
        if row["isFirstEleven"] == 1:
            return row["subbedOutExpandedMinute"] if pd.notna(row["subbedOutExpandedMinute"]) else 90.0
        elif pd.notna(row["subbedInExpandedMinute"]):
            return row["subbedOutExpandedMinute"] - row["subbedInExpandedMinute"] if pd.notna(row["subbedOutExpandedMinute"]) else 90.0 - row["subbedInExpandedMinute"]
        else:
            return 0.0

    filtered["minutesPlayed"] = filtered.apply(get_minutes, axis=1)

    # Clean numerics
    filtered["ratings_clean"] = pd.to_numeric(filtered["ratings"].astype(str).str.replace(",", "."), errors="coerce")

    # Aggregate player info
    summary_df_top5 = (
        filtered.groupby(["playerId", "playerName"], as_index=False)
        .agg(
            teamId=("teamId", "first"),
            age=("age", "first"),
            shirtNo=("shirtNo", "first"),
            height=("height", "first"),
            weight=("weight", "first"),
            matches_played=("matchId", "nunique"),
            games_as_starter=("isFirstEleven", "sum"),
            total_minutes=("minutesPlayed", "sum"),
            total_rating=("ratings_clean", "sum")
        )
    )
    summary_df_top5 = pd.merge(summary_df_top5, top_teams[["teamId", "teamName"]], on="teamId", how="left")
    summary_df_top5 = summary_df_top5.sort_values(by="total_rating", ascending=False).head(5)

    # Debug
    if summary_df_top5.empty:
        print("🚨 No players found for position:", position)
        print("👉 Mapped to codes:", position_codes)
        print("👉 Filtered DF shape:", filtered.shape)
        print("👉 Top teams:", top_team_ids)

    return summary_df_top5, filtered





def calculate_kpis_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a DataFrame of aggregated player stats (one row per player) with advanced KPIs
    aligned with process_player_metrics, for consistent comparison in the Watford dashboard.

    Parameters:
    - df: DataFrame with aggregated totals per player

    Returns:
    - Enriched DataFrame with calculated KPIs
    """
    df = df.copy()

    # Convert to numeric
    numeric_cols = [
        "passesAccurate", "passesTotal", "passesKey",
        "aerialsTotal", "aerialsWon",
        "dribblesAttempted", "dribblesWon",
        "shotsTotal", "shotsOnTarget",
        "matches_played", "goals",
        "passes_into_penalty_area", "carries_into_final_third",
        "carries_into_penalty_area", "xG", "xA", "ps_xG",
        "long_pass_pct", "tackle_success_pct", "recoveries",
        "interceptions", "clearances", "def_actions_outside_box",
        "shot_creation_actions"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", ".", regex=False).replace(["None", "nan", "NaN"], pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["matches_played"] = df["matches_played"].replace(0, np.nan)

    # --- Percentage Metrics ---
    df["pass_completion_pct"] = (df["passesAccurate"] / df["passesTotal"].replace(0, np.nan)) * 100
    df["aerial_duel_pct"] = (df["aerialsWon"] / df["aerialsTotal"].replace(0, np.nan)) * 100
    df["take_on_success_pct"] = (df["dribblesWon"] / df["dribblesAttempted"].replace(0, np.nan)) * 100
    df["shots_on_target_pct"] = (df["shotsOnTarget"] / df["shotsTotal"].replace(0, np.nan)) * 100

    # --- Per-match KPIs ---
    df["key_passes"] = df["passesKey"] / df["matches_played"]
    df["goal_creating_actions"] = (df["passesKey"] + df["dribblesWon"]) / df["matches_played"]
    df["shot_creating_actions"] = (df["shotsTotal"] + df["passesKey"]) / df["matches_played"]
    df["passes_into_penalty_area"] = df["passes_into_penalty_area"] / df["matches_played"]
    df["carries_into_final_third"] = df["carries_into_final_third"] / df["matches_played"]
    df["carries_into_penalty_area"] = df["carries_into_penalty_area"] / df["matches_played"]
    df["xG"] = df["xG"] / df["matches_played"]
    df["xA"] = df["xA"] / df["matches_played"]
    df["ps_xG"] = df["ps_xG"] / df["matches_played"]
    df["recoveries"] = df["recoveries"] / df["matches_played"]
    df["interceptions"] = df["interceptions"] / df["matches_played"]
    df["clearances"] = df["clearances"] / df["matches_played"]
    df["def_actions_outside_box"] = df["def_actions_outside_box"] / df["matches_played"]
    df["shot_creation_actions"] = df["shot_creation_actions"] / df["matches_played"]

    # --- Round percentages ---
    percentage_cols = [
        "pass_completion_pct", "aerial_duel_pct",
        "take_on_success_pct", "shots_on_target_pct",
        "long_pass_pct", "tackle_success_pct"
    ]
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].round(1)

    return df.reset_index(drop=True)

def get_top5_teams(match_data, team_data, start_date, end_date):
    match_data["startDate"] = pd.to_datetime(match_data["startDate"], errors="coerce")
    filtered_matches = match_data[(match_data["startDate"].dt.date >= start_date) & (match_data["startDate"].dt.date <= end_date)]

    score_split = filtered_matches["score"].str.replace(" ", "", regex=False).str.split(":", expand=True)
    filtered_matches["home_goals"] = pd.to_numeric(score_split[0], errors="coerce")
    filtered_matches["away_goals"] = pd.to_numeric(score_split[1], errors="coerce")

    team_scores_df = pd.merge(team_data, filtered_matches[["matchId", "home_goals", "away_goals"]], on="matchId", how="inner")
    team_scores_df["home_away"] = team_scores_df.apply(lambda row: "home" if row["scores.fulltime"] == row["home_goals"] else "away", axis=1)

    def assign_points(row):
        if row["home_away"] == "home":
            return 3 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 0
        else:
            return 3 if row["away_goals"] > row["home_goals"] else 1 if row["away_goals"] == row["home_goals"] else 0

    team_scores_df["points"] = team_scores_df.apply(assign_points, axis=1)

    team_points = (
        team_scores_df
        .groupby(["teamId", "teamName"], as_index=False)["points"]
        .sum()
        .sort_values(by="points", ascending=False)
        .head(5)
    )

    return team_points

def get_top5_players(all_player_stats, player_info_df, match_data, team_data, position_codes, top_team_ids, start_date, end_date):
    players_full = pd.merge(
        all_player_stats,
        player_info_df[['playerId', 'matchId', 'position', 'age', 'shirtNo', 'height', 'weight', 'isFirstEleven', 'subbedInExpandedMinute', 'subbedOutExpandedMinute']],
        on=['playerId', 'matchId'], how='left'
    )
    players_full = pd.merge(players_full, match_data[['matchId', 'startDate']], on='matchId', how='left')
    players_full['startDate'] = pd.to_datetime(players_full['startDate'])

    players_full['ratings_clean'] = pd.to_numeric(players_full['ratings'].astype(str).str.replace(",", "."), errors='coerce')
    players_full = players_full.drop_duplicates(subset=["playerId", "matchId"])

    players_full['minutesPlayed'] = players_full.apply(lambda row: get_minutes(row), axis=1)

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
            total_rating=("ratings_clean", "sum")
        )
    )

    summary_df_top5 = pd.merge(summary_df_top5, team_data[["teamId", "teamName"]], on="teamId", how="left")
    summary_df_top5 = summary_df_top5.sort_values(by="total_rating", ascending=False).drop_duplicates(subset="playerId").head(5)

    return summary_df_top5

def get_minutes(row):
    if row['isFirstEleven'] == 1:
        return row['subbedOutExpandedMinute'] if pd.notna(row['subbedOutExpandedMinute']) else 90.0
    elif pd.notna(row['subbedInExpandedMinute']):
        return row['subbedOutExpandedMinute'] - row['subbedInExpandedMinute'] if pd.notna(row['subbedOutExpandedMinute']) else 90.0 - row['subbedInExpandedMinute']
    else:
        return 0.0

def calculate_top5_metrics(merged_top5, sum_metrics, mean_metrics, summary_df_top5):
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

    summary_df_top5_metrics = pd.merge(
        summary_df_top5[["playerId", "playerName", "teamName"]],
        summary_df_top5_metrics,
        on="playerId", how="left"
    )
    return summary_df_top5_metrics

def calculate_logged_player_metrics(metrics_summary, player_id, player_name, team_name, sum_metrics, mean_metrics):
    logged_player_agg = {}
    for metric in sum_metrics:
        if metric in metrics_summary.columns:
            logged_player_agg[metric] = metrics_summary[metric].sum()
    for metric in mean_metrics:
        if metric in metrics_summary.columns:
            logged_player_agg[metric] = metrics_summary[metric].mean()

    logged_player_df = pd.DataFrame({
        "playerId": [player_id],
        "playerName": [player_name],
        "teamName": [team_name],
        **logged_player_agg
    })
    return logged_player_df

def plot_kpi_comparison(combined_metrics_df, metric_keys, metric_labels, player_name):
    st.markdown("### 📊 KPI Comparison")
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
