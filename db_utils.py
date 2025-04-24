import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import List, Tuple

def connect_to_db():
    user = 'admin'
    password = 'mbdsf*2022'
    host = 'dbmbds.cfngygfor8bi.us-east-1.rds.amazonaws.com'
    port = 3306
    database = 'db_watford'

    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
    return engine


def load_all_data():
    engine = connect_to_db()
    event_data = pd.read_sql("SELECT * FROM event_data", engine)
    match_data = pd.read_sql("SELECT * FROM match_data", engine)
    player_data = pd.read_sql("SELECT * FROM player_data", engine)
    team_data = pd.read_sql("SELECT * FROM team_data", engine)
    player_stats = pd.read_sql("SELECT * FROM player_stats", engine)
    return event_data, match_data, player_data, team_data, player_stats


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
    # Filter data
    player_stats = player_stats[player_stats['playerId'] == player_id].copy()
    event_data = event_data[event_data['playerName'].str.lower().str.contains(player_name.lower(), na=False)].copy()

    # Clean numeric columns from player_stats
    metric_cols = player_stats.columns.drop(['playerId', 'playerName', 'matchId', 'field', 'teamId', 'teamName'], errors='ignore')
    for col in metric_cols:
        player_stats[col] = (
            player_stats[col].astype(str).str.replace(',', '.', regex=False)
        )
        player_stats[col] = pd.to_numeric(player_stats[col], errors='coerce')

    # Clean event_data columns
    event_data['playerName'] = event_data['playerName'].astype(str).str.lower()
    for col in ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY', 'value_Length', 'xG', 'xA', 'ps_xG']:
        if col in event_data.columns:
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

        assists = df[df['value_IntentionalAssist'] == 1].groupby('matchId').size().rename('assists')

        crosses = df[df['value_Cross'] == 1].groupby('matchId').size().rename('crosses')

        long_passes = df[df['value_Length'] >= 30]
        long_passes_total = long_passes.groupby('matchId').size().rename('long_passes_total')
        long_passes_success = long_passes[long_passes['outcomeType_displayName'] == 'Successful'].groupby('matchId').size().rename('long_passes_success')
        long_pass_pct = (long_passes_success / long_passes_total.replace(0, np.nan) * 100).rename('long_pass_pct')

        progressive_pass_distance = df[(df['type_displayName'] == 'Pass') & (df['value_Length'] > 10)].groupby('matchId')['value_Length'].sum().rename('progressive_pass_distance')

        progressive_carry_distance = df[
            (df['type_displayName'] == 'Carry') & (df['endX'] - df['x'] > 10)
        ].assign(distance=lambda d: d['endX'] - d['x']) \
        .groupby('matchId')['distance'].sum() \
        .rename('progressive_carry_distance')

        recoveries = df[df['type_displayName'] == 'Recovery'].groupby('matchId').size().rename('recoveries')

        interceptions = df[df['type_displayName'] == 'Interception'].groupby('matchId').size().rename('interceptions')

        clearances = df[df['type_displayName'] == 'Clearance'].groupby('matchId').size().rename('clearances')

        defensive_actions_outside_box = df[
            (df['x'] > 25) & 
            (df['type_displayName'].isin(['Tackle', 'Interception', 'Clearance']))
        ].groupby('matchId').size().rename('def_actions_outside_box')

        shot_creation_actions = df[df['value_ShotAssist'] > 0].groupby('matchId').size().rename('shot_creation_actions')

        xG = df.groupby('matchId')['xG'].sum().rename('xG')
        xA = df.groupby('matchId')['xA'].sum().rename('xA')
        ps_xG = df.groupby('matchId')['ps_xG'].sum().rename('ps_xG')

        return pd.concat([
            passes_into_penalty_area,
            carries_into_final_third,
            carries_into_penalty_area,
            goals,
            assists,
            crosses,
            long_pass_pct,
            progressive_pass_distance,
            progressive_carry_distance,
            recoveries,
            interceptions,
            clearances,
            defensive_actions_outside_box,
            shot_creation_actions,
            xG,
            xA,
            ps_xG
        ], axis=1).fillna(0).reset_index()

    event_metrics = calculate_event_metrics(event_data)

    # Merge and compute derived metrics
    metrics_summary = pd.merge(player_stats, event_metrics, on='matchId', how='left').fillna(0)

    metrics_summary['pass_completion_pct'] = (metrics_summary['passesAccurate'] / metrics_summary['passesTotal'].replace(0, np.nan)) * 100
    metrics_summary['aerial_duel_pct'] = (metrics_summary['aerialsWon'] / metrics_summary['aerialsTotal'].replace(0, np.nan)) * 100
    metrics_summary['take_on_success_pct'] = (metrics_summary['dribblesWon'] / metrics_summary['dribblesAttempted'].replace(0, np.nan)) * 100
    metrics_summary['shots_on_target_pct'] = (metrics_summary['shotsOnTarget'] / metrics_summary['shotsTotal'].replace(0, np.nan)) * 100
    metrics_summary['tackle_success_pct'] = (metrics_summary['tackleSuccessful'] / metrics_summary['tacklesTotal'].replace(0, np.nan)) * 100
    metrics_summary['throwin_accuracy_pct'] = (metrics_summary['throwInsAccurate'] / metrics_summary['throwInsTotal'].replace(0, np.nan)) * 100

    # Add final KPIs
    metrics_summary["key_passes"] = metrics_summary["passesKey"]
    metrics_summary["goal_creating_actions"] = metrics_summary["passesKey"] + metrics_summary["dribblesWon"]
    metrics_summary["shot_creating_actions"] = metrics_summary["shotsTotal"] + metrics_summary["passesKey"]

    # Round selected columns
    percent_cols = [
        'pass_completion_pct', 'aerial_duel_pct', 'take_on_success_pct',
        'shots_on_target_pct', 'tackle_success_pct', 'throwin_accuracy_pct', 'long_pass_pct'
    ]
    metrics_summary[percent_cols] = metrics_summary[percent_cols].round(1)

    # Drop potential matchId duplicates caused by merges
    metrics_summary = metrics_summary.drop_duplicates(subset="matchId", keep="last").reset_index(drop=True)

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

def load_player_data(player_id, player_name):
    engine = connect_to_db()

    player_stats = pd.read_sql(
        "SELECT * FROM player_stats WHERE playerId = %s",
        con=engine,
        params=(player_id,)
    )

    event_data = pd.read_sql(
        "SELECT * FROM event_data WHERE LOWER(playerName) LIKE %s",
        con=engine,
        params=(f"%{player_name.lower()}%",)
    )

    match_data = pd.read_sql("SELECT * FROM match_data", engine)
    player_data = pd.read_sql(
        "SELECT * FROM player_data WHERE playerId = %s",
        engine,
        params=(player_id,)
    )
    team_data = pd.read_sql("SELECT * FROM team_data", engine)

    # Add minutesPlayed
    player_data = prepare_player_data_with_minutes(player_data)

    # Aggregates
    total_minutes = player_data['minutesPlayed'].sum()
    games_as_starter = player_data['isFirstEleven'].fillna(0).sum()

    return event_data, match_data, player_data, team_data, player_stats, total_minutes, games_as_starter


###### CHECK IF WE NEED THIS ###########





###### def get_top5_aml_players(start_date, end_date, match_data, team_data, player_stats, player_data):
    """
    Returns AML players from top 5 teams during the filtered period,
    including calculated KPIs ready for comparison dashboard.

    Parameters:
    - start_date, end_date: date filter range
    - match_data, team_data, player_stats, player_data: DataFrames

    Returns:
    - DataFrame with all AML players from top 5 teams + metric_keys KPIs
    """

    # --- Filter match_data by date ---
    match_data = match_data.copy()
    match_data['startDate'] = pd.to_datetime(match_data['startDate'])
    match_data_filtered = match_data[
        (match_data['startDate'] >= pd.to_datetime(start_date)) &
        (match_data['startDate'] <= pd.to_datetime(end_date))
    ]

    # --- Extract home and away goals from match_data ---
    score_split = match_data_filtered['score'].str.replace(" ", "", regex=False).str.split(":", expand=True)
    match_data_filtered['home_goals'] = pd.to_numeric(score_split[0], errors='coerce')
    match_data_filtered['away_goals'] = pd.to_numeric(score_split[1], errors='coerce')

    # --- Merge team_data with scores ---
    team_data = team_data.copy()
    team_scores_df = pd.merge(
        team_data,
        match_data_filtered[['matchId', 'home_goals', 'away_goals']],
        on='matchId',
        how='left'
    )

    # --- Infer home/away by comparing team score and home_goals ---
    team_scores_df['home_away'] = team_scores_df.apply(
        lambda row: 'home' if row['scores.fulltime'] == row['home_goals'] else 'away',
        axis=1
    )

    # --- Assign points ---
    def assign_points(row):
        if row['home_away'] == 'home':
            if row['home_goals'] > row['away_goals']:
                return 3
            elif row['home_goals'] == row['away_goals']:
                return 1
            else:
                return 0
        else:
            if row['away_goals'] > row['home_goals']:
                return 3
            elif row['away_goals'] == row['home_goals']:
                return 1
            else:
                return 0

    team_scores_df['points'] = team_scores_df.apply(assign_points, axis=1)

    # --- Get top 5 teams by total points ---
    top5_teams = (
        team_scores_df
        .groupby(['teamId', 'teamName'], as_index=False)['points']
        .sum()
        .sort_values(by='points', ascending=False)
        .head(5)
    )

    # --- Merge player_stats with player_data to get position/age/etc ---
    players_full = pd.merge(
        player_stats,
        player_data[['playerId', 'matchId', 'position', 'age', 'shirtNo', 'height', 'weight']],
        on=['playerId', 'matchId'],
        how='left'
    )

    # --- Filter only AML players from top 5 teams ---
    aml_players = players_full[
        (players_full['position'] == 'AML') &
        (players_full['teamId'].isin(top5_teams['teamId']))
    ]

    print("🔎 POSIÇÕES DISPONÍVEIS:", players_full['position'].dropna().unique())
    print("🔎 TEAM IDs DISPONÍVEIS:", players_full['teamId'].dropna().unique())
    print("🔎 PLAYER FULL SHAPE:", players_full.shape)
    print("🔎 TOP5 TEAM IDs:", top5_teams['teamId'].tolist())

    # --- Convert ratings to float ---
    aml_players['ratings_clean'] = (
        aml_players['ratings']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .replace("None", pd.NA)
    )
    aml_players['ratings_clean'] = pd.to_numeric(aml_players['ratings_clean'], errors='coerce')

    # --- Aggregate per player ---
    aml_summary = (
        aml_players
        .groupby(['playerId', 'playerName', 'teamName'], as_index=False)
        .agg(
            average_rating=('ratings_clean', 'mean'),
            total_rating=('ratings_clean', 'sum'),
            matches_played=('matchId', 'nunique'),
            passesAccurate=('passesAccurate', 'sum'),
            passesTotal=('passesTotal', 'sum'),
            passesKey=('passesKey', 'sum'),
            aerialsTotal=('aerialsTotal', 'sum'),
            aerialsWon=('aerialsWon', 'sum'),
            dribblesAttempted=('dribblesAttempted', 'sum'),
            dribblesWon=('dribblesWon', 'sum'),
            shotsTotal=('shotsTotal', 'sum'),
            shotsOnTarget=('shotsOnTarget', 'sum'),
            age=('age', 'first'),
            shirtNo=('shirtNo', 'first'),
            height=('height', 'first'),
            weight=('weight', 'first')
        )
    )

    # --- Add KPI calculations directly ---
    df = aml_summary.copy()

    def to_float(col):
        return pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False).replace("None", pd.NA), errors="coerce").fillna(0)

    df["passesAccurate"] = to_float("passesAccurate")
    df["passesTotal"] = to_float("passesTotal")
    df["passesKey"] = to_float("passesKey")
    df["aerialsTotal"] = to_float("aerialsTotal")
    df["aerialsWon"] = to_float("aerialsWon")
    df["dribblesAttempted"] = to_float("dribblesAttempted")
    df["dribblesWon"] = to_float("dribblesWon")
    df["shotsTotal"] = to_float("shotsTotal")
    df["shotsOnTarget"] = to_float("shotsOnTarget")

    df["pass_completion_pct"] = (df["passesAccurate"] / df["passesTotal"].replace(0, pd.NA)) * 100
    df["aerial_duel_pct"] = (df["aerialsWon"] / df["aerialsTotal"].replace(0, pd.NA)) * 100
    df["take_on_success_pct"] = (df["dribblesWon"] / df["dribblesAttempted"].replace(0, pd.NA)) * 100
    df["shots_on_target_pct"] = (df["shotsOnTarget"] / df["shotsTotal"].replace(0, pd.NA)) * 100

    df["key_passes"] = df["passesKey"] / df["matches_played"].replace(0, pd.NA)
    df["goal_creating_actions"] = (df["passesKey"] + df["dribblesWon"]) / df["matches_played"].replace(0, pd.NA)
    df["shot_creating_actions"] = (df["shotsTotal"] + df["passesKey"]) / df["matches_played"].replace(0, pd.NA)

    df[["pass_completion_pct", "aerial_duel_pct", "take_on_success_pct", "shots_on_target_pct"]] = df[[
        "pass_completion_pct", "aerial_duel_pct", "take_on_success_pct", "shots_on_target_pct"
    ]].round(1)

    return df



########## def calculate_kpis_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of AML players, calculate and add all KPIs used in the dashboard comparison.

    The function assumes the input dataframe is aggregated by player
    (e.g., one row per player with summed totals across matches).
    """
    df = df.copy()

    # --- Define columns to convert from text to float ---
    raw_columns = [
        "passesTotal", "passesAccurate", "passesKey",
        "aerialsTotal", "aerialsWon", "dribblesAttempted", "dribblesWon",
        "shotsTotal", "shotsOnTarget", "goals"
    ]

    # --- Convert European text format to float safely ---
    for col in raw_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace(["None", "nan", "NaN"], pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Avoid division by zero using .replace(0, pd.NA) ---
    df["pass_completion_pct"] = (df["passesAccurate"] / df["passesTotal"].replace(0, pd.NA)) * 100
    df["aerial_duel_pct"] = (df["aerialsWon"] / df["aerialsTotal"].replace(0, pd.NA)) * 100
    df["take_on_success_pct"] = (df["dribblesWon"] / df["dribblesAttempted"].replace(0, pd.NA)) * 100
    df["shots_on_target_pct"] = (df["shotsOnTarget"] / df["shotsTotal"].replace(0, pd.NA)) * 100

    # --- Normalize per match (assumes matches_played is correct) ---
    df["key_passes"] = df["passesKey"] / df["matches_played"].replace(0, pd.NA)
    df["goal_creating_actions"] = (df["passesKey"] + df["dribblesWon"]) / df["matches_played"].replace(0, pd.NA)
    df["shot_creating_actions"] = (df["shotsTotal"] + df["passesKey"]) / df["matches_played"].replace(0, pd.NA)

    # --- Optional: Normalize additional metrics per match ---
    optional_cols = [
        "passes_into_penalty_area", "carries_into_final_third",
        "carries_into_penalty_area", "goals"
    ]

    for col in optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False).replace(["None", "nan"], pd.NA),
                errors="coerce"
            ).fillna(0)
            df[col] = df[col] / df["matches_played"].replace(0, pd.NA)

    # --- Round percentage columns to 1 decimal place ---
    pct_cols = [
        "pass_completion_pct", "aerial_duel_pct",
        "take_on_success_pct", "shots_on_target_pct"
    ]
    df[pct_cols] = df[pct_cols].round(1)

    return df


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

    def get_minutes(row):
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
