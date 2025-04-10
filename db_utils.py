from sqlalchemy import create_engine
import pandas as pd
from typing import List

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


def get_top5_aml_players(start_date, end_date, match_data, team_data, player_stats, player_data):
    """
    Returns AML players from top 5 teams during the filtered period,
    including calculated KPIs ready for comparison dashboard.

    Parameters:
    - start_date, end_date: date filter range
    - match_data, team_data, player_stats, player_data: DataFrames

    Returns:
    - DataFrame with all AML players from top 5 teams + metric_keys KPIs
    """
    import pandas as pd

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



def calculate_kpis_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
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
