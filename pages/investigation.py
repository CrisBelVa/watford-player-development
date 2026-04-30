import os
from typing import Dict, Any

import pandas as pd
import streamlit as st

from db_utils import connect_to_db, get_player_position, prepare_player_data_with_minutes
from player_ids import normalize_whoscored_player_id
from utils.sheets_client import GoogleSheetsClient


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "img")
LOGO_PATH = os.path.join(IMG_DIR, "watford_logo.png")

st.set_page_config(
    page_title="Investigation",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)


if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be logged in to view this page.")
    st.stop()

if st.session_state.get("user_type") != "staff":
    st.warning("Only staff users can access this page.")
    st.stop()

staff_role = str(st.session_state.get("staff_info", {}).get("role", "")).strip().lower()
if staff_role not in {"admin", "administrator"}:
    st.warning("Only administrators can access the Investigation page.")
    st.stop()


def get_sheets_client() -> GoogleSheetsClient:
    cache_key = "_investigation_sheets_client"
    cached_client = st.session_state.get(cache_key)
    if isinstance(cached_client, GoogleSheetsClient):
        return cached_client
    client = GoogleSheetsClient()
    st.session_state[cache_key] = client
    return client


@st.cache_data(show_spinner=False)
def load_players_index() -> pd.DataFrame:
    players_df = None
    sheets_client = get_sheets_client()
    if sheets_client.is_configured():
        try:
            players_df = sheets_client.read_players_df()
        except Exception:
            players_df = None

    if players_df is None:
        local_candidates = [
            os.path.join(BASE_DIR, "data", "watford_players_login_info.csv"),
            os.path.join(BASE_DIR, "data", "watford_players_login_info.xlsx"),
            os.path.join(BASE_DIR, "watford_players_login_info.xlsx"),
        ]
        for path in local_candidates:
            if os.path.exists(path):
                if path.endswith(".csv"):
                    players_df = pd.read_csv(path, dtype={"playerId": "string"})
                else:
                    players_df = pd.read_excel(path, converters={"playerId": lambda x: str(x).strip() if pd.notna(x) else None})
                break

    if players_df is None or players_df.empty:
        return pd.DataFrame(columns=["playerName", "playerId", "internal_position", "activo"])

    players_df = players_df.copy()
    players_df.columns = [str(c).strip() for c in players_df.columns]
    if "playerName" not in players_df.columns:
        players_df["playerName"] = ""
    if "playerId" not in players_df.columns:
        players_df["playerId"] = None
    if "internal_position" not in players_df.columns:
        players_df["internal_position"] = None
    if "activo" not in players_df.columns:
        players_df["activo"] = 1

    players_df["playerName"] = players_df["playerName"].astype(str).str.strip()
    players_df["playerId"] = players_df["playerId"].astype("string").where(players_df["playerId"].notna(), None)
    players_df["playerId"] = players_df["playerId"].apply(lambda value: value.strip() if isinstance(value, str) else value)
    players_df["playerId"] = players_df["playerId"].apply(normalize_whoscored_player_id)
    players_df["activo"] = pd.to_numeric(players_df["activo"], errors="coerce").fillna(1).astype(int)

    players_df = players_df[players_df["playerName"] != ""].copy()
    players_df = players_df.drop_duplicates(subset=["playerId", "playerName"], keep="first")
    players_df["label"] = players_df["playerName"]
    players_df.loc[players_df["activo"] != 1, "label"] = players_df["label"] + " (Inactive)"
    players_df = players_df.sort_values("label", key=lambda s: s.str.lower()).reset_index(drop=True)
    return players_df


@st.cache_data(show_spinner=False, ttl=300)
def load_player_investigation(player_id: str) -> Dict[str, Any]:
    engine = connect_to_db()
    if engine is None:
        raise RuntimeError("Could not connect to the database.")

    player_df = pd.read_sql(
        """
        SELECT playerId, playerName, matchId, teamId, position, isFirstEleven,
               subbedInExpandedMinute, subbedOutExpandedMinute
        FROM player_data
        WHERE playerId = %s
        """,
        con=engine,
        params=(player_id,),
    )

    if player_df.empty:
        return {
            "player_df": pd.DataFrame(),
            "event_df": pd.DataFrame(),
            "match_df": pd.DataFrame(),
            "table_df": pd.DataFrame(),
            "direct_position": "Unknown",
            "scoped_mode_position": None,
        }

    player_df = prepare_player_data_with_minutes(player_df)
    for col in ["playerId", "matchId", "teamId"]:
        if col in player_df.columns:
            player_df[col] = player_df[col].astype("string")

    match_ids = player_df["matchId"].dropna().astype(str).unique().tolist()
    placeholders = ",".join(["%s"] * len(match_ids))

    match_df = pd.read_sql(
        f"""
        SELECT matchId, startDate, season, competition
        FROM match_data
        WHERE matchId IN ({placeholders})
        """,
        con=engine,
        params=tuple(match_ids),
    )
    if not match_df.empty:
        match_df["matchId"] = pd.to_numeric(match_df["matchId"], errors="coerce").astype("Int64").astype("string")
        match_df["startDate"] = pd.to_datetime(match_df["startDate"], errors="coerce")

    team_df = pd.read_sql(
        f"""
        SELECT team_id, team_name, match_id
        FROM team_data
        WHERE match_id IN ({placeholders})
        """,
        con=engine,
        params=tuple(match_ids),
    )
    if not team_df.empty:
        team_df = team_df.rename(columns={"team_id": "teamId", "team_name": "teamName", "match_id": "matchId"})
        for col in ["teamId", "matchId"]:
            team_df[col] = pd.to_numeric(team_df[col], errors="coerce").astype("Int64").astype("string")

    event_df = pd.read_sql(
        f"""
        SELECT playerId, matchId, position, teamName, oppositionTeamName
        FROM event_data
        WHERE playerId = %s AND matchId IN ({placeholders})
        """,
        con=engine,
        params=(player_id, *match_ids),
    )
    if not event_df.empty:
        event_df["playerId"] = event_df["playerId"].astype("string")
        event_df["matchId"] = event_df["matchId"].astype("string")

    player_with_dates = player_df.merge(match_df, on="matchId", how="left")
    direct_position = get_player_position(player_with_dates, event_df, player_id)

    table_df = player_with_dates.copy()
    if not team_df.empty:
        player_team_lookup = (
            team_df.drop_duplicates(subset=["matchId", "teamId"])[["matchId", "teamId", "teamName"]]
        )
        table_df = table_df.merge(player_team_lookup, on=["matchId", "teamId"], how="left")

    if not event_df.empty:
        event_summary = (
            event_df.dropna(subset=["position"])
            .groupby("matchId")["position"]
            .agg(lambda values: ", ".join(sorted({str(v).strip().upper() for v in values if str(v).strip()})))
            .reset_index(name="eventPositions")
        )
        opposition_lookup = (
            event_df.groupby("matchId")[["teamName", "oppositionTeamName"]]
            .first()
            .reset_index()
        )
        table_df = table_df.merge(event_summary, on="matchId", how="left")
        table_df = table_df.merge(opposition_lookup, on="matchId", how="left")
    else:
        table_df["eventPositions"] = None
        table_df["teamName"] = table_df.get("teamName")
        table_df["oppositionTeamName"] = None

    table_df["position"] = table_df["position"].astype("string").fillna("")
    non_sub_scope = table_df[~table_df["position"].isin(["", "SUB", "Sub", "NONE", "NAN"])].copy()
    scoped_mode_position = None
    if not non_sub_scope.empty:
        scoped_mode_position = non_sub_scope["position"].astype(str).str.upper().value_counts().index[0]

    table_df["matchDate"] = pd.to_datetime(table_df["startDate"], errors="coerce")
    table_df["started"] = table_df["isFirstEleven"].fillna(0).astype(int).map({1: "Yes", 0: "No"})
    table_df["minutesPlayed"] = pd.to_numeric(table_df["minutesPlayed"], errors="coerce").fillna(0).round(0).astype(int)
    table_df = table_df.rename(
        columns={
            "playerName": "Player",
            "matchDate": "Date",
            "competition": "Competition",
            "season": "Season",
            "teamName": "Team",
            "oppositionTeamName": "Opponent",
            "position": "playerDataPosition",
        }
    )

    display_columns = [
        "Date",
        "Season",
        "Competition",
        "matchId",
        "Team",
        "Opponent",
        "playerDataPosition",
        "eventPositions",
        "started",
        "minutesPlayed",
        "subbedInExpandedMinute",
        "subbedOutExpandedMinute",
    ]
    for column in display_columns:
        if column not in table_df.columns:
            table_df[column] = None
    table_df = table_df[display_columns].sort_values(["Date", "matchId"], ascending=[False, False]).reset_index(drop=True)

    return {
        "player_df": player_df,
        "event_df": event_df,
        "match_df": match_df,
        "table_df": table_df,
        "direct_position": direct_position,
        "scoped_mode_position": scoped_mode_position,
    }


st.title("Investigation")
st.caption("Admin-only workspace to inspect raw match and position data for a player.")

players_df = load_players_index()
if players_df.empty:
    st.error("No players could be loaded.")
    st.stop()

selected_label = st.selectbox(
    "Search player",
    options=players_df["label"].tolist(),
    index=None,
    placeholder="Type a player name...",
)

if not selected_label:
    st.info("Select a player to inspect his match-by-match position history.")
    st.stop()

selected_row = players_df.loc[players_df["label"] == selected_label].iloc[0]
selected_player_id = selected_row.get("playerId")
selected_player_name = selected_row.get("playerName", "")

if not selected_player_id or str(selected_player_id).strip() == "":
    st.error("This player does not have a WhoScored playerId assigned.")
    st.stop()

try:
    investigation = load_player_investigation(str(selected_player_id))
except Exception as exc:
    st.error(f"Could not load player investigation data: {exc}")
    st.stop()

table_df = investigation["table_df"]
player_df = investigation["player_df"]
event_df = investigation["event_df"]

st.subheader(selected_player_name)
meta_cols = st.columns(4)
meta_cols[0].metric("WhoScored playerId", str(selected_player_id))
meta_cols[1].metric("Matches found", int(len(table_df)))
meta_cols[2].metric("Dashboard position", str(investigation["direct_position"]))
meta_cols[3].metric("Comparison mode", str(investigation["scoped_mode_position"] or "Unknown"))

if pd.notna(selected_row.get("internal_position")) and str(selected_row.get("internal_position")).strip():
    st.caption(f"Internal position override: `{selected_row.get('internal_position')}`")
else:
    st.caption("Internal position override: none")

position_summary_cols = st.columns(2)
with position_summary_cols[0]:
    st.markdown("**player_data position counts**")
    if player_df.empty or "position" not in player_df.columns:
        st.info("No player_data positions found.")
    else:
        pd_counts = (
            player_df["position"]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
            .str.upper()
            .value_counts()
            .rename_axis("position")
            .reset_index(name="matches")
        )
        st.dataframe(pd_counts, use_container_width=True, hide_index=True)

with position_summary_cols[1]:
    st.markdown("**event_data position counts**")
    if event_df.empty or "position" not in event_df.columns:
        st.info("No event_data positions found.")
    else:
        ed_counts = (
            event_df["position"]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
            .str.upper()
            .value_counts()
            .rename_axis("position")
            .reset_index(name="events")
        )
        st.dataframe(ed_counts, use_container_width=True, hide_index=True)

st.markdown("**Match list and positions**")
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.caption(
    "Useful for debugging comparison buckets: `Dashboard position` comes from `get_player_position()`, "
    "while `Comparison mode` reflects the most frequent non-sub `player_data` position in the filtered comparison scope."
)
