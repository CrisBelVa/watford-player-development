import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import List, Tuple, Dict, Any, Optional, Union
import streamlit as st
import altair as alt
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from typing import Dict, Any

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

percentage_formula_map = {
        "pass_completion_pct": ("passesAccurate", "passesTotal"),
        "aerial_duel_pct": ("aerialsWon", "aerialsTotal"),
        "take_on_success_pct": ("dribblesWon", "dribblesAttempted"),
        "shots_on_target_pct": ("shotsOnTarget", "shotsTotal"),
        "tackle_success_pct": ("tackleSuccessful", "tacklesTotal"),
        "throwin_accuracy_pct": ("throwInsAccurate", "throwInsTotal"),
        "long_pass_pct": ("long_passes_success", "long_passes_total"),
    }

def connect_to_db():
    """
    Establish a connection to the database with detailed error handling.
    
    Returns:
        SQLAlchemy engine if connection is successful, None otherwise.
    """
    try:
        # Load environment variables
        load_dotenv()

        # Get database credentials
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = os.getenv("DB_NAME")

        # Check for missing credentials
        missing = []
        if not user: missing.append("DB_USER")
        if not password: missing.append("DB_PASSWORD")
        if not host: missing.append("DB_HOST")
        if not port: missing.append("DB_PORT")
        if not database: missing.append("DB_NAME")
        
        if missing:
            error_msg = f"Missing required database credentials: {', '.join(missing)}\n"
            error_msg += f"Please check your .env file in: {os.path.abspath('.env')}"
            st.error(error_msg)
            return None

        # Escape special characters in password
        safe_password = password.replace('*', '%2A')

        # Create connection string (without password for logging)
        conn_str = f"mysql+pymysql://{user}:*****@{host}:{port}/{database}"
        
        try:
            # Create engine with connection test
            engine = create_engine(f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{database}")
            
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            st.success("✅ Successfully connected to the database")
            return engine
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"❌ Failed to connect to database:\n"
            error_msg += f"Connection string: {conn_str.replace(safe_password, '*****')}\n\n"
            error_msg += f"Error Type: {error_type}\n"
            error_msg += f"Error Details: {str(e)}\n\n"
            error_msg += "Troubleshooting steps:\n"
            error_msg += "1. Verify the database server is running\n"
            error_msg += "2. Check if the credentials in .env are correct\n"
            error_msg += "3. Verify the database user has proper permissions\n"
            error_msg += "4. Check if the database exists and is accessible\n"
            error_msg += "5. Verify the MySQL server allows remote connections if applicable"
            
            st.error(error_msg)
            st.exception(e)  # This will show the full traceback in the Streamlit UI
            return None
            
    except Exception as e:
        st.error(f"❌ Unexpected error while setting up database connection: {str(e)}")
        st.exception(e)
        return None


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
    Returns the dominant position for a player. If 'Sub', finds the most frequent real position
    using event_data. Uses safe dtype normalization to avoid empty selections.
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
        "Sub": "Substitute",
    }

    # ---- normalize types to compare reliably
    pid_str = str(player_id)
    try:
        pdd = player_data_df.copy()
        pdd["playerId"] = pdd["playerId"].astype(str)
    except Exception:
        return "Midfielder"

    # 1) Try player_data.position first
    try:
        raw_position = (
            pdd.loc[pdd["playerId"] == pid_str, "position"]
            .dropna()
            .astype(str)
            .values[0]
        )
    except Exception:
        raw_position = None

    if raw_position and raw_position != "Sub":
        return position_map.get(raw_position, "Midfielder")

    # 2) If 'Sub' or missing, infer from events
    try:
        edd = event_data_df.copy()
        edd["playerId"] = edd["playerId"].astype(str)
        edd = edd[(edd["playerId"] == pid_str)]
        if "position" in edd.columns:
            edd = edd[edd["position"].notna() & (edd["position"] != "Sub")]
            if not edd.empty:
                most_common = edd["position"].astype(str).value_counts().idxmax()
                return position_map.get(most_common, "Midfielder")
    except Exception:
        pass

    # 3) Safe default
    return "Midfielder"



def process_player_metrics(player_stats, event_data, player_id, player_name):
    """
    Aligns with your current schemas:
    - player_stats: has passSuccess, tackleSuccess, etc. (percent values), plus base counts.
    - event_data: no xG/xA columns; compute heuristic xG, ps_xG, and event KPIs.

    Always returns a DataFrame (possibly empty).
    """
    # -------- Guards --------
    try:
        pid = int(player_id)
    except Exception as e:
        st.error(f"🚨 player_id→int failed: {e}")
        return pd.DataFrame()

    if not isinstance(player_stats, pd.DataFrame) or player_stats.empty:
        st.error("🚨 No player_stats available.")
        return pd.DataFrame()

    if not isinstance(event_data, pd.DataFrame) or event_data.empty:
        st.error("🚨 No event_data available.")
        return pd.DataFrame()

    ps = player_stats.copy()
    ev = event_data.copy()

    # Required columns
    for c in ["playerId", "matchId"]:
        if c not in ps.columns:
            st.error(f"🚨 player_stats missing '{c}'")
            return pd.DataFrame()
    if "playerId" not in ev.columns: ev["playerId"] = np.nan
    if "matchId"  not in ev.columns: ev["matchId"]  = np.nan

    # -------- Normalize IDs --------
    ps["playerId"] = pd.to_numeric(ps["playerId"], errors="coerce").astype("Int64")
    ps["matchId"]  = pd.to_numeric(ps["matchId"],  errors="coerce").astype("Int64")
    ev["playerId"] = pd.to_numeric(ev["playerId"], errors="coerce").astype("Int64")
    ev["matchId"]  = pd.to_numeric(ev["matchId"],  errors="coerce").astype("Int64")

    # Filter to this player
    ps = ps[ps["playerId"] == pid].copy()
    # Use only playerId for filtering events (names can differ in casing/spacing)
    ev_player = ev[ev["playerId"] == pid].copy()

    # Quick diagnostics (optional)
    try:
        st.caption(f"📦 player_stats rows (player): {len(ps)} | event_data rows (player): {len(ev_player)}")
    except Exception:
        pass

    # If no rows after filtering, return empty DF with expected join keys
    if ps.empty:
        return pd.DataFrame(columns=["playerId","matchId"]).astype({"playerId":"Int64","matchId":"Int64"})

    # -------- Clean numeric columns in player_stats --------
    # Convert any decimal strings to numeric
    for col in ps.columns:
        if ps[col].dtype == object:
            ps[col] = pd.to_numeric(ps[col].astype(str).str.replace(',', '.', regex=False), errors="ignore")

    # -------- Event metrics (with heuristic xG / ps_xG) --------
    def calculate_event_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["playerId","matchId"]).astype({"playerId":"Int64","matchId":"Int64"})

        df = df.copy()

        # Ensure fields exist with correct dtypes
        for c in ["x","y","endX","endY","value_Length","value_PassEndX","value_PassEndY"]:
            if c not in df.columns: df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Binary flags (coerce to 0/1)
        bin_cols = [
            "isShot","isGoal","value_Penalty","value_BigChance","value_OneOnOne",
            "value_Head","value_LeftFoot","value_RightFoot",
            "value_LowLeft","value_LowRight","value_HighLeft","value_HighRight","value_HighCentre","value_LowCentre",
            "value_ShotAssist","value_Cross","value_IntentionalGoalAssist"
        ]
        for c in bin_cols:
            if c not in df.columns: df[c] = 0
            df[c] = (df[c].astype(str).str.strip().str.lower()
                     .replace({"true":"1","false":"0","yes":"1","no":"0"}))
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        # Strings used for masks
        for c in ["type_displayName","outcomeType_displayName","qualifiers"]:
            if c not in df.columns: df[c] = ""
            df[c] = df[c].astype(str)

        # Identify shots robustly
        shot_mask = (
            (df["isShot"] == 1) |
            (df["type_displayName"].isin(["Shot","SavedShot","Goal","Miss","ShotOnPost"])) |
            (df["outcomeType_displayName"].isin(["Saved","Goal","Off T","Post"]))
        )

        # Geometry heuristics on 0–100 pitch
        dx = (100 - df["x"])
        dy = (df["y"] - 50).abs()
        dist = np.sqrt(np.square(dx) + np.square(dy))
        angle = np.degrees(np.arctan2(5.5, dx.clip(lower=1e-6)))
        in_six  = (df["endX"] >= 94)   & (df["endY"].between(44, 56))
        in_box  = (df["endX"] >= 88.5) & (df["endY"].between(13.6, 54.4))

        # Heuristic xG
        base_xg = np.where(in_six, 0.36,
                    np.where(in_box & (dist < 8), 0.28,
                    np.where(in_box & (dist < 12), 0.18,
                    np.where(in_box & (dist < 18), 0.12,
                    np.where(dist < 25, 0.06, 0.03)))))
        base_xg = np.where(df["value_Penalty"] == 1, 0.76, base_xg)
        base_xg = np.where(df["value_BigChance"] == 1, np.maximum(base_xg, 0.35), base_xg)
        base_xg = np.where(df["value_OneOnOne"] == 1, base_xg + 0.10, base_xg)
        base_xg = np.where(df["value_Head"] == 1, base_xg * 0.75, base_xg)
        base_xg = np.where(angle < 15, base_xg * 0.70, base_xg)
        base_xg = np.where(angle > 60, base_xg * 1.15, base_xg)
        base_xg = np.where(~in_box, np.minimum(base_xg, 0.07), base_xg)
        est_xg  = np.clip(base_xg, 0.01, 0.95)
        est_xg  = np.where(shot_mask, est_xg, 0.0)

        # ps_xG (post-shot)
        on_target = df["type_displayName"].isin(["SavedShot","Goal"]) | (df["outcomeType_displayName"] == "Saved")
        placement_mult = np.where(df["value_HighLeft"] | df["value_HighRight"], 1.25,
                           np.where(df["value_LowLeft"] | df["value_LowRight"], 1.15,
                           np.where(df["value_HighCentre"] | df["value_LowCentre"], 0.85, 1.00)))
        goal_mask = (df["isGoal"] == 1) | (df["type_displayName"] == "Goal")
        est_psxg  = np.where(on_target, est_xg * placement_mult, 0.0)
        est_psxg  = np.where(goal_mask, est_psxg * 1.10, est_psxg)
        est_psxg  = np.clip(est_psxg, 0.0, 1.2)

        # Group helpers
        def gsize(mask):
            return df[mask].groupby(["playerId","matchId"]).size()

        # Your KPIs from events
        passes_into_pa = gsize((df["value_PassEndX"] >= 88.5) & (df["value_PassEndY"].between(13.6, 54.4))).rename("passes_into_penalty_area")
        carries_ft     = gsize((df["x"] < 66.7) & (df["endX"] >= 66.7)).rename("carries_into_final_third")
        carries_pa     = gsize((df["endX"] >= 88.5) & (df["endY"].between(13.6, 54.4))).rename("carries_into_penalty_area")

        goals   = gsize(df["type_displayName"] == "Goal").rename("goals")
        assists = gsize(df["value_IntentionalGoalAssist"] == 1).rename("assists")
        crosses = gsize(df["value_Cross"] == 1).rename("crosses")

        long_passes         = df[df["value_Length"] >= 30]
        long_passes_total   = long_passes.groupby(["playerId","matchId"]).size().rename("long_passes_total")
        long_passes_success = long_passes[long_passes["outcomeType_displayName"] == "Successful"] \
                                .groupby(["playerId","matchId"]).size().rename("long_passes_success")
        long_pass_pct       = (long_passes_success / long_passes_total.replace(0, np.nan) * 100).rename("long_pass_pct")

        progressive_passes = gsize(
            (df["type_displayName"] == "Pass") &
            (df["outcomeType_displayName"] == "Successful") &
            (df["value_Length"] >= 9.11) &
            (df["x"] >= 35) &
            (~df["qualifiers"].str.contains("CornerTaken|Freekick", na=False))
        ).rename("progressive_passes")

        progressive_carry_distance = df[
            (df["type_displayName"] == "Carry") & ((df["endX"] - df["x"]) >= 9.11)
        ].assign(distance=lambda d: d["endX"] - d["x"]).groupby(["playerId","matchId"])["distance"].sum().rename("progressive_carry_distance")

        def_actions_ob = gsize((df["x"] > 25) & (df["type_displayName"].isin(["Tackle","Interception","Clearance"]))).rename("def_actions_outside_box")
        recoveries     = gsize(df["type_displayName"] == "BallRecovery").rename("recoveries")
        sca            = gsize(df["value_ShotAssist"] > 0).rename("shot_creation_actions")

        # Heuristic xG/ps_xG/xA grouped
        xg_series   = pd.Series(est_xg,  index=df.index, name="xG")
        psxg_series = pd.Series(est_psxg,index=df.index, name="ps_xG")
        xG    = xg_series[shot_mask].groupby([df.loc[shot_mask,"playerId"], df.loc[shot_mask,"matchId"]]).sum(min_count=1)
        ps_xG = psxg_series[shot_mask].groupby([df.loc[shot_mask,"playerId"], df.loc[shot_mask,"matchId"]]).sum(min_count=1)
        xA    = df.groupby(["playerId","matchId"])["value_ShotAssist"].sum(min_count=1).rename("xA")

        gca = goals.add(assists, fill_value=0).rename("goal_creating_actions")

        # GK-style metrics
        saves = gsize((df["type_displayName"] == "SavedShot") & (df["outcomeType_displayName"] == "Successful")).rename("saves")
        sot_faced = gsize((df["isShot"] == 1) & (df["type_displayName"].isin(["SavedShot","Goal"]))).rename("shots_on_target_faced")
        goals_conceded = gsize((df["isShot"] == 1) & (df["type_displayName"] == "Goal")).rename("goals_conceded")

        out = pd.concat([
            passes_into_pa, carries_ft, carries_pa,
            goals, assists, crosses,
            long_passes_total, long_passes_success, long_pass_pct,
            progressive_passes, progressive_carry_distance,
            def_actions_ob, recoveries,
            sca, gca,
            xG, xA, ps_xG,
            saves, sot_faced, goals_conceded
        ], axis=1).fillna(0).reset_index()

        # Ensure ids Int64
        for c in ["playerId","matchId"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

        return out

    event_metrics = calculate_event_metrics(ev_player)

    # -------- Merge on keys --------
    metrics_summary = pd.merge(ps, event_metrics, on=["playerId","matchId"], how="left").fillna(0)

    # -------- Derived KPIs / formatting --------
    # Prefer base counts when available; otherwise use % fields from player_stats
    def pct_from_counts(n, d):
        return (metrics_summary[n] / metrics_summary[d].replace(0, np.nan) * 100)

    # Pass completion
    if {"passesAccurate","passesTotal"}.issubset(metrics_summary.columns):
        metrics_summary["pass_completion_pct"] = pct_from_counts("passesAccurate","passesTotal")
    elif "passSuccess" in metrics_summary.columns:
        metrics_summary["pass_completion_pct"] = pd.to_numeric(metrics_summary["passSuccess"], errors="coerce")
    else:
        metrics_summary["pass_completion_pct"] = 0.0

    # Aerial duels
    if {"aerialsWon","aerialsTotal"}.issubset(metrics_summary.columns):
        metrics_summary["aerial_duel_pct"] = pct_from_counts("aerialsWon","aerialsTotal")
    elif "aerialSuccess" in metrics_summary.columns:
        metrics_summary["aerial_duel_pct"] = pd.to_numeric(metrics_summary["aerialSuccess"], errors="coerce")
    else:
        metrics_summary["aerial_duel_pct"] = 0.0

    # Take-ons
    if {"dribblesWon","dribblesAttempted"}.issubset(metrics_summary.columns):
        metrics_summary["take_on_success_pct"] = pct_from_counts("dribblesWon","dribblesAttempted")
    elif "dribbleSuccess" in metrics_summary.columns:
        metrics_summary["take_on_success_pct"] = pd.to_numeric(metrics_summary["dribbleSuccess"], errors="coerce")
    else:
        metrics_summary["take_on_success_pct"] = 0.0

    # Tackles
    if {"tackleSuccessful","tacklesTotal"}.issubset(metrics_summary.columns):
        metrics_summary["tackle_success_pct"] = pct_from_counts("tackleSuccessful","tacklesTotal")
    elif "tackleSuccess" in metrics_summary.columns:
        metrics_summary["tackle_success_pct"] = pd.to_numeric(metrics_summary["tackleSuccess"], errors="coerce")
    else:
        metrics_summary["tackle_success_pct"] = 0.0

    # Throw-ins
    if {"throwInsAccurate","throwInsTotal"}.issubset(metrics_summary.columns):
        metrics_summary["throwin_accuracy_pct"] = pct_from_counts("throwInsAccurate","throwInsTotal")
    elif "throwInAccuracy" in metrics_summary.columns:
        metrics_summary["throwin_accuracy_pct"] = pd.to_numeric(metrics_summary["throwInAccuracy"], errors="coerce")
    else:
        metrics_summary["throwin_accuracy_pct"] = 0.0

    # Shots on target %
    if {"shotsOnTarget","shotsTotal"}.issubset(metrics_summary.columns):
        metrics_summary["shots_on_target_pct"] = pct_from_counts("shotsOnTarget","shotsTotal")
    else:
        metrics_summary["shots_on_target_pct"] = 0.0

    # Save %
    if {"saves","shots_on_target_faced"}.issubset(metrics_summary.columns):
        metrics_summary["save_pct"] = (metrics_summary["saves"] / metrics_summary["shots_on_target_faced"].replace(0, np.nan)) * 100
    else:
        metrics_summary["save_pct"] = 0.0

    # Aliases / passthroughs
    metrics_summary["key_passes"] = metrics_summary["passesKey"] if "passesKey" in metrics_summary.columns else 0
    if "goal_creating_actions" not in metrics_summary.columns:
        metrics_summary["goal_creating_actions"] = 0
    # Use counts for SCA if present
    if "shot_creation_actions" in metrics_summary.columns and "shotsTotal" in metrics_summary.columns:
        metrics_summary["shot_creating_actions"] = metrics_summary["shot_creation_actions"] + metrics_summary["shotsTotal"]
    elif "shotsTotal" in metrics_summary.columns:
        metrics_summary["shot_creating_actions"] = metrics_summary["shotsTotal"]
    else:
        metrics_summary["shot_creating_actions"] = 0

    # Round percentages
    for c in ["pass_completion_pct","aerial_duel_pct","take_on_success_pct",
              "shots_on_target_pct","tackle_success_pct","throwin_accuracy_pct",
              "long_pass_pct","save_pct"]:
        if c in metrics_summary.columns:
            metrics_summary[c] = pd.to_numeric(metrics_summary[c], errors="coerce").round(1).fillna(0)

    # De-dupe by match (left-most kept)
    if "matchId" in metrics_summary.columns:
        metrics_summary = metrics_summary.drop_duplicates(subset=["matchId"], keep="last").reset_index(drop=True)
    else:
        metrics_summary = metrics_summary.reset_index(drop=True)

    # Final diag
    try:
        st.caption(f"✅ metrics_summary: {metrics_summary.shape[0]} rows, {metrics_summary.shape[1]} cols")
    except Exception:
        pass

    return metrics_summary

def prepare_player_data_with_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize player_data for downstream use and compute minutesPlayed.

    Uses your schema:
      - isFirstEleven (TINYINT 0/1)
      - subbedInExpandedMinute, subbedOutExpandedMinute (SMALLINT, can be NULL)
      - subbedInPeriod_value, subbedOutPeriod_value (TINYINT 1=1st half, 2=2nd, 3/4=ET; optional)

    Minutes logic (simple, robust):
      - Starter (isFirstEleven=1):
          start = 0
          end   = subbedOutExpandedMinute if not null else nominal_end (90 or 120 if ET)
      - Sub (isFirstEleven=0):
          if subbedInExpandedMinute is null → 0 minutes (unused sub)
          else start = subbedInExpandedMinute; end = nominal_end
      - minutesPlayed = max(0, end - start)

    Note: Expanded-minute already accounts for stoppage in many feeds; if not, this is still consistent across matches.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["playerId", "matchId", "teamId", "isFirstEleven", "minutesPlayed"])

    out = df.copy()

    # ---- Coerce IDs to nullable Int64 (handles 12345, "12345", 12345.0, etc.)
    for col in ["playerId", "matchId", "teamId"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # ---- isFirstEleven → 0/1 int
    if "isFirstEleven" not in out.columns:
        out["isFirstEleven"] = 0
    out["isFirstEleven"] = (
        out["isFirstEleven"]
        .astype(str).str.strip().str.lower()
        .replace({"true": "1", "yes": "1", "y": "1", "t": "1",
                  "false": "0", "no": "0", "n": "0", "f": "0"})
    )
    out["isFirstEleven"] = pd.to_numeric(out["isFirstEleven"], errors="coerce").fillna(0).astype(int)

    # ---- Bring in the substitution columns (ensure numeric / NaN)
    for col in ["subbedInExpandedMinute", "subbedOutExpandedMinute"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    for col in ["subbedInPeriod_value", "subbedOutPeriod_value"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    # ---- Nominal match end based on periods observed
    # Default 90; if we ever see periods > 2, assume 120 (extra time).
    # This is per-row; if your dataset guarantees no ET in league games, it’ll stay 90.
    def infer_nominal_end(row):
        # Prefer explicit out period if available
        periods = [row.get("subbedOutPeriod_value"), row.get("subbedInPeriod_value")]
        if any((p is not np.nan and p and p >= 3) for p in periods):
            return 120.0
        return 90.0

    nominal_end = out.apply(infer_nominal_end, axis=1)

    # ---- Compute start/end
    start = np.where(out["isFirstEleven"] == 1, 0.0, out["subbedInExpandedMinute"])
    end   = np.where(out["isFirstEleven"] == 1,
                     np.where(out["subbedOutExpandedMinute"].notna(),
                              out["subbedOutExpandedMinute"],
                              nominal_end),
                     np.where(out["subbedInExpandedMinute"].notna(),
                              nominal_end,
                              0.0))  # unused sub → 0

    # ---- minutesPlayed
    out["minutesPlayed"] = pd.to_numeric(end, errors="coerce") - pd.to_numeric(start, errors="coerce")
    out["minutesPlayed"] = out["minutesPlayed"].clip(lower=0).fillna(0.0)

    # Ensure expected columns exist
    for c in ["teamId"]:
        if c not in out.columns:
            out[c] = pd.Series(dtype="Int64")

    return out


def load_player_data(player_id: str, player_name: Optional[str] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, int]:
    """
    Load all season data for a player.

    Returns:
        match_data         : DataFrame (rows from match_data for matches the player appeared in)
        player_data        : DataFrame (rows from player_data for the player, with minutesPlayed computed)
        team_data          : DataFrame (entire team_data table; untouched)
        player_stats       : DataFrame (rows from player_stats for the player)
        total_minutes      : float (sum of minutesPlayed across player_data rows)
        games_as_starter   : int   (sum of isFirstEleven across player_data rows)

    Notes:
      - Prefers looking up player_stats by playerId; falls back to LOWER(playerName) if needed.
      - Normalizes key id columns to pandas nullable Int64 to avoid dtype-join issues later.
      - Uses prepare_player_data_with_minutes() to compute minutesPlayed from substitution info.
    """
    # 0) Connect
    engine = connect_to_db()
    if not engine:
        raise Exception("Failed to connect to database")

    # 1) Resolve the player exists in player_data by ID (canonical row)
    player_info = pd.read_sql(
        "SELECT * FROM player_data WHERE playerId = %s LIMIT 1",
        con=engine,
        params=(player_id,)
    )
    if player_info.empty:
        raise Exception(f"No player found with ID: {player_id}. Name hint: {player_name}")

    # Optional name from DB (for fallback later)
    name_from_db = player_info.iloc[0].get("playerName")

    # 2) Pull all player_data rows for this player
    player_data = pd.read_sql(
        "SELECT * FROM player_data WHERE playerId = %s",
        con=engine,
        params=(player_id,)
    )

    # Normalize ID columns early to avoid later mismatches
    for col in ["playerId", "matchId", "teamId"]:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors="coerce").astype("Int64")

    # 3) Player stats — prefer ID; fallback to LOWER(name) if ID not present / returns empty
    try:
        player_stats = pd.read_sql(
            "SELECT * FROM player_stats WHERE playerId = %s",
            con=engine,
            params=(player_id,)
        )
    except Exception:
        player_stats = pd.DataFrame()

    if player_stats.empty and (player_name or name_from_db):
        pname = player_name or name_from_db
        # Use case-insensitive name equality; avoids partial contains mismatches
        player_stats = pd.read_sql(
            "SELECT * FROM player_stats WHERE LOWER(playerName) = LOWER(%s)",
            con=engine,
            params=(pname,)
        )

    # Normalize ID columns in player_stats to align with other frames
    if not player_stats.empty:
        for col in ["playerId", "matchId", "teamId"]:
            if col in player_stats.columns:
                player_stats[col] = pd.to_numeric(player_stats[col], errors="coerce").astype("Int64")

        # Coerce numeric %/counts where needed (safe if already numeric)
        # These exist in your schema; we’ll coerce broadly but safely.
        numeric_cols = [
            "ratings","aerialSuccess","dribbleSuccess","passSuccess","tackleSuccess","throwInAccuracy",
            "possession","aerialsTotal","aerialsWon","claimsHigh","clearances","collected",
            "cornersAccurate","cornersTotal","defensiveAerials","dispossessed","dribbledPast",
            "dribblesAttempted","dribblesLost","dribblesWon","errors","foulsCommited","interceptions",
            "offensiveAerials","offsidesCaught","parriedDanger","parriedSafe","passesAccurate","passesKey",
            "passesTotal","shotsBlocked","shotsOffTarget","shotsOnPost","shotsOnTarget","shotsTotal",
            "tackleSuccessful","tackleUnsuccesful","tacklesTotal","throwInsAccurate","throwInsTotal",
            "totalSaves","touches"
        ]
        for c in numeric_cols:
            if c in player_stats.columns:
                # convert “87.00” (decimal) or strings cleanly
                player_stats[c] = pd.to_numeric(player_stats[c], errors="coerce")

    # 4) Compute minutesPlayed on player_data (uses your substitution fields)
    #    Make sure this helper is defined ABOVE this function.
    player_data = prepare_player_data_with_minutes(player_data)

    # 5) Match data for this player's matches
    if "matchId" in player_data.columns:
        match_ids = player_data["matchId"].dropna().unique().tolist()
    else:
        match_ids = []

    if match_ids:
        placeholders = ",".join(["%s"] * len(match_ids))
        match_data = pd.read_sql(
            f"SELECT * FROM match_data WHERE matchId IN ({placeholders})",
            con=engine,
            params=tuple(match_ids)
        )
        # Normalize matchId
        if "matchId" in match_data.columns:
            match_data["matchId"] = pd.to_numeric(match_data["matchId"], errors="coerce").astype("Int64")
    else:
        match_data = pd.DataFrame(columns=["matchId"])

    # 6) Team data (full table; you can filter later if needed)
    team_data = pd.read_sql("SELECT * FROM team_data", con=engine)
    # Note: team_data has snake_case columns (match_id, team_id, ...). We leave as-is to avoid breaking callers.

    # 7) Aggregates
    total_minutes = float(player_data.get("minutesPlayed", pd.Series(dtype=float)).fillna(0).sum())
    games_as_starter = int(player_data.get("isFirstEleven", pd.Series(dtype=float)).fillna(0).sum())

    return match_data, player_data, team_data, player_stats, total_minutes, games_as_starter

@st.cache_data(ttl=300)
def load_event_data_for_matches(player_id, match_ids, team_id=None):
    engine = connect_to_db()

    if not isinstance(player_id, str):
        st.error("🚨 player_id must be a string.")
        st.stop()

    if not isinstance(match_ids, list) or not all(isinstance(mid, str) for mid in match_ids):
        st.error("🚨 match_ids must be a list of strings.")
        st.stop()

    if not match_ids:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(match_ids))

    # Use the optimized view
    query = f"""
    SELECT * FROM event_data
    WHERE playerId = %s AND matchId IN ({placeholders})
    """

    params = (player_id, *match_ids)

    if team_id:
        query += " AND teamId = %s"
        params = params + (team_id,)

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

def get_connection():
    """
    Establece y devuelve una conexión a la base de datos.
    
    Returns:
        connection: Objeto de conexión a la base de datos
    """
    try:
        engine = connect_to_db()
        if engine:
            return engine.connect()
        return None
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None


def insert_entrenamiento(jugador_id: int, fecha: str, objetivo: str, resultado: str, 
                        duracion_minutos: int, notas: str = None) -> bool:
    """
    Inserta un nuevo registro de entrenamiento individual.
    
    Args:
        jugador_id (int): ID del jugador
        fecha (str): Fecha del entrenamiento en formato 'YYYY-MM-DD'
        objetivo (str): Objetivo del entrenamiento
        resultado (str): Resultado del entrenamiento
        duracion_minutos (int): Duración en minutos
        notas (str, optional): Notas adicionales
        
    Returns:
        bool: True si la inserción fue exitosa, False en caso contrario
    """
    try:
        engine = connect_to_db()
        if not engine:
            return False
            
        query = """
            INSERT INTO entrenamientos_individuales 
            (jugador_id, fecha, objetivo, resultado, duracion_minutos, notas, created_at)
            VALUES (:jugador_id, :fecha, :objetivo, :resultado, :duracion_minutos, :notas, NOW())
        """
        
        with engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'jugador_id': jugador_id,
                    'fecha': fecha,
                    'objetivo': objetivo,
                    'resultado': resultado,
                    'duracion_minutos': duracion_minutos,
                    'notas': notas
                }
            )
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error en insert_entrenamiento: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False


def insert_meeting(jugador_id: int, fecha: str, tipo: str, titulo: str, 
                  descripcion: str = None, notas: str = None) -> bool:
    """
    Inserta un nuevo registro de reunión.
    
    Args:
        jugador_id (int): ID del jugador
        fecha (str): Fecha de la reunión en formato 'YYYY-MM-DD'
        tipo (str): Tipo de reunión (ej. 'Individual', 'Grupal', 'Técnica')
        titulo (str): Título de la reunión
        descripcion (str, optional): Descripción detallada
        notas (str, optional): Notas adicionales
        
    Returns:
        bool: True si la inserción fue exitosa, False en caso contrario
    """
    try:
        engine = connect_to_db()
        if not engine:
            return False
            
        query = """
            INSERT INTO meetings 
            (jugador_id, fecha, tipo, titulo, descripcion, notas, created_at)
            VALUES (:jugador_id, :fecha, :tipo, :titulo, :descripcion, :notas, NOW())
        """
        
        with engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'jugador_id': jugador_id,
                    'fecha': fecha,
                    'tipo': tipo,
                    'titulo': titulo,
                    'descripcion': descripcion,
                    'notas': notas
                }
            )
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error en insert_meeting: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False


def insert_review_clip(jugador_id: int, fecha: str, titulo: str, descripcion: str,
                      enlace_video: str, duracion_segundos: int,
                      etiquetas: str = None, notas: str = None) -> bool:
    """
    Inserta un nuevo registro de review clip.
    
    Args:
        jugador_id (int): ID del jugador
        fecha (str): Fecha del clip en formato 'YYYY-MM-DD'
        titulo (str): Título del clip
        descripcion (str): Descripción del clip
        enlace_video (str): URL del video
        duracion_segundos (int): Duración en segundos
        etiquetas (str, optional): Etiquetas separadas por comas
        notas (str, optional): Notas adicionales
        
    Returns:
        bool: True si la inserción fue exitosa, False en caso contrario
    """
    try:
        engine = connect_to_db()
        if not engine:
            return False
            
        query = """
            INSERT INTO review_clips 
            (jugador_id, fecha, titulo, descripcion, enlace_video, 
             duracion_segundos, etiquetas, notas, created_at)
            VALUES (:jugador_id, :fecha, :titulo, :descripcion, :enlace_video, 
                   :duracion_segundos, :etiquetas, :notas, NOW())
        """
        
        with engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'jugador_id': jugador_id,
                    'fecha': fecha,
                    'titulo': titulo,
                    'descripcion': descripcion,
                    'enlace_video': enlace_video,
                    'duracion_segundos': duracion_segundos,
                    'etiquetas': etiquetas,
                    'notas': notas
                }
            )
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error en insert_review_clip: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False


def get_player_activities(player_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene todas las actividades de un jugador en un rango de fechas.
    
    Args:
        player_id (int): ID del jugador
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: DataFrame con las actividades del jugador
    """
    try:
        engine = connect_to_db()
        if not engine:
            return pd.DataFrame()
            
        # Consulta para entrenamientos individuales
        query_entrenamientos = """
            SELECT 
                id,
                jugador_id,
                fecha,
                'Entrenamiento' as tipo,
                objetivo as titulo,
                resultado as descripcion,
                NULL as enlace_video,
                duracion_minutos as duracion,
                notas,
                created_at
            FROM entrenamientos_individuales
            WHERE jugador_id = :player_id
            AND fecha BETWEEN :start_date AND :end_date
        """
        
        # Consulta para reuniones
        query_meetings = """
            SELECT 
                id,
                jugador_id,
                fecha,
                CONCAT('Reunión - ', tipo) as tipo,
                titulo,
                descripcion,
                NULL as enlace_video,
                NULL as duracion,
                notas,
                created_at
            FROM meetings
            WHERE jugador_id = :player_id
            AND fecha BETWEEN :start_date AND :end_date
        """
        
        # Consulta para review clips
        query_review_clips = """
            SELECT 
                id,
                jugador_id,
                fecha,
                'Review Clip' as tipo,
                titulo,
                descripcion,
                enlace_video,
                duracion_segundos as duracion,
                CONCAT_WS(' | ', notas, CONCAT('Etiquetas: ', etiquetas)) as notas,
                created_at
            FROM review_clips
            WHERE jugador_id = :player_id
            AND fecha BETWEEN :start_date AND :end_date
        """
        
        params = {
            'player_id': player_id,
            'start_date': start_date,
            'end_date': end_date
        }
        
        with engine.connect() as conn:
            # Ejecutar consultas y combinar resultados
            df_entrenamientos = pd.read_sql(text(query_entrenamientos), conn, params=params)
            df_meetings = pd.read_sql(text(query_meetings), conn, params=params)
            df_review_clips = pd.read_sql(text(query_review_clips), conn, params=params)
            
            # Combinar todos los DataFrames
            df = pd.concat([df_entrenamientos, df_meetings, df_review_clips], ignore_index=True)
            
            # Ordenar por fecha descendente
            if not df.empty:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.sort_values('fecha', ascending=False)
                
                # Formatear fechas para mostrar
                df['fecha_str'] = df['fecha'].dt.strftime('%d/%m/%Y')
                
                # Asegurar que las columnas de duración sean consistentes
                if 'duracion' in df.columns:
                    df['duracion'] = df['duracion'].fillna(0).astype(int)
                
            return df
            
    except Exception as e:
        print(f"Error en get_player_activities: {e}")
        return pd.DataFrame()


def get_all_players() -> pd.DataFrame:
    """
    Devuelve todos los jugadores con su información, incluyendo el campo 'activo'.
    """
    try:
        engine = connect_to_db()
        if not engine:
            return pd.DataFrame()

        query = """
            SELECT 
                id,
                nombre,
                apellido,
                CONCAT(nombre, ' ', apellido) as nombre_completo,
                posicion,
                equipo,
                activo
            FROM jugadores
            ORDER BY nombre, apellido
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    except Exception as e:
        print(f"Error en get_all_players: {e}")
        return pd.DataFrame()


def get_monthly_summary_by_player(month: str) -> pd.DataFrame:
    """
    Obtiene un resumen de actividades por jugador para un mes específico.
    
    Args:
        month (str): Mes en formato 'YYYY-MM'
        
    Returns:
        pd.DataFrame: DataFrame con el resumen de actividades por jugador
    """
    try:
        engine = connect_to_db()
        if not engine:
            return pd.DataFrame()
            
        # Obtener el primer y último día del mes
        start_date = f"{month}-01"
        next_month = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=32)).replace(day=1)
        end_date = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
        
        with engine.connect() as conn:
            # Obtener resumen de entrenamientos por jugador
            query = """
                SELECT 
                    j.id as jugador_id,
                    CONCAT(j.nombre, ' ', j.apellido) as jugador,
                    COUNT(e.id) as entrenamientos
                FROM jugadores j
                LEFT JOIN entrenamientos_individuales e ON j.id = e.jugador_id 
                    AND e.fecha BETWEEN :start_date AND :end_date
                WHERE j.activo = 1
                GROUP BY j.id, j.nombre, j.apellido
            """
            df_entrenamientos = pd.read_sql(
                text(query), 
                conn, 
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            # Obtener resumen de reuniones por jugador
            query = """
                SELECT 
                    j.id as jugador_id,
                    CONCAT(j.nombre, ' ', j.apellido) as jugador,
                    COUNT(m.id) as meetings
                FROM jugadores j
                LEFT JOIN meetings m ON j.id = m.jugador_id 
                    AND m.fecha BETWEEN :start_date AND :end_date
                WHERE j.activo = 1
                GROUP BY j.id, j.nombre, j.apellido
            """
            df_meetings = pd.read_sql(
                text(query), 
                conn, 
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            # Obtener resumen de review clips por jugador
            query = """
                SELECT 
                    j.id as jugador_id,
                    CONCAT(j.nombre, ' ', j.apellido) as jugador,
                    COUNT(rc.id) as review_clips
                FROM jugadores j
                LEFT JOIN review_clips rc ON j.id = rc.jugador_id 
                    AND rc.fecha BETWEEN :start_date AND :end_date
                WHERE j.activo = 1
                GROUP BY j.id, j.nombre, j.apellido
            """
            df_review_clips = pd.read_sql(
                text(query), 
                conn, 
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            # Combinar todos los DataFrames
            dfs = [df_entrenamientos, df_meetings, df_review_clips]
            df_final = None
            
            for df in dfs:
                if not df.empty:
                    if df_final is None:
                        df_final = df
                    else:
                        df_final = df_final.merge(df, on=['jugador_id', 'jugador'], how='outer')
            
            # Si no hay datos, devolver un DataFrame vacío con las columnas esperadas
            if df_final is None or df_final.empty:
                return pd.DataFrame(columns=['jugador_id', 'jugador', 'entrenamientos', 'meetings', 'review_clips'])
                
            # Rellenar valores nulos con 0
            df_final = df_final.fillna(0)
            
            # Convertir columnas numéricas a enteros
            for col in ['entrenamientos', 'meetings', 'review_clips']:
                if col in df_final.columns:
                    df_final[col] = df_final[col].astype(int)
            
            return df_final
            
    except Exception as e:
        print(f"Error en get_monthly_summary_by_player: {e}")
        return pd.DataFrame()


def get_monthly_summary_all_players(months: int = 6) -> pd.DataFrame:
    """
    Obtiene un resumen mensual de actividades para todos los jugadores.
    
    Args:
        months (int): Número de meses a incluir en el resumen (por defecto: 6)
        
    Returns:
        pd.DataFrame: DataFrame con el resumen mensual de actividades
    """
    try:
        engine = connect_to_db()
        if not engine:
            return pd.DataFrame()
            
        # Calcular fecha de inicio (hace 'months' meses)
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=months*30)).strftime('%Y-%m-01')
        end_date = end_date.strftime('%Y-%m-%d')
        
        with engine.connect() as conn:
            # Obtener resumen de entrenamientos por mes
            query = """
                SELECT 
                    DATE_FORMAT(fecha, '%Y-%m-01') as mes,
                    COUNT(*) as total_entrenamientos
                FROM entrenamientos_individuales
                WHERE fecha BETWEEN :start_date AND :end_date
                GROUP BY DATE_FORMAT(fecha, '%Y-%m-01')
            """
            df_entrenamientos = pd.read_sql(
                text(query), 
                conn, 
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            # Obtener resumen de reuniones por mes
            query = """
                SELECT 
                    DATE_FORMAT(fecha, '%Y-%m-01') as mes,
                    COUNT(*) as total_meetings
                FROM meetings
                WHERE fecha BETWEEN :start_date AND :end_date
                GROUP BY DATE_FORMAT(fecha, '%Y-%m-01')
            """
            df_meetings = pd.read_sql(
                text(query), 
                conn, 
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            # Obtener resumen de review clips por mes
            query = """
                SELECT 
                    DATE_FORMAT(fecha, '%Y-%m-01') as mes,
                    COUNT(*) as total_review_clips
                FROM review_clips
                WHERE fecha BETWEEN :start_date AND :end_date
                GROUP BY DATE_FORMAT(fecha, '%Y-%m-01')
            """
            df_review_clips = pd.read_sql(
                text(query), 
                conn, 
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            # Combinar todos los DataFrames
            dfs = [df_entrenamientos, df_meetings, df_review_clips]
            df_final = None
            
            for df in dfs:
                if not df.empty:
                    if df_final is None:
                        df_final = df
                    else:
                        df_final = df_final.merge(df, on='mes', how='outer')
            
            # Si no hay datos, devolver un DataFrame vacío con las columnas esperadas
            if df_final is None or df_final.empty:
                return pd.DataFrame(columns=['mes', 'total_entrenamientos', 'total_meetings', 'total_review_clips'])
                
            # Rellenar valores nulos con 0
            df_final = df_final.fillna(0)
            
            # Ordenar por mes
            df_final = df_final.sort_values('mes')
            
            return df_final
            
    except Exception as e:
        print(f"Error en get_monthly_summary_all_players: {e}")
        return pd.DataFrame()


def get_department_metrics(month: str) -> Dict[str, Any]:
    """
    Obtiene las métricas del departamento para un mes específico.
    
    Args:
        month (str): Mes en formato 'YYYY-MM'
        
    Returns:
        dict: Diccionario con las métricas del departamento
    """
    try:
        engine = connect_to_db()
        if not engine:
            return {}
            
        # Obtener el primer y último día del mes
        start_date = f"{month}-01"
        next_month = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=32)).replace(day=1)
        end_date = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
        
        with engine.connect() as conn:
            # Contar entrenamientos individuales
            query = """
                SELECT COUNT(*) as total 
                FROM entrenamientos_individuales 
                WHERE fecha BETWEEN :start_date AND :end_date
            """
            entrenamientos = conn.execute(text(query), 
                                       {'start_date': start_date, 'end_date': end_date}).scalar()
            
            # Contar reuniones
            query = """
                SELECT COUNT(*) as total 
                FROM meetings 
                WHERE fecha BETWEEN :start_date AND :end_date
            """
            meetings = conn.execute(text(query), 
                                  {'start_date': start_date, 'end_date': end_date}).scalar()
            
            # Contar review clips
            query = """
                SELECT COUNT(*) as total 
                FROM review_clips 
                WHERE fecha BETWEEN :start_date AND :end_date
            """
            review_clips = conn.execute(text(query), 
                                      {'start_date': start_date, 'end_date': end_date}).scalar()
            
            # Contar jugadores activos (que tuvieron al menos una actividad)
            query = """
                SELECT COUNT(DISTINCT jugador_id) as total
                FROM (
                    SELECT jugador_id FROM entrenamientos_individuales WHERE fecha BETWEEN :start_date AND :end_date
                    UNION
                    SELECT jugador_id FROM meetings WHERE fecha BETWEEN :start_date AND :end_date
                    UNION
                    SELECT jugador_id FROM review_clips WHERE fecha BETWEEN :start_date AND :end_date
                ) as actividades
            """
            jugadores_activos = conn.execute(text(query), 
                                          {'start_date': start_date, 'end_date': end_date}).scalar()
            
            # Total de jugadores en el sistema
            query = "SELECT COUNT(*) as total FROM jugadores WHERE activo = 1"
            total_jugadores = conn.execute(text(query)).scalar() or 4  # 4 como valor por defecto
            
            # Calcular porcentaje de participación
            porcentaje_participacion = round((jugadores_activos / total_jugadores * 100) if total_jugadores > 0 else 0, 2)
            
            return {
                'total_entrenamientos': int(entrenamientos or 0),
                'total_meetings': int(meetings or 0),
                'total_review_clips': int(review_clips or 0),
                'jugadores_activos': int(jugadores_activos or 0),
                'total_jugadores_activos': int(total_jugadores or 4),
                'porcentaje_participacion': porcentaje_participacion,
                'mes': month
            }
            
    except Exception as e:
        print(f"Error en get_department_metrics: {e}")
        return {}


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


def process_player_comparison_metrics(stats_df, events_df, player_position):
    """
    Calculates summary metrics (KPIs) for multiple players for comparison purposes.
    Applies the same logic as `process_player_metrics`, adapted for a batch of players.
    """

    if stats_df.empty or events_df.empty:
        st.warning("🚨 No stats or event data to process.")
        return pd.DataFrame()

    # --- Step 1: Prepare IDs ---
    stats_df["playerId"] = stats_df["playerId"].astype(str)
    stats_df["matchId"] = stats_df["matchId"].astype(str)
    events_df["playerId"] = events_df["playerId"].astype(str)
    events_df["matchId"] = events_df["matchId"].astype(str)

    # --- Step 2: Clean numeric columns ---
    metric_cols_stats = stats_df.columns.drop(['playerId', 'playerName', 'matchId', 'teamId', 'teamName'], errors='ignore')
    for col in metric_cols_stats:
        stats_df[col] = stats_df[col].astype(str).str.replace(",", ".", regex=False)
        stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    event_cols_to_clean = ['x', 'y', 'value_PassEndX', 'value_PassEndY', 'endX', 'endY', 'value_Length', 'xG', 'xA', 'ps_xG']
    for col in event_cols_to_clean:
        if col in events_df.columns:
            events_df[col] = events_df[col].astype(str).str.replace(",", ".", regex=False)
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

    # --- Step 3: Event-based metrics ---
    def calculate_event_metrics(df):
        df = df.copy()

        passes_into_penalty_area = df[
            (df['value_PassEndX'] >= 88.5) & (df['value_PassEndY'].between(13.6, 54.4))
        ].groupby(['playerId', 'matchId']).size().rename('passes_into_penalty_area')

        carries_into_final_third = df[
            (df['x'] < 66.7) & (df['endX'] >= 66.7)
        ].groupby(['playerId', 'matchId']).size().rename('carries_into_final_third')

        carries_into_penalty_area = df[
            (df['endX'] >= 88.5) & (df['endY'].between(13.6, 54.4))
        ].groupby(['playerId', 'matchId']).size().rename('carries_into_penalty_area')

        goals = df[df['type_displayName'] == 'Goal'].groupby(['playerId', 'matchId']).size().rename('goals')
        assists = df[df['value_IntentionalGoalAssist'] == 1].groupby(['playerId', 'matchId']).size().rename('assists')
        crosses = df[df['value_Cross'] == 1].groupby(['playerId', 'matchId']).size().rename('crosses')

        long_passes = df[df['value_Length'] >= 30]
        long_passes_total = long_passes.groupby(['playerId', 'matchId']).size().rename('long_passes_total')
        long_passes_success = long_passes[long_passes['outcomeType_displayName'] == 'Successful']\
            .groupby(['playerId', 'matchId']).size().rename('long_passes_success')
        long_pass_pct = (long_passes_success / long_passes_total.replace(0, np.nan) * 100).rename('long_pass_pct')

        progressive_passes = df[
            (df['type_displayName'] == 'Pass') &
            (df['outcomeType_displayName'] == 'Successful') &
            (df['value_Length'] >= 9.11) &
            (df['x'] >= 35) &
            (~df['qualifiers'].str.contains('CornerTaken|Freekick', na=False))
        ].groupby(['playerId', 'matchId']).size().rename('progressive_passes')

        progressive_carry_distance = df[
            (df['type_displayName'] == 'Carry') & ((df['endX'] - df['x']) >= 9.11)
        ].assign(distance=lambda d: d['endX'] - d['x']) \
        .groupby(['playerId', 'matchId'])['distance'].sum().rename('progressive_carry_distance')

        def_actions_outside_box = df[
            (df['x'] > 25) & df['type_displayName'].isin(['Tackle', 'Interception', 'Clearance'])
        ].groupby(['playerId', 'matchId']).size().rename('def_actions_outside_box')

        recoveries = df[df['type_displayName'] == 'BallRecovery']\
            .groupby(['playerId', 'matchId']).size().rename('recoveries')

        shot_creation_actions = df[df['value_ShotAssist'] > 0]\
            .groupby(['playerId', 'matchId']).size().rename('shot_creation_actions')

        xG = df.groupby(['playerId', 'matchId'])['xG'].sum().rename('xG')
        xA = df.groupby(['playerId', 'matchId'])['xA'].sum().rename('xA')
        ps_xG = df.groupby(['playerId', 'matchId'])['ps_xG'].sum().rename('ps_xG')

        goal_creating_actions = goals.add(assists, fill_value=0).rename("goal_creating_actions")

        saves = df[(df["type_displayName"] == "SavedShot") & (df["outcomeType_displayName"] == "Successful")]\
            .groupby(["playerId", "matchId"]).size().rename("saves")

        shots_on_target_faced = df[(df["isShot"] == 1) & df["type_displayName"].isin(["SavedShot", "Goal"])]\
            .groupby(["playerId", "matchId"]).size().rename("shots_on_target_faced")

        goals_conceded = df[(df["isShot"] == 1) & (df["type_displayName"] == "Goal")]\
            .groupby(["playerId", "matchId"]).size().rename("goals_conceded")

        return pd.concat([
            passes_into_penalty_area,
            carries_into_final_third,
            carries_into_penalty_area,
            goals, assists, crosses,
            long_passes_total, long_passes_success, long_pass_pct,
            progressive_passes, progressive_carry_distance,
            def_actions_outside_box, recoveries,
            shot_creation_actions, goal_creating_actions,
            xG, xA, ps_xG,
            saves, shots_on_target_faced, goals_conceded
        ], axis=1).fillna(0).reset_index()

    event_metrics = calculate_event_metrics(events_df)

    # --- Step 4: Merge event and stat metrics ---
    combined_df = pd.merge(stats_df, event_metrics, on=["playerId", "matchId"], how="left").fillna(0)

    # --- Step 5: Derived KPIs ---
    combined_df["pass_completion_pct"] = (combined_df["passesAccurate"] / combined_df["passesTotal"].replace(0, np.nan)) * 100
    combined_df["aerial_duel_pct"] = (combined_df["aerialsWon"] / combined_df["aerialsTotal"].replace(0, np.nan)) * 100
    combined_df["take_on_success_pct"] = (combined_df["dribblesWon"] / combined_df["dribblesAttempted"].replace(0, np.nan)) * 100
    combined_df["shots_on_target_pct"] = (combined_df["shotsOnTarget"] / combined_df["shotsTotal"].replace(0, np.nan)) * 100
    combined_df["tackle_success_pct"] = (combined_df["tackleSuccessful"] / combined_df["tacklesTotal"].replace(0, np.nan)) * 100
    combined_df["throwin_accuracy_pct"] = (combined_df["throwInsAccurate"] / combined_df["throwInsTotal"].replace(0, np.nan)) * 100
    combined_df["save_pct"] = (combined_df["saves"] / combined_df["shots_on_target_faced"].replace(0, np.nan)) * 100

    combined_df["key_passes"] = combined_df["passesKey"]
    combined_df["goal_creating_actions"] = combined_df["goal_creating_actions"].fillna(0)
    combined_df["shot_creating_actions"] = combined_df["shotsTotal"] + combined_df["passesKey"]

    # Optional: goalkeeper extras
    combined_df["claimsHigh"] = stats_df.get("claimsHigh", 0)
    combined_df["collected"] = stats_df.get("collected", 0)
    combined_df["totalSaves"] = stats_df.get("totalSaves", 0)

    # --- Step 6: Format ---
    percent_cols = [
        "pass_completion_pct", "aerial_duel_pct", "take_on_success_pct",
        "shots_on_target_pct", "tackle_success_pct", "throwin_accuracy_pct",
        "long_pass_pct", "save_pct"
    ]
    combined_df[percent_cols] = combined_df[percent_cols].clip(0, 100).round(1)
    combined_df = combined_df.drop_duplicates(subset=["playerId", "matchId"], keep="last").reset_index(drop=True)

    return combined_df
