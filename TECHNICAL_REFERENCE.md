# Watford Player Development Hub - Technical Reference

This document provides a deep dive into the technical architecture, data processing logic, and metric calculations of the Watford Player Development application.

---

## 1. Data Architecture

### Database & Environment
- **Engine**: MySQL 8.0+
- **Connection**: SQLAlchemy with `pymysql` driver.
- **Configuration**: Managed via environment variables in a `.env` file (requires `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_NAME`).
- **Schema Joins**: Most performance views are joined dynamically using the `matchId` (Int64) as the primary key across `player_stats`, `event_data`, and `match_data`.

### Core Tables & Views
| Table/View | Description |
| :--- | :--- |
| `player_data` | Static player info (age, height, weight) and match appearance records. |
| `player_stats` | Aggregate performance data per match (e.g., total passes, tackles). |
| `event_data` | Granular event-level data (coordinates, qualifiers) for every action in a match. |
| `match_data` | Match metadata including dates, scores, competitions, and seasons. |
| `team_data` | Team-level information and match results. |
| `jugadores` | Master list of players for the Individual Development module. |
| `entrenamientos_individuales` | Records of individual training sessions. |
| `meetings` | Records of staff-player meetings. |
| `review_clips` | Records of video analysis sessions. |

---

## 2. Position Logic & Mapping

The application uses a sophisticated mapping system to ensure players are evaluated against relevant KPIs.

### Position Normalization
Raw position codes from the database are mapped to human-readable labels in `db_utils.py`:
- **GK** → Goalkeeper
- **DC, CB** → Center Back
- **DL, LB, DML, LWB** → Left Back
- **DR, RB, DMR, RWB** → Right Back
- **DMC, DM** → Defensive Midfielder
- **MC, CM, ML, LM, MR, RM** → Midfielder
- **AMC, AM** → Attacking Midfielder
- **AML, FWL, LW** → Left Winger
- **AMR, FWR, RW** → Right Winger
- **FW, ST, CF** → Striker

### Position Inference
If a player is listed as a "Sub" or has no position for a specific match, the system infers their primary position by looking at their most frequent non-substitute position in both `event_data` and `player_data` history.

---

## 3. Metric & Delta Calculations

The application uses specific mathematical approaches to ensure data integrity when comparing different timeframes.

### Delta Calculation Logic
Deltas are calculated in the `calculate_delta` function within `player_dashboard.py`. The "Baseline" is always the **Full Season Average** for the selected player.

#### 1. Weighted Averages (Percentages)
For percentage-based metrics (e.g., Pass Completion %), the system does **not** average the percentages of each match. Instead, it uses a weighted approach:
- `Delta = (Sum of Numerators / Sum of Denominators) * 100`
- This ensures that a match with 50 passes carries more weight than a match with only 2 passes.

#### 2. Per-Match Averages (Counts)
For count-based metrics (e.g., Interceptions, Goals):
- `Value = Total Count / Number of Matches in Selection`
- `Delta = Value (Filtered) - Value (Season Average)`

### Heuristic Advanced Metrics (Event-Based)
The system calculates advanced metrics from raw `event_data` coordinates (0-100 scale):

#### 1. Expected Goals (xG)
Calculated based on shot location and type:
- **Base xG**: Assigned by zone (e.g., 0.36 for 6-yard box, 0.12 for rest of penalty area).
- **Modifiers**: 
    - Penalties: Fixed at 0.76.
    - Big Chances: Minimum 0.35.
    - Headers: Multiplied by 0.75.
    - Angle: Multiplied by 0.70 for tight angles, 1.15 for central angles.

#### 2. Post-Shot xG (ps_xG)
Evaluates shot placement for on-target shots:
- **High Corners**: 1.25x multiplier.
- **Low Corners**: 1.15x multiplier.
- **Central/Low**: 0.85x multiplier.

#### 3. Progressive Actions
- **Progressive Passes**: Successful passes that move the ball at least 9.11 meters (10 yards) closer to the opponent's goal, starting from outside the defensive 35% of the pitch.
- **Progressive Carries**: Ball carries that move at least 9.11 meters closer to the opponent's goal.

---

## 4. Playing Time Logic

Minutes played are calculated dynamically in `prepare_player_data_with_minutes`:
- **Starters**: `90` (or `120` if ET) minus `subbedOutExpandedMinute`.
- **Substitutes**: `subbedOutExpandedMinute` (or match end) minus `subbedInExpandedMinute`.
- **Unused Subs**: Fixed at `0`.

---

## 5. Staff-Specific Logic

### Player Login Management
Staff manage player access via `data/watford_players_login_info.xlsx`. The `manage_players.py` page allows staff to:
- Toggle the `activo` flag (1 for active, 0 for inactive).
- Sync new players from the `player_data` database table into the login list.

### Data Import Validation
When staff upload an Excel file in the "Files" section, the system performs:
- **Header Check**: Ensures columns `Mes`, `Player`, and at least one activity column exist.
- **Date Validation**: Ensures dates are between 2020 and the current year + 1.
- **Data Integrity**: Verifies that names are not empty and at least one activity is recorded per row.

---
*Technical Reference - Watford FC Player Development*
