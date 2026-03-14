import streamlit as st
import pandas as pd
import os
from PIL import Image
import base64
from io import BytesIO
from db_utils import connect_to_db, load_player_data  # Your db functions
from player_ids import normalize_whoscored_player_id
from utils.sheets_client import GoogleSheetsClient

# --- Page settings ---
import os

# Obtener la ruta absoluta al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"  # Mostrar sidebar para ver el logo
)

# Master password for Player login (can be overridden via env var)
MASTER_PASSWORD = os.environ.get("PLAYER_MASTER_PASSWORD", "admin123")

@st.cache_resource(show_spinner=False)
def get_sheets_client() -> GoogleSheetsClient:
    return GoogleSheetsClient()

# --- Load Players from Google Sheets (fallback local CSV/XLSX) ---
@st.cache_data(ttl=600)
def load_players_for_login():
    """
    Loads players from Google Sheets tab 'Players' (fallback local files)
    - Filters only activo == 1
    - Normalizes columns: internal_id (string), playerId (string), playerName (string), activo (int)
    - Keeps only players that have WhoScored `playerId` for player login
    Returns a DataFrame with at least these columns.
    """
    try:
        df = None
        sheets_client = get_sheets_client()
        if sheets_client.is_configured():
            try:
                df = sheets_client.read_players_df()
            except Exception as exc:
                st.warning(f"Could not read Players from Google Sheets. Falling back to local file. ({exc})")

        if df is None:
            candidate_paths = [
                os.path.join(DATA_DIR, "watford_players_login_info.csv"),
                os.path.join(DATA_DIR, "watford_players_login_info.xlsx"),
                os.path.join(BASE_DIR, "watford_players_login_info.csv"),
                os.path.join(BASE_DIR, "watford_players_login_info.xlsx"),
                os.path.join("data", "watford_players_login_info.csv"),
                os.path.join("data", "watford_players_login_info.xlsx"),
                "watford_players_login_info.csv",
                "watford_players_login_info.xlsx",
            ]

            existing = [p for p in candidate_paths if os.path.exists(p)]
            if not existing:
                st.error(
                    "Players source not found. Configure Google Sheets (tab 'Players') "
                    "or add watford_players_login_info.xlsx/CSV locally."
                )
                return pd.DataFrame(columns=["internal_id", "playerId", "playerName", "activo"])

            file_path = existing[0]
            if file_path.lower().endswith(".csv"):
                df = pd.read_csv(file_path, dtype={"internal_id": "string", "playerId": "string"})
            else:
                # Keep playerId as string to preserve any leading zeros
                df = pd.read_excel(
                    file_path,
                    converters={
                        "internal_id": lambda x: str(x).strip() if pd.notna(x) else None,
                        "playerId": lambda x: str(x).strip() if pd.notna(x) else None,
                    },
                )

        # Normalize columns
        df.columns = [str(c).strip() for c in df.columns]

        # Ensure required columns
        if 'playerName' not in df.columns:
            st.error("The players file requires a 'playerName' column.")
            return pd.DataFrame(columns=["internal_id", "playerId", "playerName", "activo"]) 

        if 'internal_id' not in df.columns:
            df['internal_id'] = None
        else:
            df['internal_id'] = df['internal_id'].astype('string')
            df['internal_id'] = df['internal_id'].where(df['internal_id'].notna(), None)
            df['internal_id'] = df['internal_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)

        # activo: default to 1 if missing
        if 'activo' in df.columns:
            df['activo'] = pd.to_numeric(df['activo'], errors='coerce').fillna(1).astype(int)
        else:
            df['activo'] = 1

        # playerId: keep as string or None
        if 'playerId' in df.columns:
            df['playerId'] = df['playerId'].astype('string')
            df['playerId'] = df['playerId'].where(df['playerId'].notna(), None)
            df['playerId'] = df['playerId'].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df["playerId"] = df["playerId"].apply(normalize_whoscored_player_id)
            df["playerId"] = df["playerId"].astype("string").where(df["playerId"].notna(), None)
        else:
            df['playerId'] = None

        # Filter only active players
        df = df[df['activo'] == 1].copy()
        # Clean names
        df['playerName'] = df['playerName'].astype(str).str.strip()
        # Drop rows without name
        df = df[df['playerName'] != ""]
        # For player login, only players with a VALID WhoScored playerId are valid.
        pid = df["playerId"].astype("string").where(df["playerId"].notna(), None)
        df = df[pid.notna() & pid.str.fullmatch(r"\d+", na=False)].copy()

        return df
    except Exception as e:
        st.error(f"Error loading players source: {e}")
        return pd.DataFrame(columns=["internal_id", "playerId", "playerName", "activo"]) 

# --- Load Staff Users ---
@st.cache_data(ttl=600)
def load_staff_users():
    staff_users_path = os.path.join(BASE_DIR, "staff_users.csv")
    if not os.path.exists(staff_users_path):
        # Create empty staff users file if it doesn't exist
        pd.DataFrame(columns=['username', 'password', 'full_name', 'role']).to_csv(staff_users_path, index=False)
    return pd.read_csv(staff_users_path)

def validate_staff_login(username, password):
    staff_df = load_staff_users()
    user = staff_df[(staff_df['username'] == username) & (staff_df['password'] == password)]
    if not user.empty:
        return user.iloc[0].to_dict()
    return None

# --- Main code ---

# Ocultar solo el menú de navegación multipágina en el sidebar (no el sidebar completo)
st.markdown("""
    <style>
        section[data-testid="stSidebarNav"],
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Inyectar logo en el header del sidebar
def inject_sidebar_logo():
    try:
        img = Image.open(LOGO_PATH)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
            <style>
            div[data-testid="stSidebarHeader"] {{
                background-image: url('data:image/png;base64,{b64}');
                background-repeat: no-repeat;
                background-position: center center;
                background-size: contain;
                min-height: 130px;
                padding: 10px 8px 8px 8px;
                margin-bottom: 0.4rem;
            }}
            </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

inject_sidebar_logo()

# Login page visual polish (UI only; no functional changes)
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            border-right: 1px solid #d8d8d8;
            background: #f0f2f6;
        }

        [data-testid="stAppViewContainer"] {
            background: #f0f2f6;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
        }

        h1 {
            letter-spacing: 0.2px;
            font-weight: 800;
            color: #1f2430;
            margin-bottom: 0.3rem;
        }

        h3 {
            color: #262b38;
            margin-top: 0.6rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #d0d4da !important;
            border-radius: 14px !important;
            background: #ffffff;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.07);
        }

        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stTextInput"] > div > div {
            border-radius: 10px !important;
            background-color: #ffffff !important;
            border: 1px solid #cdd3db !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Watford Player Development")

# Inicializar variables de sesión si no existen
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.player_id = None
    st.session_state.player_name = None
    st.session_state.staff_info = None

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    st.subheader("Login")
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        # Selector de rol
        role = st.selectbox("Select Role", ["Player", "Staff"])
        
        # UI condicional según el rol
        if role == "Player":
            players_df = load_players_for_login()
            if players_df.empty:
                st.warning("No active players with WhoScored ID available. Please update tab 'Players' in Google Sheets.")
            else:
                name_to_id = {row['playerName']: row['playerId'] for _, row in players_df.iterrows()}
                player_options = sorted(list(name_to_id.keys()), key=lambda s: s.lower())
                selected_player = st.selectbox("Select Player", options=player_options)

                password = st.text_input("Password", type="password", help="Master password required for player login")

                if st.button("Login"):
                    if password != MASTER_PASSWORD:
                        st.error("Invalid master password.")
                        st.stop()
                    pid = name_to_id.get(selected_player)
                    if pid is None or str(pid).strip() == "":
                        st.error("Selected player has no playerId. Please update the players file.")
                    else:
                        st.success(f"Welcome, {selected_player}!")
                        st.session_state.logged_in = True
                        st.session_state.user_type = "player"
                        # Keep as string; downstream code will normalize
                        st.session_state.player_id = str(pid).strip()
                        st.session_state.player_name = selected_player
                        st.rerun()
        else:
            # Campos de autenticación para Staff
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                staff_user = validate_staff_login(username, password)
                if staff_user:
                    st.success(f"Welcome, {staff_user['full_name']}!")
                    st.session_state.logged_in = True
                    st.session_state.user_type = "staff"
                    st.session_state.staff_info = staff_user
                    st.rerun()
                else:
                    st.error("Invalid staff credentials.")

# Redirect based on user type after login
if st.session_state.logged_in:
    if st.session_state.user_type == "player":
        try:
            player_id = st.session_state.player_id
            player_name = st.session_state.player_name
            # (Optional) preload everything here if you want to cache it
            # event_data, match_data, player_data, team_data, player_stats, total_minutes, games_as_starter = load_player_data(player_id, player_name)
            st.switch_page("pages/player_dashboard.py")
        except Exception as e:
            st.error("Error connecting to the database. Please try again later.")
            # Limpiar el estado de la sesión para permitir un nuevo intento
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:  # staff
        st.switch_page("pages/staff_dashboard.py")
