import streamlit as st
import pandas as pd
import os
from PIL import Image
import base64
from io import BytesIO
from db_utils import connect_to_db, load_player_data  # Your db functions

# --- Page settings ---
import os

# Obtener la ruta absoluta al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')

st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"  # Mostrar sidebar para ver el logo
)

# Master password for Player login (can be overridden via env var)
MASTER_PASSWORD = os.environ.get("PLAYER_MASTER_PASSWORD", "admin123")

# --- Load Players from local CSV/XLSX for login (Players role) ---
@st.cache_data(ttl=600)
def load_players_for_login():
    """
    Loads players from data/watford_players_login_info.[xlsx|csv]
    - Filters only activo == 1
    - Normalizes columns: playerId (string), playerName (string), activo (int)
    Returns a DataFrame with at least these columns.
    """
    try:
        csv_path = os.path.join('data', 'watford_players_login_info.csv')
        xlsx_path = os.path.join('data', 'watford_players_login_info.xlsx')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype={"playerId": "string"})
        elif os.path.exists(xlsx_path):
            # Keep playerId as string to preserve any leading zeros
            df = pd.read_excel(xlsx_path, converters={"playerId": lambda x: str(x).strip() if pd.notna(x) else None})
        else:
            st.error("Players file not found in data/. Please add watford_players_login_info.xlsx or CSV.")
            return pd.DataFrame(columns=["playerId", "playerName", "activo"]) 

        # Normalize columns
        df.columns = [str(c).strip() for c in df.columns]

        # Ensure required columns
        if 'playerName' not in df.columns:
            st.error("The players file requires a 'playerName' column.")
            return pd.DataFrame(columns=["playerId", "playerName", "activo"]) 

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
        else:
            df['playerId'] = None

        # Filter only active players
        df = df[df['activo'] == 1].copy()
        # Clean names
        df['playerName'] = df['playerName'].astype(str).str.strip()
        # Drop rows without name
        df = df[df['playerName'] != ""]

        return df
    except Exception as e:
        st.error(f"Error loading players file: {e}")
        return pd.DataFrame(columns=["playerId", "playerName", "activo"]) 

# --- Load Staff Users ---
@st.cache_data(ttl=600)
def load_staff_users():
    if not os.path.exists('staff_users.csv'):
        # Create empty staff users file if it doesn't exist
        pd.DataFrame(columns=['username', 'password', 'full_name', 'role']).to_csv('staff_users.csv', index=False)
    return pd.read_csv('staff_users.csv')

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
        /* Requested UI tweak 
        .st-emotion-cache-zy6yx3 {
            padding: 0rem !important;
        }*/
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
                background-position: center;
                background-size: 90% auto;
                min-height: 100px;
                margin-bottom: 0.25rem;
            }}
            </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

inject_sidebar_logo()

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
                st.warning("No active players available. Please update data/watford_players_login_info.xlsx.")
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