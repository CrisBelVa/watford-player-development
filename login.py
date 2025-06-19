import streamlit as st
import pandas as pd
import os
from PIL import Image
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
    initial_sidebar_state="collapsed"  # Ocultar sidebar por defecto
)

# --- Load Watford Players (only when needed) ---
def load_watford_players():
    try:
        engine = connect_to_db()
        player_data = pd.read_sql("SELECT * FROM player_data", engine)
        watford_players = player_data[player_data["teamName"] == "Watford"]
        return watford_players
    except Exception as e:
        st.error("Error connecting to the database. Please try again later.")
        st.stop()  # Detiene la ejecución de la aplicación

# --- Load Staff Users ---
@st.cache_data(ttl=600)
def load_staff_users():
    if not os.path.exists('staff_users.csv'):
        # Create default staff users file if it doesn't exist
        default_staff = pd.DataFrame({
            'username': ['admin'],
            'password': ['admin123'],
            'full_name': ['Administrator'],
            'role': ['admin']
        })
        default_staff.to_csv('staff_users.csv', index=False)
    return pd.read_csv('staff_users.csv')

def validate_staff_login(username, password):
    staff_df = load_staff_users()
    user = staff_df[(staff_df['username'] == username) & (staff_df['password'] == password)]
    if not user.empty:
        return user.iloc[0].to_dict()
    return None

# --- Main code ---

# Ocultar sidebar si no está logueado
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

try:
    logo = Image.open(LOGO_PATH)
    st.image(logo, width=100)
except FileNotFoundError:
    st.error("Logo image not found. Please check the image path.")
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
        role = st.selectbox("Select Role", ["Staff", "Player"], index=0)
        
        # Campos de autenticación
        username = st.text_input("Username" if role == "Staff" else "Player Name", 
                              value="admin" if role == "Staff" else "")
        password = st.text_input("Password" if role == "Staff" else "Player ID", 
                              value="admin123" if role == "Staff" else "", 
                              type="password")

        if st.button("Login"):
            if role == "Player":
                # Cargar jugadores solo cuando sea necesario
                try:
                    watford_players = load_watford_players()
                    player_row = watford_players[
                        (watford_players["playerName"].str.lower() == username.lower()) &
                        (watford_players["playerId"].astype(str) == password)
                    ]
                except Exception as e:
                    st.error("Error connecting to the database. Please try again later.")
                    st.stop()

                if not player_row.empty:
                    st.success(f"Welcome, {username}!")
                    st.session_state.logged_in = True
                    st.session_state.user_type = "player"
                    st.session_state.player_id = int(password)
                    st.session_state.player_name = username
                    st.rerun()
                else:
                    st.error("Invalid credentials or not a Watford player.")
            
            else:  # Staff
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