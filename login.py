import streamlit as st
import pandas as pd
from PIL import Image
from db_utils import connect_to_db

# Page title
st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon="img/watford_logo.png",
    layout="wide"
)

# --- Load only player_data ---
@st.cache_data
def load_player_data():
    engine = connect_to_db()
    player_data = pd.read_sql("SELECT * FROM player_data", engine)
    return player_data

# Load and filter only Watford players
player_data = load_player_data()
watford_players = player_data[player_data["teamName"] == "Watford"]

# --- LOGIN INTERFACE ---
logo = Image.open("img/watford_logo.png")
st.image(logo, width=100)
st.title("Watford Player Development")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    st.subheader("Login")
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        username = st.text_input("Player Name")
        password = st.text_input("Player ID", type="password")

        if st.button("Login"):
            player_row = watford_players[
                (watford_players["playerName"].str.lower() == username.lower()) &
                (watford_players["playerId"].astype(str) == password)
            ]

            if not player_row.empty:
                st.success(f"Welcome, {username}!")
                st.session_state.logged_in = True
                st.session_state.player_id = int(password)
                st.session_state.player_name = username
                st.rerun()
            else:
                st.error("Invalid credentials or not a Watford player.")

if st.session_state.logged_in:
    st.switch_page("pages/Player_Dashboard.py")
