import streamlit as st
import pandas as pd
from PIL import Image
from db_utils import connect_to_db, load_player_data  # Your db functions

# --- Page settings ---
st.set_page_config(
    page_title="Watford Player Development Hub",
    page_icon="img/watford_logo.png",
    layout="wide"
)

# --- Load Watford Players (only player_data table first) ---
@st.cache_data(ttl=600)
def load_watford_players():
    engine = connect_to_db()
    player_data = pd.read_sql("SELECT * FROM player_data", engine)
    watford_players = player_data[player_data["teamName"] == "Watford"]
    return watford_players

# --- Main code ---

logo = Image.open("img/watford_logo.png")
st.image(logo, width=100)
st.title("Watford Player Development")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

watford_players = load_watford_players()

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

                # 🔥 Here you could preload heavier data if you want (optional)
                st.rerun()
            else:
                st.error("Invalid credentials or not a Watford player.")

# Only load the heavy player data after login
if st.session_state.logged_in:
    player_id = st.session_state.player_id
    player_name = st.session_state.player_name

    # (Optional) preload everything here if you want to cache it
    # event_data, match_data, player_data, team_data, player_stats, total_minutes, games_as_starter = load_player_data(player_id, player_name)

    st.switch_page("pages/player_dashboard.py")