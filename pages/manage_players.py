import os
import streamlit as st
import pandas as pd
from io import BytesIO
from db_utils import connect_to_db
from PIL import Image

# --- Config --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "img")
LOGO_PATH = os.path.join(IMG_DIR, "watford_logo.png")

st.set_page_config(
    page_title="Gestionar lista de jugadores",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Auth ----------------------------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You must be logged in to view this page.")
    st.stop()

if st.session_state.get("user_type") != "staff":
    st.warning("Only staff users can access this page.")
    st.stop()

# --- Helpers -------------------------------------------------
PLAYERS_FILE_XLSX = os.path.join(BASE_DIR, "data", "watford_players_login_info.xlsx")

@st.cache_data(show_spinner=False)
def load_players_df(path: str) -> pd.DataFrame:
    """Load players Excel into a DataFrame. If the file doesn't exist create an empty df."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["playerName", "activo"])
    return pd.read_excel(path)


def save_players_df(df: pd.DataFrame, path: str):
    """Save dataframe back to Excel safely."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure 'activo' column is int and only contains 0/1
    if "activo" in df.columns:
        df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    with BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        with open(path, "wb") as f:
            f.write(buffer.read())

# --- UI ------------------------------------------------------
st.title("⚙️ Gestionar lista de jugadores")

# Display logo on the sidebar for consistency
with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=120)
    except FileNotFoundError:
        pass

st.markdown("Revise y modifique el estado *Activo/Inactivo* de los jugadores. Use el menú de opciones en la columna **activo** para cambiar 1=Activo, 0=Inactivo. Haga clic en **Guardar cambios** para persistir los datos.")

players_df = load_players_df(PLAYERS_FILE_XLSX).copy()

# --- DB players selector ------------------------------------------------
engine = connect_to_db()
if engine is not None:
    try:
        db_players_df = pd.read_sql("SELECT DISTINCT playerName FROM player_data ORDER BY playerName", engine)
        db_players = db_players_df["playerName"].dropna().unique().tolist()
    except Exception as e:
        st.warning(f"No se pudo cargar jugadores desde la base de datos: {e}")
        db_players = []
else:
    db_players = []

# Build list of players not yet in excel
existing_names = players_df["playerName"].fillna("").str.strip().tolist()
available_to_add = [p for p in db_players if p.strip() not in existing_names]

with st.expander("➕ Añadir jugador desde base de datos", expanded=False):
    if not available_to_add:
        st.info("No hay jugadores nuevos para añadir.")
    else:
        selected_new_player = st.selectbox("Seleccionar jugador a añadir", options=["-- Seleccione --"] + available_to_add, key="new_player_selector")
        if st.button("Añadir jugador", key="add_player_btn"):
            if selected_new_player and selected_new_player != "-- Seleccione --":
                # Append to dataframe with activo=1
                players_df = players_df.append({"playerName": selected_new_player, "activo": 1}, ignore_index=True)
                save_players_df(players_df, PLAYERS_FILE_XLSX)
                st.success(f"Jugador '{selected_new_player}' añadido correctamente.")
                # Refresh available list
                st.experimental_rerun()
            else:
                st.warning("Seleccione un jugador válido.")

if players_df.empty:
    st.info("No se encontraron jugadores. Puede cargar el archivo correspondiente desde la página principal.")

# Configure column 'activo' as selectbox with 0/1 choices
edited_df = st.data_editor(
    players_df,
    num_rows="dynamic",
    column_config={
        "activo": st.column_config.SelectboxColumn(
            "Activo (1=Sí, 0=No)",
            help="1 = jugador activo, 0 = jugador inactivo",
            options=[1, 0],
            required=True,
            width="small"
        )
    },
    key="players_table_editor"
)

if st.button("Guardar cambios", type="primary"):
    try:
        save_players_df(edited_df, PLAYERS_FILE_XLSX)
        st.success("✅ Cambios guardados correctamente!")
        # clear cache to reflect new data if user returns later
        load_players_df.clear()
    except Exception as e:
        st.error(f"❌ Error al guardar los cambios: {e}")
