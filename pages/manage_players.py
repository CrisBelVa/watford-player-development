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
    """Load players Excel into a DataFrame. If the file doesn't exist create an empty df.
    Ensures the columns: playerName (str), playerId (string), activo (int).
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["playerName", "playerId", "activo"])
    # Keep playerId as string to preserve format
    df = pd.read_excel(path, converters={"playerId": lambda x: str(x).strip() if pd.notna(x) else None})
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    if "playerName" not in df.columns:
        df["playerName"] = ""
    if "playerId" not in df.columns:
        df["playerId"] = None
    if "activo" not in df.columns:
        df["activo"] = 1
    # Coerce types
    df["playerName"] = df["playerName"].astype(str).str.strip()
    df["playerId"] = df["playerId"].astype("string").where(df["playerId"].notna(), None)
    df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    return df[["playerName", "playerId", "activo"]]


def save_players_df(df: pd.DataFrame, path: str):
    """Save dataframe back to Excel safely."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure 'activo' column is int and only contains 0/1
    if "activo" in df.columns:
        df["activo"] = pd.to_numeric(df["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    # Normalize order and whitespace
    df["playerName"] = df["playerName"].astype(str).str.strip()
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].astype("string")
        df["playerId"] = df["playerId"].where(df["playerId"].notna(), None)
        df["playerId"] = df["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[["playerName", "playerId", "activo"]]
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
# Sort alphabetically by playerName (case-insensitive)
if not players_df.empty and "playerName" in players_df.columns:
    players_df = (
        players_df
        .assign(_name_lower=players_df["playerName"].astype(str).str.lower())
        .sort_values(by="_name_lower", kind="stable")
        .drop(columns=["_name_lower"]) 
        .reset_index(drop=True)
    )

# --- DB players selector ------------------------------------------------
engine = connect_to_db()
if engine is not None:
    try:
        db_players_df = pd.read_sql(
            "SELECT DISTINCT playerId, playerName FROM player_data WHERE playerName IS NOT NULL ORDER BY playerName",
            engine
        )
        # Normalize
        db_players_df["playerName"] = db_players_df["playerName"].astype(str).str.strip()
        db_players_df["playerId"] = db_players_df["playerId"].apply(lambda x: str(x).strip() if pd.notna(x) else None)
        db_players = db_players_df.dropna(subset=["playerName"]).to_dict(orient="records")
    except Exception as e:
        st.warning(f"No se pudo cargar jugadores desde la base de datos: {e}")
        db_players = []
else:
    db_players = []

# Build list of players not yet in excel (by playerName)
existing_names = set(players_df["playerName"].fillna("").str.strip().tolist())
available_to_add = [p for p in db_players if p["playerName"].strip() not in existing_names]

with st.expander("➕ Añadir jugador desde base de datos", expanded=False):
    if not available_to_add:
        st.info("No hay jugadores nuevos para añadir.")
    else:
        # Build pretty labels "Name (ID)" but keep mapping to dict
        labels = [f"{p['playerName']} ({p['playerId'] if p['playerId'] is not None else 'sin ID'})" for p in available_to_add]
        idx = st.selectbox("Seleccionar jugador a añadir", options=["-- Seleccione --"] + labels, key="new_player_selector")
        if st.button("Añadir jugador", key="add_player_btn"):
            if idx and idx != "-- Seleccione --":
                # Find the selected record
                sel_index = labels.index(idx)
                chosen = available_to_add[sel_index]
                new_row = pd.DataFrame([{ "playerName": chosen["playerName"], "playerId": chosen["playerId"], "activo": 1 }])
                players_df = pd.concat([players_df, new_row], ignore_index=True)
                save_players_df(players_df, PLAYERS_FILE_XLSX)
                st.success(f"Jugador '{chosen['playerName']}' añadido correctamente.")
                st.rerun()
            else:
                st.warning("Seleccione un jugador válido.")

if players_df.empty:
    st.info("No se encontraron jugadores. Puede cargar el archivo correspondiente desde la página principal.")

# ---- Filtro de Activo/Inactivo ----
status_filter = st.selectbox(
    "Filtrar por estado",
    options=["Todos", "Activos", "Inactivos"],
    index=0,
    help="Filtra la tabla por el estado de activo"
)

if status_filter == "Activos":
    filtered_df = players_df[players_df["activo"] == 1].copy()
elif status_filter == "Inactivos":
    filtered_df = players_df[players_df["activo"] == 0].copy()
else:
    filtered_df = players_df.copy()

# Configure column 'activo' as selectbox with 0/1 choices
edited_df = st.data_editor(
    filtered_df,
    num_rows="dynamic",
    column_config={
        "playerName": st.column_config.TextColumn(
            "Nombre del jugador",
            required=True,
            help="Nombre completo tal como aparece en la base de datos"
        ),
        "playerId": st.column_config.TextColumn(
            "playerId",
            required=False,
            help="Identificador del jugador (requerido para jugadores activos)"
        ),
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
        # Merge edits back into the full dataframe by playerName
        df_to_save = players_df.copy()
        edits = edited_df.copy()
        # Normalize keys for merge
        df_to_save["playerName_key"] = df_to_save["playerName"].astype(str).str.strip().str.lower()
        edits["playerName_key"] = edits["playerName"].astype(str).str.strip().str.lower()

        # Update rows existing in df_to_save with edits
        update_map = edits.set_index("playerName_key")[['playerName','playerId','activo']].to_dict(orient='index')
        def apply_update(row):
            key = row["playerName_key"]
            if key in update_map:
                upd = update_map[key]
                row["playerName"] = upd.get("playerName", row["playerName"])
                row["playerId"] = upd.get("playerId", row.get("playerId"))
                row["activo"] = upd.get("activo", row.get("activo", 1))
            return row
        df_to_save = df_to_save.apply(apply_update, axis=1)
        df_to_save = df_to_save.drop(columns=["playerName_key"], errors='ignore')

        # Normalize whitespace and types
        df_to_save["playerName"] = df_to_save["playerName"].astype(str).str.strip()
        df_to_save["playerId"] = df_to_save["playerId"].astype("string").where(df_to_save["playerId"].notna(), None)
        df_to_save["playerId"] = df_to_save["playerId"].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df_to_save["activo"] = pd.to_numeric(df_to_save["activo"], errors="coerce").fillna(1).astype(int).clip(0, 1)

        # Validation rules
        errors = []
        # 1) Names required
        if (df_to_save["playerName"].str.len() == 0).any():
            errors.append("Hay jugadores con nombre vacío.")
        # 2) Active players must have non-empty playerId
        active_without_id = df_to_save[(df_to_save["activo"] == 1) & ((df_to_save["playerId"].isna()) | (df_to_save["playerId"].astype(str).str.strip() == ""))]
        if not active_without_id.empty:
            errors.append("Jugadores activos sin playerId. Asigne un playerId antes de guardar.")
        # 3) playerId uniqueness among non-null IDs
        ids = df_to_save["playerId"].dropna().astype(str).str.strip()
        if ids.duplicated().any():
            dupes = ids[ids.duplicated()].unique().tolist()
            errors.append(f"playerId duplicados: {', '.join(dupes)}")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            st.stop()

        save_players_df(df_to_save, PLAYERS_FILE_XLSX)
        st.success("✅ Cambios guardados correctamente!")
        # clear cache to reflect new data if user returns later
        load_players_df.clear()
    except Exception as e:
        st.error(f"❌ Error al guardar los cambios: {e}")
