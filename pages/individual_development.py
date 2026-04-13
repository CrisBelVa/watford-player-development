import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime, timedelta
import os
from PIL import Image
from pathlib import Path
import io
import streamlit.components.v1 as components
from utils.pdf_generator import generate_individual_development_report_landscape
from utils.sheets_client import GoogleSheetsClient
from utils.pdf_cover_photos import list_cover_photos, save_cover_photo


# --- Configuración de página ---
# Obtener la ruta absoluta al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')
BACKGROUND_COVER_PATH = os.path.join(IMG_DIR, 'Watford_portada_d.jpg')
LOCAL_TRAINING_FILE = Path(BASE_DIR) / "data" / "Individuals - Training.xlsx"

# Configuración de la página
st.set_page_config(
    page_title="Watford - Individual Development",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Verificación de autenticación ---
if "logged_in" not in st.session_state or not st.session_state.logged_in or st.session_state.user_type != "staff":
    st.warning("You must be logged in as staff to view this page.")
    st.stop()

# Cargar logo
@st.cache_data
def load_logo():
    try:
        return Image.open(LOGO_PATH)
    except FileNotFoundError:
        st.error("Logo image not found. Please check the image path.")
        return None

logo = load_logo()

def get_sheets_client() -> GoogleSheetsClient:
    cache_key = "_individual_development_sheets_client"
    cached_client = st.session_state.get(cache_key)
    if isinstance(cached_client, GoogleSheetsClient):
        return cached_client

    client = GoogleSheetsClient()
    st.session_state[cache_key] = client
    return client


if hasattr(st, "dialog"):
    @st.dialog("PDF Cover Photo")
    def _pdf_cover_photo_dialog(player_key: str, player_label: str, session_key: str, key_prefix: str):
        st.caption("Upload a player photo and keep it in history for future PDF reports.")
        uploaded_photo = st.file_uploader(
            "Upload player photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"{key_prefix}_upload",
        )
        if st.button("Save uploaded photo", key=f"{key_prefix}_save_upload"):
            if uploaded_photo is None:
                st.warning("Please upload an image first.")
            else:
                try:
                    saved_path = save_cover_photo(
                        uploaded_file=uploaded_photo,
                        base_dir=BASE_DIR,
                        player_key=player_key,
                    )
                    st.session_state[session_key] = saved_path
                    st.success("Photo saved and set as active cover.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save photo: {exc}")

        history = list_cover_photos(base_dir=BASE_DIR, player_key=player_key)
        if not history:
            st.info(f"No saved cover photos yet for {player_label}.")
            if st.button("Use no photo", key=f"{key_prefix}_none_only"):
                st.session_state[session_key] = None
                st.success("PDF cover will use no player photo.")
            return

        options = [None] + [item["path"] for item in history]
        labels = {None: "No photo"}
        for item in history:
            labels[item["path"]] = item["label"]

        current_value = st.session_state.get(session_key)
        if current_value not in options:
            current_value = history[0]["path"]
            st.session_state[session_key] = current_value

        selected_value = st.selectbox(
            "Photo history",
            options=options,
            index=options.index(current_value) if current_value in options else 0,
            format_func=lambda value: labels.get(value, "No photo"),
            key=f"{key_prefix}_history_select",
        )
        if selected_value and os.path.exists(selected_value):
            st.image(selected_value, width=220, caption="Cover preview")

        col_apply, col_clear = st.columns(2)
        with col_apply:
            if st.button("Use selected photo", key=f"{key_prefix}_apply"):
                st.session_state[session_key] = selected_value
                st.success("Active cover photo updated.")
        with col_clear:
            if st.button("Use no photo", key=f"{key_prefix}_clear"):
                st.session_state[session_key] = None
                st.success("PDF cover will use no player photo.")
else:
    def _pdf_cover_photo_dialog(*args, **kwargs):
        st.warning("This Streamlit version does not support popup dialogs.")

def _read_local_training_file() -> pd.DataFrame:
    if not LOCAL_TRAINING_FILE.exists():
        return pd.DataFrame()
    expected_tokens = ("mes", "month", "date", "fecha", "player", "jugador")
    best_header = 0
    best_score = -1

    # Detect the most likely header row (0..4) to avoid shifting data rows into headers.
    for header_idx in range(5):
        try:
            candidate_df = pd.read_excel(LOCAL_TRAINING_FILE, header=header_idx, nrows=1)
        except Exception:
            continue

        normalized_cols = [str(c).strip().lower() for c in candidate_df.columns]
        score = sum(any(token == col for col in normalized_cols) for token in expected_tokens)
        if score > best_score:
            best_score = score
            best_header = header_idx

    return pd.read_excel(LOCAL_TRAINING_FILE, header=best_header)

def _load_sessions_raw_df() -> pd.DataFrame:
    sheets_client = get_sheets_client()
    if sheets_client.is_configured():
        try:
            return sheets_client.read_sessions_df()
        except Exception as exc:
            st.warning(f"Could not read Google Sheets tab 'Sesions'. Using local fallback. ({exc})")
    return _read_local_training_file()

def _normalize_sessions_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["Date", "Player", "Training_Type", "Meeting", "Review_Clips"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=base_columns)

    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    normalized_map = {
        col: (
            str(col)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("  ", " ")
        )
        for col in df.columns
    }

    def pick_col(candidates):
        for cand in candidates:
            cand_l = cand.lower()
            for original, normalized in normalized_map.items():
                if normalized == cand_l:
                    return original
        return None

    date_col = pick_col(["date", "mes", "fecha"])
    player_col = pick_col(["player", "jugador"])
    training_col = pick_col(["individual training", "training type", "training"])
    meeting_col = pick_col(["meeting", "reunion", "meeting type"])
    review_col = pick_col(["review clips", "review clips ", "review", "review clip"])

    # Fallback by position for legacy files.
    if date_col is None and len(df.columns) > 0:
        date_col = df.columns[0]
    if player_col is None and len(df.columns) > 1:
        player_col = df.columns[1]
    if training_col is None and len(df.columns) > 2:
        training_col = df.columns[2]
    if meeting_col is None and len(df.columns) > 3:
        meeting_col = df.columns[3]
    if review_col is None and len(df.columns) > 4:
        review_col = df.columns[4]

    out = pd.DataFrame(columns=base_columns)
    out["Date"] = df[date_col] if date_col in df.columns else None
    out["Player"] = df[player_col] if player_col in df.columns else None
    out["Training_Type"] = df[training_col] if training_col in df.columns else None
    out["Meeting"] = df[meeting_col] if meeting_col in df.columns else None
    out["Review_Clips"] = df[review_col] if review_col in df.columns else None
    return out

def _save_sessions_raw_df(df_to_save: pd.DataFrame) -> str:
    sheets_client = get_sheets_client()
    if sheets_client.is_configured():
        try:
            sheets_client.write_sessions_df(df_to_save)
            return "Google Sheets / Sesions"
        except Exception as exc:
            st.warning(f"Could not save to Google Sheets (Sesions). Saving locally. ({exc})")

    LOCAL_TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(LOCAL_TRAINING_FILE, engine="openpyxl") as writer:
        df_to_save.to_excel(writer, index=False)
    return str(LOCAL_TRAINING_FILE)

def _pick_existing_column(columns, candidates):
    normalized = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in columns
    }
    for cand in candidates:
        key = cand.strip().lower().replace("_", " ").replace("-", " ")
        if key in normalized:
            return normalized[key]
    return None

def _append_training_session_row(
    session_date,
    player_name: str,
    training_text: str = "",
    meeting_text: str = "",
    review_text: str = "",
) -> str:
    raw_df = _load_sessions_raw_df().copy()

    if raw_df.empty:
        raw_df = pd.DataFrame(columns=["Mes", "Player", "Individual Training", "Meeting", "Review Clips"])

    date_col = _pick_existing_column(raw_df.columns, ["mes", "month", "date", "fecha"]) or "Mes"
    player_col = _pick_existing_column(raw_df.columns, ["player", "jugador"]) or "Player"
    training_col = _pick_existing_column(raw_df.columns, ["individual training", "training type", "training"]) or "Individual Training"
    meeting_col = _pick_existing_column(raw_df.columns, ["meeting", "reunion", "meeting type"]) or "Meeting"
    review_col = _pick_existing_column(raw_df.columns, ["review clips", "review clip", "review"]) or "Review Clips"

    for required_col in [date_col, player_col, training_col, meeting_col, review_col]:
        if required_col not in raw_df.columns:
            raw_df[required_col] = ""

    new_row = {col: "" for col in raw_df.columns}
    new_row[date_col] = pd.to_datetime(session_date).strftime("%Y-%m-%d")
    new_row[player_col] = str(player_name).strip()
    new_row[training_col] = str(training_text).strip()
    new_row[meeting_col] = str(meeting_text).strip()
    new_row[review_col] = str(review_text).strip()

    updated_df = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)
    return _save_sessions_raw_df(updated_df)

# --- Estilos CSS personalizados ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #212529;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    .section-title {
        color: #212529;
        border-bottom: 2px solid #fcec03;
        padding-bottom: 5px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Funciones para cargar datos ---
@st.cache_data(ttl=3600)
def load_training_data():
    """Carga y preprocesa los datos de entrenamiento desde Sheets/archivo local."""
    try:
        raw_df = _load_sessions_raw_df()
        df = _normalize_sessions_df(raw_df)

        if df.empty:
            return df
        
        # Eliminar filas sin fecha o con fecha inválida
        df = df[df['Date'].notna()]
        
        # Convertir fechas a datetime, manejando diferentes formatos
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Eliminar filas donde la fecha no pudo ser convertida
            df = df[df['Date'].notna()]
        except Exception as e:
            st.error(f"Error converting dates: {e}")
            return pd.DataFrame()
        
        # Extraer año, mes y año-mes para agrupaciones
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
        
        # Limpiar cadenas en las columnas de texto
        text_columns = ['Player', 'Training_Type', 'Meeting', 'Review_Clips']
        for col in text_columns:
            if col in df.columns:
                cleaned = df[col].astype("string").str.strip()
                lowered = cleaned.str.lower()
                # Normalize spreadsheet blanks/placeholders to missing values.
                df[col] = cleaned.mask(lowered.isin(["", "nan", "none", "null", "nat"]), pd.NA)
        
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame()

# --- Funciones de ayuda para el Dashboard ---
@st.cache_data(ttl=3600)  # Cachear por 1 hora
def get_current_month_metrics(start_date=None, end_date=None):
    """Obtener métricas para el rango de fechas especificado"""
    try:
        # Cargar datos de entrenamiento
        df = load_training_data()
        if df.empty:
            return {
                'entrenamientos': 0,
                'meetings': 0,
                'review_clips': 0,
                'jugadores_activos': 0,
                'total_jugadores': 0,
                'porcentaje_participacion': 0
            }
        
        # Filtrar datos por el rango de fechas
        if start_date and end_date:
            # Convertir a datetime si son strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()
                
            # Ajustar la fecha de fin para incluir todo el día
            end_date = datetime.combine(end_date, datetime.max.time())
            
            # Filtrar por rango de fechas
            date_mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date.date())
            filtered_data = df[date_mask].copy()
        else:
            # Si no se especifican fechas, usar el mes actual
            now = datetime.now()
            current_year_month = now.strftime('%Y-%m')
            filtered_data = df[df['Year_Month'] == current_year_month].copy()
        
        # Contar activities
        entrenamientos = filtered_data['Training_Type'].count()
        meetings = filtered_data['Meeting'].count()
        review_clips = filtered_data['Review_Clips'].count()
        
        # Contar jugadores únicos con activities
        jugadores_activos = filtered_data['Player'].nunique()
        total_jugadores = df['Player'].dropna().nunique()
        
        # Calcular porcentaje de participación
        porcentaje_participacion = (jugadores_activos / total_jugadores) * 100 if total_jugadores > 0 else 0
        
        return {
            'entrenamientos': int(entrenamientos),
            'meetings': int(meetings),
            'review_clips': int(review_clips),
            'jugadores_activos': int(jugadores_activos),
            'total_jugadores': int(total_jugadores),
            'porcentaje_participacion': round(porcentaje_participacion, 2)
        }
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return {
            'entrenamientos': 0,
            'meetings': 0,
            'review_clips': 0,
            'jugadores_activos': 0,
            'total_jugadores': 0,
            'porcentaje_participacion': 0
        }

@st.cache_data(ttl=3600)  # Cachear por 1 hora
def get_monthly_summary(start_date=None, end_date=None, months=6):
    """Obtener resumen mensual de activities para un rango de fechas"""
    try:
        # Cargar datos de entrenamiento
        df = load_training_data()
        if df.empty:
            return pd.DataFrame(columns=['mes', 'entrenamientos', 'meetings', 'review_clips', 'mes_formateado'])
        
        # Si no se especifican fechas, usar los últimos 'months' meses
        if not start_date or not end_date:
            start_date = (datetime.now() - pd.DateOffset(months=months-1)).replace(day=1)
            end_date = datetime.now()
        
        # Convertir a datetime si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Ajustar la fecha de fin para incluir todo el día
        end_date = datetime.combine(end_date, datetime.max.time())
        
        # Filtrar datos por rango de fechas
        date_mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date.date())
        df_filtered = df[date_mask].copy()
        
        if df_filtered.empty:
            return pd.DataFrame(columns=['mes', 'entrenamientos', 'meetings', 'review_clips', 'mes_formateado'])
        
        # Agrupar por mes y contar activities
        monthly_summary = df_filtered.groupby('Year_Month').agg({
            'Training_Type': 'count',
            'Meeting': lambda x: x.notna().sum(),
            'Review_Clips': lambda x: x.notna().sum()
        }).reset_index()
        
        # Renombrar columnas
        monthly_summary = monthly_summary.rename(columns={
            'Year_Month': 'mes',
            'Training_Type': 'entrenamientos',
            'Meeting': 'meetings',
            'Review_Clips': 'review_clips'
        })
        
        # Ordenar por fecha
        monthly_summary = monthly_summary.sort_values('mes')
        
        # Formatear fechas para mostrar
        monthly_summary['mes_formateado'] = pd.to_datetime(monthly_summary['mes']).dt.strftime('%b %Y')
        
        return monthly_summary
    except Exception as e:
        st.error(f"Error loading monthly summary: {e}")
        return pd.DataFrame(columns=['mes', 'entrenamientos', 'meetings', 'review_clips', 'mes_formateado'])

@st.cache_data(ttl=3600)  # Cachear por 1 hora
def get_players_summary(start_date=None, end_date=None):
    """Obtener resumen de activities por jugador para un rango de fechas"""
    try:
        # Cargar datos de entrenamiento
        df = load_training_data()
        if df.empty:
            return pd.DataFrame(columns=['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_activities'])
        
        # Filtrar por rango de fechas si se especifican
        if start_date and end_date:
            # Convertir a datetime si son strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()
                
            # Ajustar la fecha de fin para incluir todo el día
            end_date = datetime.combine(end_date, datetime.max.time())
            
            # Filtrar por rango de fechas
            date_mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date.date())
            filtered_data = df[date_mask].copy()
        else:
            # Si no se especifican fechas, usar el mes actual
            current_year_month = datetime.now().strftime('%Y-%m')
            filtered_data = df[df['Year_Month'] == current_year_month].copy()
        
        if filtered_data.empty:
            return pd.DataFrame(columns=['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_activities'])
        
        # Agrupar por jugador y contar activities
        player_summary = filtered_data.groupby('Player').agg({
            'Training_Type': 'count',
            'Meeting': lambda x: x.notna().sum(),
            'Review_Clips': lambda x: x.notna().sum()
        }).reset_index()
        
        # Renombrar columnas
        player_summary = player_summary.rename(columns={
            'Player': 'jugador',
            'Training_Type': 'entrenamientos',
            'Meeting': 'meetings',
            'Review_Clips': 'review_clips'
        })
        
        # Calcular total de activities
        player_summary['total_activities'] = (
            player_summary['entrenamientos'] + 
            player_summary['meetings'] + 
            player_summary['review_clips']
        )
        
        # Ordenar por total de activities (descendente)
        player_summary = player_summary.sort_values('total_activities', ascending=False)
        
        return player_summary
    except Exception as e:
        st.error(f"Error loading player summary: {e}")
        return pd.DataFrame(columns=['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_activities'])

# Cargar datos iniciales (sin filtros)
current_month_metrics = get_current_month_metrics()
# No cargar monthly_summary aquí, se cargará con las fechas seleccionadas
players_summary = get_players_summary()

# --- Sidebar ---
with st.sidebar:

    # Time Filter
    st.markdown("### Time Filter")
    fecha_inicio = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    fecha_fin = st.date_input("End Date", datetime.now())  


    st.markdown("---")    
    st.title("Individual Development")
      

    # Navegación
    page = st.radio(
        "Select a section:",
        ["General Dashboard", "Players Profile", "Files"],
        label_visibility="collapsed"
    )
    
    
    # Información del usuario
    st.markdown("---")
    st.markdown("### User Info")
    if "staff_info" in st.session_state:
        st.write(f"**Name:** {st.session_state.staff_info['full_name']}")
        st.write(f"**Role:** {st.session_state.staff_info['role']}")
    
    if st.button("Logout", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Contenido principal ---

# 1. Dashboard General
if page == "General Dashboard":
    # Mostrar logo en la parte superior
    if logo:
        st.image(logo, width=100)
    
    st.header("General Dashboard")
    
    # Mostrar indicador de carga mientras se obtienen los datos
    with st.spinner('Loading dashboard data...'):
        
        # Obtener métricas para el rango de fechas seleccionado
        metrics = get_current_month_metrics(fecha_inicio, fecha_fin)
        
        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Trainings</div>
                <div class='metric-value'>{metrics['entrenamientos']}</div>
                <div class='metric-delta'>+2 vs previous month</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Meetings</div>
                <div class='metric-value'>{metrics['meetings']}</div>
                <div class='metric-delta'>+1 vs previous month</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Review Clips</div>
                <div class='metric-value'>{metrics['review_clips']}</div>
                <div class='metric-delta'>{'-1' if metrics['review_clips'] > 0 else '0'} vs previous month</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Participation</div>
                <div class='metric-value'>{metrics['jugadores_activos']}/{metrics['total_jugadores']}</div>
                <div class='metric-delta'>{metrics['porcentaje_participacion']}% of players</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de evolución mensual
    
        # Obtener resumen mensual para el rango de fechas seleccionado
        monthly_summary = get_monthly_summary(fecha_inicio, fecha_fin)
        
        if not monthly_summary.empty:
            # Crear gráfico de barras agrupadas
            fig = go.Figure()
            
            # Añadir barras para cada tipo de actividad
            fig.add_trace(go.Bar(
                x=monthly_summary['mes_formateado'],
                y=monthly_summary['entrenamientos'],
                name='Trainings',
                marker_color='#fcec03',  # Amarillo Watford
                hovertemplate='%{y} trainings<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                x=monthly_summary['mes_formateado'],
                y=monthly_summary['meetings'],
                name='Meetings',
                marker_color='#ff6b6b',  # Rojo
                hovertemplate='%{y} meetings<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                x=monthly_summary['mes_formateado'],
                y=monthly_summary['review_clips'],
                name='Review Clips',
                marker_color='#4ecdc4',  # Turquesa
                hovertemplate='%{y} review clips<extra></extra>'
            ))
            
            # Actualizar diseño del gráfico
            fig.update_layout(
                barmode='group',
                xaxis_title='Month',
                yaxis_title='Activity Count',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available to display the evolution chart.")
        
        # Resumen por jugador
        st.subheader(f"Summary All Players - ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
        
        # Obtener resumen por jugador para el rango de fechas seleccionado
        players_summary = get_players_summary(fecha_inicio, fecha_fin)
        
        # Mostrar tabla con resumen por jugador
        if not players_summary.empty:
            st.dataframe(
                players_summary[['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_activities']],
                column_config={
                    "jugador": "Player",
                    "entrenamientos": st.column_config.NumberColumn("Trainings", format="%d"),
                    "meetings": st.column_config.NumberColumn("Meetings", format="%d"),
                    "review_clips": st.column_config.NumberColumn("Review Clips", format="%d"),
                    "total_activities": st.column_config.NumberColumn(
                        "Total activities", 
                        format="%d",
                        help="Sum of all player activities"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Gráfico de evolución por jugador
            
            # Obtener datos para el gráfico
            df_activities = load_training_data()
            
            # Filtrar por rango de fechas
            date_mask = (df_activities['Date'].dt.date >= fecha_inicio) & (df_activities['Date'].dt.date <= fecha_fin)
            df_activities = df_activities[date_mask].copy()
            
            if not df_activities.empty:
                # Crear una columna para el mes-año
                df_activities['mes_anio'] = df_activities['Date'].dt.to_period('M').astype(str)
                
                # Agrupar por jugador y mes para contar activities
                df_agrupado = df_activities.groupby(['Player', 'mes_anio']).agg({
                    'Training_Type': 'count',
                    'Meeting': lambda x: x.notna().sum(),
                    'Review_Clips': lambda x: x.notna().sum()
                }).reset_index()
                
                # Renombrar columnas
                df_agrupado = df_agrupado.rename(columns={
                    'Player': 'Player',
                    'mes_anio': 'Month',
                    'Training_Type': 'Trainings',
                    'Meeting': 'Meetings',
                    'Review_Clips': 'Review_Clips'
                })
                
                # Ordenar por mes
                df_agrupado = df_agrupado.sort_values('Month')
                
                # Crear pestañas para cada tipo de actividad
                tab1, tab2, tab3 = st.tabs(["Trainings", "Meetings", "Review Clips"])
                
                with tab1:
                    fig_entrenamientos = px.line(
                        df_agrupado, 
                        x='Month', 
                        y='Trainings',
                        color='Player',
                        title='Training Evolution by Player',
                        labels={'Trainings': 'Count', 'Month': 'Month'},
                        markers=True
                    )
                    fig_entrenamientos.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Training Count',
                        legend_title='Player',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_entrenamientos, use_container_width=True)
                
                with tab2:
                    fig_meetings = px.line(
                        df_agrupado, 
                        x='Month', 
                        y='Meetings',
                        color='Player',
                        title='Meeting Evolution by Player',
                        labels={'Meetings': 'Count', 'Month': 'Month'},
                        markers=True
                    )
                    fig_meetings.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Meeting Count',
                        legend_title='Player',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_meetings, use_container_width=True)
                
                with tab3:
                    fig_clips = px.line(
                        df_agrupado, 
                        x='Month', 
                        y='Review_Clips',
                        color='Player',
                        title='Review Clips Evolution by Player',
                        labels={'Review_Clips': 'Count', 'Month': 'Month'},
                        markers=True
                    )
                    fig_clips.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Review Clips Count',
                        legend_title='Player',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_clips, use_container_width=True)
            else:
                st.warning("No activity data to display in the selected date range.")
        else:
            st.info("No activity data to display in the selected period.")
            
        # Información adicional
        st.markdown("---")
        st.markdown("""
        **Notes:**
        - If no data displayed check time filters or Upload the File to display the data.
        """)

# 2. Perfil Individual
elif page == "Players Profile":
    st.header("Players Profile")
    
    # Obtener lista de jugadores activos
    @st.cache_data(ttl=3600)  # Cachear por 1 hora
    def load_active_players():
        try:
            # Cargar datos de entrenamiento
            df = load_training_data()
            if df.empty:
                return {}
                
            # Obtener lista única de jugadores
            jugadores = df['Player'].dropna().unique()
            # Crear un diccionario con índice numérico para compatibilidad
            return {i+1: jugador for i, jugador in enumerate(jugadores)}
        except Exception as e:
            st.error(f"Error loading active players: {e}")
            return {}
    
    # Obtener activities del jugador
    @st.cache_data(ttl=600)  # Cachear por 10 minutos
    def load_player_activities(player_id, start_date, end_date):
        try:
            # Cargar datos de entrenamiento
            df = load_training_data()
            if df.empty:
                return pd.DataFrame()
            
            # Obtener el nombre del jugador a partir del ID
            jugadores = load_active_players()
            jugador_nombre = jugadores.get(player_id)
            
            if not jugador_nombre:
                return pd.DataFrame()
            
            # Convertir las fechas proporcionadas a pandas datetime
            fecha_inicio_dt = pd.to_datetime(start_date)
            fecha_fin_dt = pd.to_datetime(end_date)
            
            # Filtrar por jugador usando el filtro de fechas del sidebar
            activities = df[
                (df['Player'] == jugador_nombre) &
                (df['Date'] >= fecha_inicio_dt) &
                (df['Date'] <= fecha_fin_dt)
            ].copy()
            
            if activities.empty:
                return pd.DataFrame()
            
            # Crear un DataFrame con las activities en formato largo
            activities_largas = []
            
            # Procesar entrenamientos
            entrenamientos = activities[activities['Training_Type'].notna()]
            for _, row in entrenamientos.iterrows():
                activities_largas.append({
                    'fecha': row['Date'],
                    'tipo': 'Training',
                    'subtipo': row['Training_Type'],
                    'descripcion': f"Training: {row['Training_Type']}",
                    'jugador': row['Player']
                })
            
            # Procesar meetings
            meetings = activities[activities['Meeting'].notna()]
            for _, row in meetings.iterrows():
                activities_largas.append({
                    'fecha': row['Date'],
                    'tipo': 'Meeting',
                    'subtipo': row['Meeting'],
                    'descripcion': f"Meeting: {row['Meeting']}",
                    'jugador': row['Player']
                })
            
            # Procesar review clips
            reviews = activities[activities['Review_Clips'].notna()]
            for _, row in reviews.iterrows():
                activities_largas.append({
                    'fecha': row['Date'],
                    'tipo': 'Review Clip',
                    'subtipo': 'Video review',
                    'descripcion': f"Review Clip: {row['Review_Clips']}",
                    'jugador': row['Player']
                })
            
            # Crear DataFrame con todas las activities
            if activities_largas:
                return pd.DataFrame(activities_largas).sort_values('fecha', ascending=False)
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error loading player activities: {e}")
            return pd.DataFrame()
    
    # Cargar jugadores activos
    jugadores = load_active_players()
    
    if not jugadores:
        st.warning("No active players found in training data.")
        st.stop()
    
    # Selector de jugador principal
    with st.expander("Player", expanded=True):
        jugador_id = st.selectbox(
            "Select Player",
            options=list(jugadores.keys()),
            format_func=lambda x: jugadores[x],
            index=0
        )
        jugador_seleccionado = jugadores[jugador_id]

    # Selector para comparar jugadores
    with st.expander("Compare with other players", expanded=False):
        jugadores_comparar = st.multiselect(
            "Select players",
            options=list(jugadores.keys()),
            format_func=lambda x: jugadores[x],
            default=[list(jugadores.keys())[0]]  # Default: first player
        )
        jugadores_comparar = [jugadores[jugador_id] for jugador_id in jugadores_comparar]


    st.subheader("Summary Activities – Compared Players")
    pdf_comparison_fig = None
    pdf_timeline_fig = None
    df_pdf_activities = pd.DataFrame()
    selected_types_for_pdf = []

    # Load full training data
    df_all = load_training_data()

    # Combine main player and comparison players
    all_selected_players = [jugador_seleccionado] + [
        p for p in jugadores_comparar if p != jugador_seleccionado
    ]

    # Filter by selected players and date range
    df_filtered = df_all[
        (df_all["Player"].isin(all_selected_players)) &
        (df_all["Date"] >= pd.to_datetime(fecha_inicio)) &
        (df_all["Date"] <= pd.to_datetime(fecha_fin))
    ]

    # Aggregate activity counts
    summary = df_filtered.groupby("Player").agg(
        Trainings=pd.NamedAgg(column="Training_Type", aggfunc=lambda x: x.notna().sum()),
        Meetings=pd.NamedAgg(column="Meeting", aggfunc=lambda x: x.notna().sum()),
        Review_Clips=pd.NamedAgg(column="Review_Clips", aggfunc=lambda x: x.notna().sum())
    ).reset_index()

    # Add total column
    summary["Total activities"] = summary[["Trainings", "Meetings", "Review_Clips"]].sum(axis=1)

    # Reorder rows: main player first
    summary = pd.concat([
        summary[summary["Player"] == jugador_seleccionado],
        summary[summary["Player"] != jugador_seleccionado]
    ], ignore_index=True)

    # Highlight the main player
    def highlight_main_player(row):
        if row["Player"] == jugador_seleccionado:
            return ['background-color: #fcec03; font-weight: bold'] * len(row)
        return [''] * len(row)

   # Display styled dataframe
    st.dataframe(
        summary.style.apply(highlight_main_player, axis=1),
        use_container_width=True
    )

    # --- Create grouped bar chart per player ---
    comparison_fig = go.Figure()

    # Trainings
    comparison_fig.add_trace(go.Bar(
        x=summary["Player"],
        y=summary["Trainings"],
        name="Trainings",
        marker_color=[
            "#fcec03" if player == jugador_seleccionado else "#d3d3d3"
            for player in summary["Player"]
        ],
        hovertemplate="%{y} trainings<extra></extra>"
    ))

    # Meetings
    comparison_fig.add_trace(go.Bar(
        x=summary["Player"],
        y=summary["Meetings"],
        name="Meetings",
        marker_color="#ff6b6b",
        hovertemplate="%{y} meetings<extra></extra>"
    ))

    # Review Clips
    comparison_fig.add_trace(go.Bar(
        x=summary["Player"],
        y=summary["Review_Clips"],
        name="Review Clips",
        marker_color="#4ecdc4",
        hovertemplate="%{y} review clips<extra></extra>"
    ))

    # Layout tweaks
    comparison_fig.update_layout(
        barmode="group",
        xaxis_title="Player",
        yaxis_title="Activity Count",
        legend_title="Type",
        plot_bgcolor="white",
        title="Activities by Player",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(comparison_fig, use_container_width=True)
    pdf_comparison_fig = comparison_fig

    # --- Load individual activities for selected player ---
    with st.spinner('Loading player data...'):
        df_activities = load_player_activities(jugador_id, fecha_inicio, fecha_fin)
        
        jugador_info = next((v for k, v in jugadores.items() if k == jugador_id), "")
        
        st.subheader(f"Individual Activities - {jugador_info}")

        if not df_activities.empty:
            tipos_disponibles = df_activities["tipo"].unique().tolist()
            tipos_seleccionados = st.multiselect(
                "Select activities",
                options=tipos_disponibles,
                default=tipos_disponibles,
                key="filtro_tipo_activities"
            )

            df_filtrado = df_activities[df_activities["tipo"].isin(tipos_seleccionados)]

            if not df_filtrado.empty:
                df_mostrar = df_filtrado.copy()
                df_mostrar["fecha"] = df_mostrar["fecha"].dt.strftime("%d/%m/%Y")
                df_mostrar = df_mostrar[["fecha", "tipo", "subtipo", "descripcion"]]
                df_pdf_activities = df_filtrado.copy()
                selected_types_for_pdf = list(tipos_seleccionados)

                st.dataframe(
                    df_mostrar,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "fecha": "Date",
                        "tipo": "Activity Type",
                        "subtipo": "Detail",
                        "descripcion": "Description"
                    }
                )

                # Botón para exportar las activities filtradas a Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_mostrar.to_excel(writer, index=False, sheet_name="Activities")

                buffer.seek(0)

                st.download_button(
                    label="📥 Export to Excel",
                    data=buffer,
                    file_name=f"individual_activities_{jugador_info.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No activities for the selected type.")
                df_pdf_activities = df_filtrado.copy()
                selected_types_for_pdf = list(tipos_seleccionados)
        else:
            st.info("No activities registered for this player.")


        # Timeline de activities
        st.subheader(f"Activity Timeline - {jugador_info}")
        
        if not df_activities.empty:
            # Crear gráfico de dispersión para la timeline
            # Usar solo las columnas disponibles en el DataFrame
            hover_columns = []
            if 'descripcion' in df_activities.columns:
                hover_columns.append('descripcion')
            if 'subtipo' in df_activities.columns:
                hover_columns.append('subtipo')
                
            timeline_fig = px.scatter(
                df_activities,
                x='fecha',
                y='tipo',
                color='tipo',
                color_discrete_map={
                    'Training': '#fcec03',  # Amarillo Watford
                    'Meeting': '#1f77b4',        # Azul
                    'Review Clip': '#2ca02c'      # Verde
                },
                size=[1] * len(df_activities),  # Size fijo para todos los puntos
                hover_data=hover_columns if hover_columns else None,
                labels={'fecha': 'Date', 'tipo': 'Activity Type'},
                title=f"({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})"
            )
            
            timeline_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Date',
                yaxis_title='',
                showlegend=True,
                legend_title='Activity Type',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(timeline_fig, use_container_width=True)
            pdf_timeline_fig = timeline_fig

        if df_pdf_activities.empty and not df_activities.empty:
            df_pdf_activities = df_activities.copy()
            selected_types_for_pdf = sorted(df_pdf_activities["tipo"].dropna().astype(str).unique().tolist())

    st.markdown("---")
    st.subheader("PDF Report")

    cover_player_key = f"individual_{jugador_id}_{jugador_seleccionado}"
    cover_session_key = f"individual_profile_cover_photo_path_{jugador_id}"
    if cover_session_key not in st.session_state:
        history = list_cover_photos(base_dir=BASE_DIR, player_key=cover_player_key)
        st.session_state[cover_session_key] = history[0]["path"] if history else None

    if st.button("Manage player cover photo", key="manage_individual_profile_cover_photo"):
        _pdf_cover_photo_dialog(
            player_key=cover_player_key,
            player_label=jugador_seleccionado,
            session_key=cover_session_key,
            key_prefix=f"individual_pdf_cover_{jugador_id}",
        )

    selected_cover_photo_path = st.session_state.get(cover_session_key)
    if selected_cover_photo_path and os.path.exists(selected_cover_photo_path):
        st.caption("Active player cover photo")
        st.image(selected_cover_photo_path, width=140)
    else:
        st.caption("No player cover photo selected.")

    cover_photo_signature = "none"
    if selected_cover_photo_path and os.path.exists(selected_cover_photo_path):
        cover_photo_signature = f"{selected_cover_photo_path}:{int(os.path.getmtime(selected_cover_photo_path))}"

    report_signature = (
        f"{jugador_seleccionado}|{fecha_inicio}|{fecha_fin}|"
        f"{','.join(sorted(selected_types_for_pdf))}|{len(df_pdf_activities)}|"
        f"{cover_photo_signature}"
    )
    if st.session_state.get("individual_profile_pdf_signature") != report_signature:
        st.session_state["individual_profile_pdf_signature"] = report_signature
        st.session_state.pop("individual_profile_pdf_bytes", None)
        st.session_state.pop("individual_profile_pdf_name", None)

    if df_pdf_activities.empty:
        st.info("No activities in the current range/filters to generate PDF.")
    else:
        generating_key = "individual_profile_pdf_generating"
        auto_download_key = "individual_profile_pdf_auto_download_pending"
        if generating_key not in st.session_state:
            st.session_state[generating_key] = False
        if auto_download_key not in st.session_state:
            st.session_state[auto_download_key] = False

        if st.button(
            "Generate & Download PDF report",
            key="generate_individual_profile_pdf",
            type="primary",
            disabled=st.session_state.get(generating_key, False),
        ):
            st.session_state[generating_key] = True
            st.session_state[auto_download_key] = True
            st.rerun()

        if st.session_state.get(generating_key, False):
            overlay_placeholder = st.empty()
            overlay_placeholder.markdown(
                """
                <style>
                  .pdf-generation-overlay {
                    position: fixed;
                    inset: 0;
                    background: rgba(0, 0, 0, 0.45);
                    z-index: 2147483000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    pointer-events: all;
                  }
                  .pdf-generation-overlay-card {
                    background: #111;
                    color: #fff;
                    padding: 1rem 1.25rem;
                    border-radius: 12px;
                    font-weight: 600;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                  }
                </style>
                <div class="pdf-generation-overlay">
                  <div class="pdf-generation-overlay-card">Generating PDF report... Please wait.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                try:
                    df_summary_pdf = (
                        df_pdf_activities
                        .groupby("tipo", dropna=False)
                        .size()
                        .reset_index(name="count")
                    )
                    df_summary_pdf["tipo"] = df_summary_pdf["tipo"].fillna("No type")

                    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(jugador_info).strip())
                    safe_name = safe_name.strip("_") or "player"
                    pdf_file_name = f"{safe_name}_individual_development.pdf"

                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = generate_individual_development_report_landscape(
                            player_name=jugador_info,
                            fecha_inicio=fecha_inicio.strftime("%Y-%m-%d"),
                            fecha_fin=fecha_fin.strftime("%Y-%m-%d"),
                            df_activities=df_pdf_activities,
                            df_summary=df_summary_pdf,
                            df_ratings=None,
                            fig_comparison=pdf_comparison_fig,
                            fig_timeline=pdf_timeline_fig,
                            logo_path=LOGO_PATH,
                            background_image_path=BACKGROUND_COVER_PATH if os.path.exists(BACKGROUND_COVER_PATH) else None,
                            player_photo_path=selected_cover_photo_path if (selected_cover_photo_path and os.path.exists(selected_cover_photo_path)) else None,
                            cover_only=True,
                        )

                    if not pdf_bytes:
                        raise ValueError("Generated PDF is empty.")

                    st.session_state["individual_profile_pdf_bytes"] = pdf_bytes
                    st.session_state["individual_profile_pdf_name"] = pdf_file_name
                    st.success("PDF generated. Download should start automatically.")
                except Exception as e:
                    st.session_state[auto_download_key] = False
                    st.error(f"Failed to generate PDF report: {e}")
            finally:
                st.session_state[generating_key] = False
                overlay_placeholder.empty()

        if st.session_state.get(auto_download_key, False) and st.session_state.get("individual_profile_pdf_bytes"):
            pdf_b64 = base64.b64encode(st.session_state["individual_profile_pdf_bytes"]).decode("utf-8")
            file_name = st.session_state.get("individual_profile_pdf_name", "individual_development_report.pdf")
            components.html(
                f"""
                <script>
                  (function() {{
                    const a = document.createElement('a');
                    a.href = 'data:application/pdf;base64,{pdf_b64}';
                    a.download = '{file_name}';
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                  }})();
                </script>
                """,
                height=0,
            )
            st.session_state[auto_download_key] = False
            st.caption("If download does not start automatically, use the fallback button below.")

        if st.session_state.get("individual_profile_pdf_bytes"):
            st.download_button(
                "Download PDF report (fallback)",
                data=st.session_state["individual_profile_pdf_bytes"],
                file_name=st.session_state.get("individual_profile_pdf_name", "individual_development_report.pdf"),
                mime="application/pdf",
                key="download_individual_profile_pdf",
            )
            

# 3. Registro de Activities
elif page == "Files":
    st.header("Files")
    st.write("Register new individual development activities for players.")

    st.markdown("### ➕ Add New Training Session")
    try:
        players_source_df = load_training_data()
        player_options = sorted(players_source_df["Player"].dropna().astype(str).str.strip().unique().tolist()) if not players_source_df.empty else []
    except Exception:
        player_options = []

    with st.form("add_single_training_session_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_date = st.date_input("Date", value=datetime.now().date(), key="new_training_date")
        with col2:
            if player_options:
                selected_player = st.selectbox("Player", options=player_options, index=0, key="new_training_player_select")
                custom_player = st.text_input("Or type new player name (optional)", key="new_training_player_custom")
                player_name = custom_player.strip() if custom_player.strip() else selected_player
            else:
                player_name = st.text_input("Player", key="new_training_player_text")

        training_text = st.text_input("Individual Training", placeholder="e.g. Defensive transitions", key="new_training_activity")
        meeting_text = st.text_input("Meeting", placeholder="Optional", key="new_meeting_activity")
        review_text = st.text_input("Review Clips", placeholder="Optional", key="new_review_activity")

        add_row = st.form_submit_button("Add row to sessions")

    if add_row:
        player_name = str(player_name).strip()
        training_text = str(training_text).strip()
        meeting_text = str(meeting_text).strip()
        review_text = str(review_text).strip()

        if not player_name:
            st.error("Player name is required.")
        elif not any([training_text, meeting_text, review_text]):
            st.error("Add at least one activity: Training, Meeting, or Review Clips.")
        else:
            try:
                destination = _append_training_session_row(
                    session_date=new_date,
                    player_name=player_name,
                    training_text=training_text,
                    meeting_text=meeting_text,
                    review_text=review_text,
                )

                if 'load_training_data' in globals():
                    load_training_data.clear()
                if 'get_current_month_metrics' in globals():
                    get_current_month_metrics.clear()
                if 'get_monthly_summary' in globals():
                    get_monthly_summary.clear()
                if 'get_players_summary' in globals():
                    get_players_summary.clear()

                st.success(f"Session added successfully to {destination}.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not add session row: {e}")

    # Descargar snapshot actual (desde Google Sheets / Sesions o fallback local)
    current_sessions_df = _load_sessions_raw_df()
    if not current_sessions_df.empty:
        download_buffer = io.BytesIO()
        with pd.ExcelWriter(download_buffer, engine="openpyxl") as writer:
            current_sessions_df.to_excel(writer, index=False, sheet_name="Sesions")
        download_buffer.seek(0)
        st.download_button(
            label="📥 Download current data",
            data=download_buffer.getvalue(),
            file_name="Individuals-Training.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No current data in the sessions source.")


    # Función para validar y cargar el archivo Excel específico
    def validar_importar_excel(file_path):
        try:
            print(f"Validating file: {file_path.name if hasattr(file_path, 'name') else file_path}")
            
            # Leer solo las primeras filas para verificar la estructura
            df_preview = pd.read_excel(file_path, nrows=5)
            print("First rows of the file:")
            print(df_preview.head())
            print("\nFile columns:", df_preview.columns.tolist())
            
            # Verificar si el archivo tiene datos
            if df_preview.empty:
                st.markdown("""
                <div style='background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;'>
                    The file is empty. Please check that the file contains data.
                </div>
                """, unsafe_allow_html=True)
                return False, "The file is empty. Please check that the file contains data."
            
            # Verificar si el archivo tiene el formato esperado
            expected_columns = ['Month', 'Player', 'Individual Training', 'Meeting', 'Review Clips']
            
            # Verificar si alguna de las filas contiene los encabezados esperados
            header_row = None
            posibles_encabezados = []
            
            for i in range(min(5, len(df_preview) + 1)):  # Revisar hasta las primeras 5 filas
                try:
                    df_test = pd.read_excel(file_path, header=i, nrows=1)
                    # Verificar si encontramos los encabezados esperados (ignorando mayúsculas y espacios)
                    columnas_archivo = [str(col).strip().lower() for col in df_test.columns]
                    
                    # Verificar si se encuentran los encabezados clave
                    if all(any(esperada in col_archivo for col_archivo in columnas_archivo) 
                          for esperada in ['mes', 'player']):
                        header_row = i
                        break
                        
                    # Guardar los encabezados para mostrarlos en el mensaje de error si es necesario
                    posibles_encabezados.append(columnas_archivo)
                    
                except Exception as e:
                    print(f"Error reading file with header={i}: {str(e)}")
                    continue
            
            if header_row is None:
                # Si no se encontraron los encabezados, mostrar un mensaje de error detallado
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                    <h4 style='margin-top: 0;'>❌ File format error</h4>
                    <p>The file does not match the expected format. Make sure it contains the following columns:</p>
                    <ul style='margin-bottom: 10px;'>
                        <li><strong>Month</strong> (activity date)</li>
                        <li><strong>Player</strong> (player name)</li>
                        <li><strong>Individual Training</strong> (training activities)</li>
                        <li><strong>Meeting</strong> (meetings)</li>
                        <li><strong>Review Clips</strong> (video review)</li>
                    </ul>
                    <p style='margin-bottom: 5px;'><strong>Tips:</strong></p>
                    <ul style='margin-top: 5px;'>
                        <li>Check that the first row contains headers</li>
                        <li>Column names must match exactly (case and spacing may vary)</li>
                        <li>Dates must be in a recognizable format (DD/MM/YYYY or YYYY-MM-DD)</li>
                    </ul>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "The file does not have the expected format. Verify it includes required columns."
            
            # Leer el archivo completo con el encabezado correcto
            df = pd.read_excel(file_path, header=header_row)
            
            # Limpiar los nombres de las columnas (eliminar espacios en blanco)
            df.columns = [str(col).strip() for col in df.columns]
            
            # Verify that required columns are present
            date_col = next((c for c in ['Mes', 'Month', 'Date', 'Fecha'] if c in df.columns), None)
            required_columns = ['Player']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns or date_col is None:
                missing_text = ', '.join(missing_columns) if missing_columns else 'Date column (Mes/Month/Date/Fecha)'
                error_msg = f"""
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Missing required columns</h4>
                    <p>The following required columns are missing: <strong>{missing_text}</strong></p>
                    <p>Please ensure the file contains at least a date column (Mes/Month/Date/Fecha) and 'Player'.</p>
                    <p>Columns found in file: {', '.join(df.columns) if len(df.columns) > 0 else 'No columns found'}</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, f"Missing required columns: {missing_text}"
            
            # Verify that there is at least one activity column
            activity_columns = [col for col in df.columns if col not in [date_col, 'Player']]
            if not activity_columns:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ No activities found</h4>
                    <p>No activity columns found in the file.</p>
                    <p>Ensure the file contains at least one activity column (Individual Training, Meeting, or Review Clips).</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No activity columns found in the file."
            
            # Verificar que haya al menos una fila con datos
            if df.empty:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Empty file</h4>
                    <p>The file is empty or contains no valid data.</p>
                    <p>Please verify the file has data in rows below headers.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "The file is empty or contains no valid data."
            
            # Verificar que las fechas sean válidas
            try:
                # Convertir la columna de fechas a datetime
                df['fecha_dt'] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Verificar si hay fechas inválidas
                if df['fecha_dt'].isna().any():
                    fechas_invalidas = df[df['fecha_dt'].isna()][date_col].head().tolist()
                    error_msg = f"""
                    <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                        <h4 style='margin-top: 0;'>❌ Invalid dates</h4>
                        <p>Invalid dates were found in column '{date_col}'.</p>
                        <p>Examples of problematic values: {', '.join(map(str, fechas_invalidas))}</p>
                        <p>Ensure dates are in a recognizable format (e.g., DD/MM/YYYY or YYYY-MM-DD).</p>
                    </div>
                    """
                    st.markdown(error_msg, unsafe_allow_html=True)
                    return False, f"Invalid dates were found: {', '.join(map(str, fechas_invalidas[:3]))}..."
                
                # Verificar que las fechas estén en un rango razonable
                fecha_min = pd.Timestamp('2020-01-01')
                fecha_max = pd.Timestamp.now() + pd.DateOffset(years=1)
                
                fechas_fuera_de_rango = df[(df['fecha_dt'] < fecha_min) | (df['fecha_dt'] > fecha_max)]
                if not fechas_fuera_de_rango.empty:
                    ejemplos = fechas_fuera_de_rango['fecha_dt'].dt.strftime('%Y-%m-%d').head().tolist()
                    error_msg = f"""
                    <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                        <h4 style='margin-top: 0;'>❌ Dates out of range</h4>
                        <p>Some dates are outside the allowed range (01/01/2020 - {fecha_max.strftime('%d/%m/%Y')}).</p>
                        <p>Problematic dates: {', '.join(ejemplos)}</p>
                        <p>Please verify all dates are within the allowed range.</p>
                    </div>
                    """
                    st.markdown(error_msg, unsafe_allow_html=True)
                    return False, f"Dates out of range: {', '.join(ejemplos[:3])}..."
                
                # Formatear las fechas
                df['fecha_formateada'] = df['fecha_dt'].dt.strftime('%Y-%m-%d')
                
            except Exception as e:
                error_msg = f"""
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Error processing dates</h4>
                    <p>An error occurred while processing dates in column '{date_col}'.</p>
                    <p>Error: {str(e)}</p>
                    <p>Please verify all dates are in a valid format and try again.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, f"Error processing dates: {str(e)}"
            
            # Verificar que haya al menos un jugador con nombre válido
            jugadores_invalidos = df[df['Player'].isna() | (df['Player'].astype(str).str.strip() == '')]
            if len(jugadores_invalidos) == len(df):
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Invalid player names</h4>
                    <p>No valid player names were found in column 'Player'.</p>
                    <p>Please ensure the 'Player' column contains player names.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No valid player names were found in column 'Player'."
            
            # Verificar que haya al menos una actividad registrada
            activities = []
            for _, fila in df.iterrows():
                # Saltar filas sin jugador o con jugador vacío
                if pd.isna(fila.get('Player')) or str(fila.get('Player', '')).strip() == '':
                    continue
                    
                for col in activity_columns:
                    if pd.notna(fila.get(col)) and str(fila[col]).strip() != '':
                        activities.append({
                            'fecha': fila['fecha_formateada'],
                            'jugador': str(fila['Player']).strip(),
                            'tipo': col,
                            'descripcion': str(fila[col]).strip()
                        })
            
            if not activities:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ No valid activities</h4>
                    <p>No valid activities were found in the file.</p>
                    <p>Ensure at least one cell in activity columns contains data.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No valid activities were found in the file."
            
            # Crear el DataFrame final con las activities
            df_activities = pd.DataFrame(activities)
            
            # Verificar que tengamos al menos una actividad válida
            if df_activities.empty:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Could not process activities</h4>
                    <p>No valid activities were found para importar.</p>
                    <p>Please verify the file format and try again.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "Could not process activities. Check the file format."
            
            # Guardar dataset tabular validado para subirlo a la pestaña Sesions.
            st.session_state["training_import_raw_df"] = df.copy()
            return True, df_activities
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing file: {error_details}")
            
            # Mensaje de error más amigable
            mensaje_error = f"""
            <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                <h4 style='margin-top: 0;'>❌ Error processing file</h4>
                <p>An error occurred while processing the file. Please verify it matches the expected format.</p>
                <p><strong>Error details:</strong> {str(e)}</p>
                <p>If the problem persists, try:</p>
                <ul>
                    <li>Download the sample template and use that format</li>
                    <li>Check that the file is not corrupted</li>
                    <li>Check that it has no formulas or special formatting</li>
                </ul>
            </div>
            """
            return False, mensaje_error
    
    # Sección para importar datos desde Excel
    with st.expander("📤 Import Data from Excel", expanded=False):
        st.markdown("### Import instructions")
        st.markdown("""
        1. **File format**: The Excel file must have the following structure:
           - **Column 1 (A)**: Date de la actividad (formato de fecha reconocible)
           - **Column 2 (B)**: Player name
           - **Column 3 (C)**: Individual training details (optional)
           - **Column 4 (D)**: Meeting details (optional)
           - **Column 5 (E)**: Review clips details (optional)
        
        2. **Requirements**:
           - The file must be in .xlsx format
           - The first row must contain headers
           - At least one activity must be registered in columns 3-5
           - Dates must be between 2020 and next year
        
        3. **Tips**:
           - Make sure player names are consistent
           - Check that dates have the correct format
           - Verify that at least one activity cell contains information
        """)
        
        st.markdown("---")
        st.markdown("### Upload Excel file")
        uploaded_file = st.file_uploader(
            "Select an Excel file (.xlsx)", 
            type=["xlsx"],
            help="Click or drag an Excel file with the expected format"
        )
        
        if uploaded_file is not None:
            # Mostrar información del archivo
            file_name = uploaded_file.name
            file_size = len(uploaded_file.getvalue()) / 1024  # Size en KB
            
            # Mostrar información del archivo en un contenedor con estilo
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("File", file_name)
                with col2:
                    st.metric("Size", f"{file_size:.1f} KB")
                
                # Botón para validar el archivo
                st.markdown("### Validate and process")
                
                if st.button("🔍 Validate file", key="validar_archivo", help="Validate file structure and data"):
                    with st.spinner("Validating file, please wait..."):
                        try:
                            # Validar y procesar el archivo
                            es_valido, resultado = validar_importar_excel(uploaded_file)
                            
                            # Guardar el estado de validación y los resultados en session_state
                            st.session_state['archivo_validado'] = es_valido
                            st.session_state['resultado_validacion'] = resultado
                            st.session_state['archivo_cargado'] = uploaded_file
                            if not es_valido:
                                st.session_state['training_import_raw_df'] = None
                            
                            # Si hay un error de validación, mostrarlo
                            if not es_valido and isinstance(resultado, str):
                                st.error(f"❌ {resultado}")
                            
                            # No hacemos rerun aquí para evitar problemas con el estado
                        except Exception as e:
                            st.error(f"❌ Unexpected error while validating file: {str(e)}")
                            st.session_state['archivo_validado'] = False
                            st.session_state['resultado_validacion'] = str(e)
                            st.session_state['training_import_raw_df'] = None
                
                # Si ya se validó el archivo, mostrar los resultados
                if st.session_state.get('archivo_validado', False):
                    resultado = st.session_state['resultado_validacion']
                    st.success("✅ Validation successful")
                    
                    # Mostrar estadísticas de las activities a importar
                    st.markdown("#### Import summary")
                    
                    # Calcular estadísticas
                    total_activities = len(resultado)
                    jugadores_unicos = resultado['jugador'].nunique()
                    fechas_unicas = resultado['fecha'].nunique()
                    activities_por_tipo = resultado['tipo'].value_counts().to_dict()
                    
                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total activities", total_activities)
                    with col2:
                        st.metric("👥 Unique players", jugadores_unicos)
                    with col3:
                        st.metric("📅 Days with activities", fechas_unicas)
                    
                    # Mostrar distribución por tipo de actividad
                    st.markdown("#### Distribution by Activity Type")
                    for tipo, cantidad in activities_por_tipo.items():
                        st.progress(cantidad / total_activities, f"{tipo}: {cantidad} activities")
                    
                    # Vista previa de los datos
                    st.markdown("#### Data preview")
                    st.dataframe(resultado.head(10), 
                                use_container_width=True,
                                column_config={
                                    'fecha': 'Date',
                                    'jugador': 'Player',
                                    'tipo': 'Activity Type',
                                    'descripcion': 'Description'
                                })
                    
                    # Botones de confirmación y cancelación
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirm import", type="primary", key="confirmar_importacion"):
                            try:
                                st.info("Processing import, please wait...")
                                raw_df_to_save = st.session_state.get("training_import_raw_df")

                                if raw_df_to_save is None or raw_df_to_save.empty:
                                    uploaded_file = st.session_state.get("archivo_cargado")
                                    if uploaded_file is None:
                                        raise ValueError("No validated file available to import.")
                                    uploaded_file.seek(0)
                                    raw_df_to_save = pd.read_excel(uploaded_file)

                                destination = _save_sessions_raw_df(raw_df_to_save)
                                st.success(f"✅ Data saved successfully to {destination}!")

                                # Limpiar cachés específicos
                                if 'load_training_data' in globals():
                                    load_training_data.clear()
                                if 'get_current_month_metrics' in globals():
                                    get_current_month_metrics.clear()
                                if 'get_monthly_summary' in globals():
                                    get_monthly_summary.clear()
                                if 'get_players_summary' in globals():
                                    get_players_summary.clear()

                                # Mostrar opción para descargar un reporte
                                csv = resultado.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download CSV report",
                                    data=csv,
                                    file_name=f"reporte_activities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime='text/csv',
                                    help="Descargar un reporte detallado de las activities importadas"
                                )

                                st.session_state['archivo_validado'] = False
                                st.session_state['resultado_validacion'] = None
                                st.session_state['archivo_cargado'] = None
                                st.session_state['training_import_raw_df'] = None
                                st.success("✅ Import completed successfully!")

                            except Exception as e:
                                st.error(f"❌ Error saving import: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                    
                    with col2:
                        if st.button("❌ Cancel", key="cancelar_importacion"):
                            st.session_state['archivo_validado'] = False
                            st.session_state['resultado_validacion'] = None
                            st.session_state['archivo_cargado'] = None
                            st.session_state['training_import_raw_df'] = None
                            st.rerun()
                
                # Mostrar mensaje de error si la validación falló
                elif 'resultado_validacion' in st.session_state and not st.session_state.get('archivo_validado', True):
                    error_msg = st.session_state['resultado_validacion']
                    if isinstance(error_msg, str):
                        st.error(f"❌ {error_msg}")
                    elif hasattr(error_msg, 'message') and isinstance(error_msg.message, str):
                        st.error(f"❌ {error_msg.message}")
                    else:
                        st.error("❌ File validation error")
                    
                    # Mostrar consejos para solucionar el problema
                    st.markdown("### How to fix the issue")
                    st.markdown("""
                    1. **Check the file format**: Make sure the file has at least 5 columns.
                    2. **Review dates**: Dates must be in a recognizable format (e.g., DD/MM/YYYY).
                    3. **Check player names**: The second column must contain player names.
                    4. **Make sure there is data**: At least one of columns 3-5 must contain information.
                    5. **Download the sample template** if you need a reference.
                    
                    - Must be an Excel file (.xlsx or .xls)
                    - It must contain columns: 'Month', 'Player', 'Individual Training', 'Meeting', 'Review Clips'
                    - The 'Month' column must contain valid dates
                    - There must be at least one player and one recorded activity
                    """)
                
                # Manejo de errores general
                if 'error_importacion' in st.session_state:
                    st.error(f"An error occurred while processing the file: {st.session_state['error_importacion']}")
                    del st.session_state['error_importacion']
