import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image
from pathlib import Path
import io
from utils.pdf_generator import generate_individual_development_report_landscape
from utils.sheets_client import GoogleSheetsClient


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

@st.cache_resource(show_spinner=False)
def get_sheets_client() -> GoogleSheetsClient:
    return GoogleSheetsClient()

def _read_local_training_file() -> pd.DataFrame:
    if not LOCAL_TRAINING_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(LOCAL_TRAINING_FILE, header=1)
    except Exception:
        return pd.read_excel(LOCAL_TRAINING_FILE, header=0)

def _load_sessions_raw_df() -> pd.DataFrame:
    sheets_client = get_sheets_client()
    if sheets_client.is_configured():
        try:
            return sheets_client.read_sessions_df()
        except Exception as exc:
            st.warning(f"No se pudo leer la pestaña 'Sesions' de Google Sheets. Usando fallback local. ({exc})")
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
            st.warning(f"No se pudo guardar en Google Sheets (Sesions). Guardando en local. ({exc})")

    LOCAL_TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(LOCAL_TRAINING_FILE, engine="openpyxl") as writer:
        df_to_save.to_excel(writer, index=False)
    return str(LOCAL_TRAINING_FILE)

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
            st.error(f"Error al convertir fechas: {e}")
            return pd.DataFrame()
        
        # Extraer año, mes y año-mes para agrupaciones
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
        
        # Limpiar cadenas en las columnas de texto
        text_columns = ['Player', 'Training_Type', 'Meeting', 'Review_Clips']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Reemplazar 'nan' con NaN
                df[col] = df[col].replace('nan', np.nan)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos de entrenamiento: {e}")
        import traceback
        st.error(f"Detalles del error: {traceback.format_exc()}")
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
                'total_jugadores': 1,  # Asumimos que hay al menos un jugador
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
        
        # Contar actividades
        entrenamientos = filtered_data['Training_Type'].count()
        meetings = filtered_data['Meeting'].count()
        review_clips = filtered_data['Review_Clips'].count()
        
        # Contar jugadores únicos con actividades
        jugadores_activos = filtered_data['Player'].nunique()
        total_jugadores = 1  # Asumimos que hay al menos un jugador
        
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
        st.error(f"Error al cargar métricas: {e}")
        return {
            'entrenamientos': 0,
            'meetings': 0,
            'review_clips': 0,
            'jugadores_activos': 0,
            'total_jugadores': 1,
            'porcentaje_participacion': 0
        }

@st.cache_data(ttl=3600)  # Cachear por 1 hora
def get_monthly_summary(start_date=None, end_date=None, months=6):
    """Obtener resumen mensual de actividades para un rango de fechas"""
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
        
        # Agrupar por mes y contar actividades
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
        st.error(f"Error al cargar resumen mensual: {e}")
        return pd.DataFrame(columns=['mes', 'entrenamientos', 'meetings', 'review_clips', 'mes_formateado'])

@st.cache_data(ttl=3600)  # Cachear por 1 hora
def get_players_summary(start_date=None, end_date=None):
    """Obtener resumen de actividades por jugador para un rango de fechas"""
    try:
        # Cargar datos de entrenamiento
        df = load_training_data()
        if df.empty:
            return pd.DataFrame(columns=['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_actividades'])
        
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
            return pd.DataFrame(columns=['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_actividades'])
        
        # Agrupar por jugador y contar actividades
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
        
        # Calcular total de actividades
        player_summary['total_actividades'] = (
            player_summary['entrenamientos'] + 
            player_summary['meetings'] + 
            player_summary['review_clips']
        )
        
        # Ordenar por total de actividades (descendente)
        player_summary = player_summary.sort_values('total_actividades', ascending=False)
        
        return player_summary
    except Exception as e:
        st.error(f"Error al cargar resumen por jugador: {e}")
        return pd.DataFrame(columns=['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_actividades'])

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
        "Seleccione una sección:",
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
    with st.spinner('Cargando datos del dashboard...'):
        
        # Obtener métricas para el rango de fechas seleccionado
        metrics = get_current_month_metrics(fecha_inicio, fecha_fin)
        
        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Entrenamientos</div>
                <div class='metric-value'>{metrics['entrenamientos']}</div>
                <div class='metric-delta'>+2 vs mes anterior</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Meetings</div>
                <div class='metric-value'>{metrics['meetings']}</div>
                <div class='metric-delta'>+1 vs mes anterior</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Review Clips</div>
                <div class='metric-value'>{metrics['review_clips']}</div>
                <div class='metric-delta'>{'-1' if metrics['review_clips'] > 0 else '0'} vs mes anterior</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Participación</div>
                <div class='metric-value'>{metrics['jugadores_activos']}/{metrics['total_jugadores']}</div>
                <div class='metric-delta'>{metrics['porcentaje_participacion']}% de jugadores</div>
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
                name='Entrenamientos',
                marker_color='#fcec03',  # Amarillo Watford
                hovertemplate='%{y} entrenamientos<extra></extra>'
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
                xaxis_title='Mes',
                yaxis_title='Cantidad de Actividades',
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
            st.warning("No hay datos disponibles para mostrar el gráfico de evolución.")
        
        # Resumen por jugador
        st.subheader(f"Summary All Players - ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
        
        # Obtener resumen por jugador para el rango de fechas seleccionado
        players_summary = get_players_summary(fecha_inicio, fecha_fin)
        
        # Mostrar tabla con resumen por jugador
        if not players_summary.empty:
            st.dataframe(
                players_summary[['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_actividades']],
                column_config={
                    "jugador": "Jugador",
                    "entrenamientos": st.column_config.NumberColumn("Entrenamientos", format="%d"),
                    "meetings": st.column_config.NumberColumn("Meetings", format="%d"),
                    "review_clips": st.column_config.NumberColumn("Review Clips", format="%d"),
                    "total_actividades": st.column_config.NumberColumn(
                        "Total Actividades", 
                        format="%d",
                        help="Suma de todas las actividades del jugador"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Gráfico de evolución por jugador
            
            # Obtener datos para el gráfico
            df_actividades = load_training_data()
            
            # Filtrar por rango de fechas
            date_mask = (df_actividades['Date'].dt.date >= fecha_inicio) & (df_actividades['Date'].dt.date <= fecha_fin)
            df_actividades = df_actividades[date_mask].copy()
            
            if not df_actividades.empty:
                # Crear una columna para el mes-año
                df_actividades['mes_anio'] = df_actividades['Date'].dt.to_period('M').astype(str)
                
                # Agrupar por jugador y mes para contar actividades
                df_agrupado = df_actividades.groupby(['Player', 'mes_anio']).agg({
                    'Training_Type': 'count',
                    'Meeting': lambda x: x.notna().sum(),
                    'Review_Clips': lambda x: x.notna().sum()
                }).reset_index()
                
                # Renombrar columnas
                df_agrupado = df_agrupado.rename(columns={
                    'Player': 'Jugador',
                    'mes_anio': 'Mes',
                    'Training_Type': 'Entrenamientos',
                    'Meeting': 'Meetings',
                    'Review_Clips': 'Review_Clips'
                })
                
                # Ordenar por mes
                df_agrupado = df_agrupado.sort_values('Mes')
                
                # Crear pestañas para cada tipo de actividad
                tab1, tab2, tab3 = st.tabs(["Entrenamientos", "Meetings", "Review Clips"])
                
                with tab1:
                    fig_entrenamientos = px.line(
                        df_agrupado, 
                        x='Mes', 
                        y='Entrenamientos',
                        color='Jugador',
                        title='Evolución de Entrenamientos por Jugador',
                        labels={'Entrenamientos': 'Cantidad', 'Mes': 'Mes'},
                        markers=True
                    )
                    fig_entrenamientos.update_layout(
                        xaxis_title='Mes',
                        yaxis_title='Cantidad de Entrenamientos',
                        legend_title='Jugador',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_entrenamientos, use_container_width=True)
                
                with tab2:
                    fig_meetings = px.line(
                        df_agrupado, 
                        x='Mes', 
                        y='Meetings',
                        color='Jugador',
                        title='Evolución de Meetings por Jugador',
                        labels={'Meetings': 'Cantidad', 'Mes': 'Mes'},
                        markers=True
                    )
                    fig_meetings.update_layout(
                        xaxis_title='Mes',
                        yaxis_title='Cantidad de Meetings',
                        legend_title='Jugador',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_meetings, use_container_width=True)
                
                with tab3:
                    fig_clips = px.line(
                        df_agrupado, 
                        x='Mes', 
                        y='Review_Clips',
                        color='Jugador',
                        title='Evolución de Review Clips por Jugador',
                        labels={'Review_Clips': 'Cantidad', 'Mes': 'Mes'},
                        markers=True
                    )
                    fig_clips.update_layout(
                        xaxis_title='Mes',
                        yaxis_title='Cantidad de Review Clips',
                        legend_title='Jugador',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_clips, use_container_width=True)
            else:
                st.warning("No hay datos de actividades para mostrar en el rango de fechas seleccionado.")
        else:
            st.info("No hay datos de actividades para mostrar en el período seleccionado.")
            
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
            st.error(f"Error al cargar jugadores activos: {e}")
            return {}
    
    # Obtener actividades del jugador
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
            actividades = df[
                (df['Player'] == jugador_nombre) &
                (df['Date'] >= fecha_inicio_dt) &
                (df['Date'] <= fecha_fin_dt)
            ].copy()
            
            if actividades.empty:
                return pd.DataFrame()
            
            # Crear un DataFrame con las actividades en formato largo
            actividades_largas = []
            
            # Procesar entrenamientos
            entrenamientos = actividades[actividades['Training_Type'].notna()]
            for _, row in entrenamientos.iterrows():
                actividades_largas.append({
                    'fecha': row['Date'],
                    'tipo': 'Entrenamiento',
                    'subtipo': row['Training_Type'],
                    'descripcion': f"Entrenamiento: {row['Training_Type']}",
                    'jugador': row['Player']
                })
            
            # Procesar meetings
            meetings = actividades[actividades['Meeting'].notna()]
            for _, row in meetings.iterrows():
                actividades_largas.append({
                    'fecha': row['Date'],
                    'tipo': 'Meeting',
                    'subtipo': row['Meeting'],
                    'descripcion': f"Meeting: {row['Meeting']}",
                    'jugador': row['Player']
                })
            
            # Procesar review clips
            reviews = actividades[actividades['Review_Clips'].notna()]
            for _, row in reviews.iterrows():
                actividades_largas.append({
                    'fecha': row['Date'],
                    'tipo': 'Review Clip',
                    'subtipo': 'Revisión de video',
                    'descripcion': f"Review Clip: {row['Review_Clips']}",
                    'jugador': row['Player']
                })
            
            # Crear DataFrame con todas las actividades
            if actividades_largas:
                return pd.DataFrame(actividades_largas).sort_values('fecha', ascending=False)
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error al cargar actividades del jugador: {e}")
            return pd.DataFrame()
    
    # Cargar jugadores activos
    jugadores = load_active_players()
    
    if not jugadores:
        st.warning("No se encontraron jugadores activos en los datos de entrenamiento.")
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
            default=[list(jugadores.keys())[0]]  # Por defecto el primer jugador
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
        Entrenamientos=pd.NamedAgg(column="Training_Type", aggfunc=lambda x: x.notna().sum()),
        Meetings=pd.NamedAgg(column="Meeting", aggfunc=lambda x: x.notna().sum()),
        Review_Clips=pd.NamedAgg(column="Review_Clips", aggfunc=lambda x: x.notna().sum())
    ).reset_index()

    # Add total column
    summary["Total Actividades"] = summary[["Entrenamientos", "Meetings", "Review_Clips"]].sum(axis=1)

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

    # Entrenamientos
    comparison_fig.add_trace(go.Bar(
        x=summary["Player"],
        y=summary["Entrenamientos"],
        name="Entrenamientos",
        marker_color=[
            "#fcec03" if player == jugador_seleccionado else "#d3d3d3"
            for player in summary["Player"]
        ],
        hovertemplate="%{y} entrenamientos<extra></extra>"
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
        xaxis_title="Jugador",
        yaxis_title="Cantidad de Actividades",
        legend_title="Tipo",
        plot_bgcolor="white",
        title="Actividades por Jugador",
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
    with st.spinner('Cargando datos del jugador...'):
        df_actividades = load_player_activities(jugador_id, fecha_inicio, fecha_fin)
        
        jugador_info = next((v for k, v in jugadores.items() if k == jugador_id), "")
        
        st.subheader(f"Individual Activities - {jugador_info}")

        if not df_actividades.empty:
            tipos_disponibles = df_actividades["tipo"].unique().tolist()
            tipos_seleccionados = st.multiselect(
                "Select Activities",
                options=tipos_disponibles,
                default=tipos_disponibles,
                key="filtro_tipo_actividades"
            )

            df_filtrado = df_actividades[df_actividades["tipo"].isin(tipos_seleccionados)]

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
                        "fecha": "Fecha",
                        "tipo": "Tipo de Actividad",
                        "subtipo": "Detalle",
                        "descripcion": "Descripción"
                    }
                )

                # Botón para exportar las actividades filtradas a Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_mostrar.to_excel(writer, index=False, sheet_name="Actividades")

                buffer.seek(0)

                st.download_button(
                    label="📥 Exportar a Excel",
                    data=buffer,
                    file_name=f"individual_activities_{jugador_info.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No hay actividades del tipo seleccionado.")
                df_pdf_activities = df_filtrado.copy()
                selected_types_for_pdf = list(tipos_seleccionados)
        else:
            st.info("No hay actividades registradas para este jugador.")


        # Timeline de actividades
        st.subheader(f"Timeline de Actividades - {jugador_info}")
        
        if not df_actividades.empty:
            # Crear gráfico de dispersión para la timeline
            # Usar solo las columnas disponibles en el DataFrame
            hover_columns = []
            if 'descripcion' in df_actividades.columns:
                hover_columns.append('descripcion')
            if 'subtipo' in df_actividades.columns:
                hover_columns.append('subtipo')
                
            timeline_fig = px.scatter(
                df_actividades,
                x='fecha',
                y='tipo',
                color='tipo',
                color_discrete_map={
                    'Entrenamiento': '#fcec03',  # Amarillo Watford
                    'Meeting': '#1f77b4',        # Azul
                    'Review Clip': '#2ca02c'      # Verde
                },
                size=[1] * len(df_actividades),  # Tamaño fijo para todos los puntos
                hover_data=hover_columns if hover_columns else None,
                labels={'fecha': 'Fecha', 'tipo': 'Tipo de Actividad'},
                title=f"({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})"
            )
            
            timeline_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Fecha',
                yaxis_title='',
                showlegend=True,
                legend_title='Tipo de Actividad',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(timeline_fig, use_container_width=True)
            pdf_timeline_fig = timeline_fig

        if df_pdf_activities.empty and not df_actividades.empty:
            df_pdf_activities = df_actividades.copy()
            selected_types_for_pdf = sorted(df_pdf_activities["tipo"].dropna().astype(str).unique().tolist())

    st.markdown("---")
    st.subheader("PDF Report")

    report_signature = (
        f"{jugador_seleccionado}|{fecha_inicio}|{fecha_fin}|"
        f"{','.join(sorted(selected_types_for_pdf))}|{len(df_pdf_activities)}"
    )
    if st.session_state.get("individual_profile_pdf_signature") != report_signature:
        st.session_state["individual_profile_pdf_signature"] = report_signature
        st.session_state.pop("individual_profile_pdf_bytes", None)
        st.session_state.pop("individual_profile_pdf_name", None)

    if df_pdf_activities.empty:
        st.info("No hay actividades en el rango/filtros actuales para generar el PDF.")
    else:
        generating_key = "individual_profile_pdf_generating"
        if generating_key not in st.session_state:
            st.session_state[generating_key] = False

        if st.button(
            "Generate PDF report",
            key="generate_individual_profile_pdf",
            type="primary",
            disabled=st.session_state.get(generating_key, False),
        ):
            st.session_state[generating_key] = True
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
                    df_summary_pdf["tipo"] = df_summary_pdf["tipo"].fillna("Sin tipo")

                    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(jugador_info).strip())
                    safe_name = safe_name.strip("_") or "player"
                    pdf_file_name = f"{safe_name}_individual_development.pdf"

                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = generate_individual_development_report_landscape(
                            player_name=jugador_info,
                            fecha_inicio=fecha_inicio.strftime("%Y-%m-%d"),
                            fecha_fin=fecha_fin.strftime("%Y-%m-%d"),
                            df_actividades=df_pdf_activities,
                            df_summary=df_summary_pdf,
                            df_ratings=None,
                            fig_comparison=pdf_comparison_fig,
                            fig_timeline=pdf_timeline_fig,
                            logo_path=LOGO_PATH,
                            background_image_path=BACKGROUND_COVER_PATH if os.path.exists(BACKGROUND_COVER_PATH) else None,
                        )

                    if not pdf_bytes:
                        raise ValueError("Generated PDF is empty.")

                    st.session_state["individual_profile_pdf_bytes"] = pdf_bytes
                    st.session_state["individual_profile_pdf_name"] = pdf_file_name
                    st.success("PDF generated. Download it below.")
                except Exception as e:
                    st.error(f"Failed to generate PDF report: {e}")
            finally:
                st.session_state[generating_key] = False
                overlay_placeholder.empty()

        if st.session_state.get("individual_profile_pdf_bytes"):
            st.download_button(
                "Download PDF report",
                data=st.session_state["individual_profile_pdf_bytes"],
                file_name=st.session_state.get("individual_profile_pdf_name", "individual_development_report.pdf"),
                mime="application/pdf",
                key="download_individual_profile_pdf",
            )
            

# 3. Registro de Actividades
elif page == "Files":
    st.header("Files")
    st.write("Registra nuevas actividades de desarrollo individual para los jugadores.")
    
    
    # Descargar snapshot actual (desde Google Sheets / Sesions o fallback local)
    current_sessions_df = _load_sessions_raw_df()
    if not current_sessions_df.empty:
        download_buffer = io.BytesIO()
        with pd.ExcelWriter(download_buffer, engine="openpyxl") as writer:
            current_sessions_df.to_excel(writer, index=False, sheet_name="Sesions")
        download_buffer.seek(0)
        st.download_button(
            label="📥 Descargar datos actuales",
            data=download_buffer.getvalue(),
            file_name="Individuals-Training.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No hay datos actuales en la fuente de sesiones.")


    # Función para validar y cargar el archivo Excel específico
    def validar_importar_excel(file_path):
        try:
            print(f"Validando archivo: {file_path.name if hasattr(file_path, 'name') else file_path}")
            
            # Leer solo las primeras filas para verificar la estructura
            df_preview = pd.read_excel(file_path, nrows=5)
            print("Primeras filas del archivo:")
            print(df_preview.head())
            print("\nColumnas del archivo:", df_preview.columns.tolist())
            
            # Verificar si el archivo tiene datos
            if df_preview.empty:
                st.markdown("""
                <div style='background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;'>
                    El archivo está vacío. Por favor, verifica que el archivo contenga datos.
                </div>
                """, unsafe_allow_html=True)
                return False, "El archivo está vacío. Por favor, verifica que el archivo contenga datos."
            
            # Verificar si el archivo tiene el formato esperado
            expected_columns = ['Mes', 'Player', 'Individual Training', 'Meeting', 'Review Clips']
            
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
                    print(f"Error al leer el archivo con header={i}: {str(e)}")
                    continue
            
            if header_row is None:
                # Si no se encontraron los encabezados, mostrar un mensaje de error detallado
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                    <h4 style='margin-top: 0;'>❌ Error en el formato del archivo</h4>
                    <p>El archivo no tiene el formato esperado. Asegúrate de que el archivo contenga las siguientes columnas:</p>
                    <ul style='margin-bottom: 10px;'>
                        <li><strong>Mes</strong> (fecha de la actividad)</li>
                        <li><strong>Player</strong> (nombre del jugador)</li>
                        <li><strong>Individual Training</strong> (actividades de entrenamiento)</li>
                        <li><strong>Meeting</strong> (reuniones)</li>
                        <li><strong>Review Clips</strong> (revisión de videos)</li>
                    </ul>
                    <p style='margin-bottom: 5px;'><strong>Consejos:</strong></p>
                    <ul style='margin-top: 5px;'>
                        <li>Verifica que la primera fila contenga los encabezados</li>
                        <li>Los nombres de las columnas deben coincidir exactamente (pueden variar mayúsculas y espacios)</li>
                        <li>Las fechas deben estar en un formato reconocible (DD/MM/YYYY o YYYY-MM-DD)</li>
                    </ul>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "El archivo no tiene el formato esperado. Verifica que contenga las columnas requeridas."
            
            # Leer el archivo completo con el encabezado correcto
            df = pd.read_excel(file_path, header=header_row)
            
            # Limpiar los nombres de las columnas (eliminar espacios en blanco)
            df.columns = [str(col).strip() for col in df.columns]
            
            # Verificar que las columnas requeridas estén presentes
            required_columns = ['Mes', 'Player']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                error_msg = f"""
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Columnas requeridas faltantes</h4>
                    <p>Faltan las siguientes columnas requeridas: <strong>{', '.join(missing_columns)}</strong></p>
                    <p>Por favor, asegúrate de que el archivo contenga al menos las columnas 'Mes' y 'Player'.</p>
                    <p>Columnas encontradas en el archivo: {', '.join(df.columns) if len(df.columns) > 0 else 'Ninguna columna encontrada'}</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, f"Faltan columnas requeridas: {', '.join(missing_columns)}"
            
            # Verificar que haya al menos una columna de actividad
            activity_columns = [col for col in df.columns if col not in ['Mes', 'Player']]
            if not activity_columns:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ No se encontraron actividades</h4>
                    <p>No se encontraron columnas de actividades en el archivo.</p>
                    <p>Asegúrate de que el archivo contenga al menos una columna de actividad (Individual Training, Meeting o Review Clips).</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No se encontraron columnas de actividades en el archivo."
            
            # Verificar que haya al menos una fila con datos
            if df.empty:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Archivo vacío</h4>
                    <p>El archivo está vacío o no contiene datos válidos.</p>
                    <p>Por favor, verifica que el archivo tenga datos en las filas debajo de los encabezados.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "El archivo está vacío o no contiene datos válidos."
            
            # Verificar que las fechas sean válidas
            try:
                # Convertir la columna de fechas a datetime
                df['fecha_dt'] = pd.to_datetime(df['Mes'], errors='coerce')
                
                # Verificar si hay fechas inválidas
                if df['fecha_dt'].isna().any():
                    fechas_invalidas = df[df['fecha_dt'].isna()]['Mes'].head().tolist()
                    error_msg = f"""
                    <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                        <h4 style='margin-top: 0;'>❌ Fechas inválidas</h4>
                        <p>Se encontraron fechas inválidas en la columna 'Mes'.</p>
                        <p>Ejemplos de valores problemáticos: {', '.join(map(str, fechas_invalidas))}</p>
                        <p>Asegúrate de que las fechas estén en un formato reconocible (ej: DD/MM/YYYY o YYYY-MM-DD).</p>
                    </div>
                    """
                    st.markdown(error_msg, unsafe_allow_html=True)
                    return False, f"Se encontraron fechas inválidas: {', '.join(map(str, fechas_invalidas[:3]))}..."
                
                # Verificar que las fechas estén en un rango razonable
                fecha_min = pd.Timestamp('2020-01-01')
                fecha_max = pd.Timestamp.now() + pd.DateOffset(years=1)
                
                fechas_fuera_de_rango = df[(df['fecha_dt'] < fecha_min) | (df['fecha_dt'] > fecha_max)]
                if not fechas_fuera_de_rango.empty:
                    ejemplos = fechas_fuera_de_rango['fecha_dt'].dt.strftime('%Y-%m-%d').head().tolist()
                    error_msg = f"""
                    <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                        <h4 style='margin-top: 0;'>❌ Fechas fuera de rango</h4>
                        <p>Algunas fechas están fuera del rango permitido (01/01/2020 - {fecha_max.strftime('%d/%m/%Y')}).</p>
                        <p>Fechas problemáticas: {', '.join(ejemplos)}</p>
                        <p>Por favor, verifica que todas las fechas estén dentro del rango permitido.</p>
                    </div>
                    """
                    st.markdown(error_msg, unsafe_allow_html=True)
                    return False, f"Fechas fuera de rango: {', '.join(ejemplos[:3])}..."
                
                # Formatear las fechas
                df['fecha_formateada'] = df['fecha_dt'].dt.strftime('%Y-%m-%d')
                
            except Exception as e:
                error_msg = f"""
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Error al procesar las fechas</h4>
                    <p>Ocurrió un error al procesar las fechas en la columna 'Mes'.</p>
                    <p>Error: {str(e)}</p>
                    <p>Por favor, verifica que todas las fechas estén en un formato válido y vuelve a intentarlo.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, f"Error al procesar las fechas: {str(e)}"
            
            # Verificar que haya al menos un jugador con nombre válido
            jugadores_invalidos = df[df['Player'].isna() | (df['Player'].astype(str).str.strip() == '')]
            if len(jugadores_invalidos) == len(df):
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ Nombres de jugadores inválidos</h4>
                    <p>No se encontraron nombres de jugadores válidos en la columna 'Player'.</p>
                    <p>Por favor, asegúrate de que la columna 'Player' contenga los nombres de los jugadores.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No se encontraron nombres de jugadores válidos en la columna 'Player'."
            
            # Verificar que haya al menos una actividad registrada
            actividades = []
            for _, fila in df.iterrows():
                # Saltar filas sin jugador o con jugador vacío
                if pd.isna(fila.get('Player')) or str(fila.get('Player', '')).strip() == '':
                    continue
                    
                for col in activity_columns:
                    if pd.notna(fila.get(col)) and str(fila[col]).strip() != '':
                        actividades.append({
                            'fecha': fila['fecha_formateada'],
                            'jugador': str(fila['Player']).strip(),
                            'tipo': col,
                            'descripcion': str(fila[col]).strip()
                        })
            
            if not actividades:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ No hay actividades válidas</h4>
                    <p>No se encontraron actividades válidas en el archivo.</p>
                    <p>Asegúrate de que al menos una celda en las columnas de actividades contenga datos.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No se encontraron actividades válidas en el archivo."
            
            # Crear el DataFrame final con las actividades
            df_actividades = pd.DataFrame(actividades)
            
            # Verificar que tengamos al menos una actividad válida
            if df_actividades.empty:
                error_msg = """
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                    <h4 style='margin-top: 0;'>❌ No se pudieron procesar las actividades</h4>
                    <p>No se encontraron actividades válidas para importar.</p>
                    <p>Por favor, verifica que el archivo tenga el formato correcto e inténtalo de nuevo.</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
                return False, "No se pudieron procesar las actividades. Verifica el formato del archivo."
            
            # Guardar dataset tabular validado para subirlo a la pestaña Sesions.
            st.session_state["training_import_raw_df"] = df.copy()
            return True, df_actividades
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error al procesar el archivo: {error_details}")
            
            # Mensaje de error más amigable
            mensaje_error = f"""
            <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;'>
                <h4 style='margin-top: 0;'>❌ Error al procesar el archivo</h4>
                <p>Ocurrió un error al intentar procesar el archivo. Por favor, verifica que el archivo cumpla con el formato esperado.</p>
                <p><strong>Detalles del error:</strong> {str(e)}</p>
                <p>Si el problema persiste, intenta:</p>
                <ul>
                    <li>Descargar la plantilla de ejemplo y usar ese formato</li>
                    <li>Verificar que el archivo no esté dañado</li>
                    <li>Comprobar que no tenga fórmulas o formatos especiales</li>
                </ul>
            </div>
            """
            return False, mensaje_error
    
    # Sección para importar datos desde Excel
    with st.expander("📤 Importar Datos desde Excel", expanded=False):
        st.markdown("### Instrucciones para la importación")
        st.markdown("""
        1. **Formato del archivo**: El archivo Excel debe tener la siguiente estructura:
           - **Columna 1 (A)**: Fecha de la actividad (formato de fecha reconocible)
           - **Columna 2 (B)**: Nombre del jugador
           - **Columna 3 (C)**: Detalles del entrenamiento individual (opcional)
           - **Columna 4 (D)**: Detalles de la reunión (opcional)
           - **Columna 5 (E)**: Detalles de la revisión de videos (opcional)
        
        2. **Requisitos**:
           - El archivo debe estar en formato .xlsx
           - La primera fila debe contener los encabezados
           - Al menos una actividad debe estar registrada en las columnas 3-5
           - Las fechas deben estar entre 2020 y el año siguiente al actual
        
        3. **Consejos**:
           - Asegúrate de que los nombres de los jugadores sean consistentes
           - Revisa que las fechas tengan el formato correcto
           - Verifica que al menos una celda de actividad contenga información
        """)
        
        st.markdown("---")
        st.markdown("### Cargar archivo Excel")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo Excel (.xlsx)", 
            type=["xlsx"],
            help="Haz clic o arrastra un archivo Excel con el formato especificado"
        )
        
        if uploaded_file is not None:
            # Mostrar información del archivo
            file_name = uploaded_file.name
            file_size = len(uploaded_file.getvalue()) / 1024  # Tamaño en KB
            
            # Mostrar información del archivo en un contenedor con estilo
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Archivo", file_name)
                with col2:
                    st.metric("Tamaño", f"{file_size:.1f} KB")
                
                # Botón para validar el archivo
                st.markdown("### Validar y Procesar")
                
                if st.button("🔍 Validar Archivo", key="validar_archivo", help="Validar la estructura y los datos del archivo"):
                    with st.spinner("Validando archivo, por favor espere..."):
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
                            st.error(f"❌ Error inesperado al validar el archivo: {str(e)}")
                            st.session_state['archivo_validado'] = False
                            st.session_state['resultado_validacion'] = str(e)
                            st.session_state['training_import_raw_df'] = None
                
                # Si ya se validó el archivo, mostrar los resultados
                if st.session_state.get('archivo_validado', False):
                    resultado = st.session_state['resultado_validacion']
                    st.success("✅ Validación exitosa")
                    
                    # Mostrar estadísticas de las actividades a importar
                    st.markdown("#### Resumen de Importación")
                    
                    # Calcular estadísticas
                    total_actividades = len(resultado)
                    jugadores_unicos = resultado['jugador'].nunique()
                    fechas_unicas = resultado['fecha'].nunique()
                    actividades_por_tipo = resultado['tipo'].value_counts().to_dict()
                    
                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Actividades", total_actividades)
                    with col2:
                        st.metric("👥 Jugadores Únicos", jugadores_unicos)
                    with col3:
                        st.metric("📅 Días con Actividades", fechas_unicas)
                    
                    # Mostrar distribución por tipo de actividad
                    st.markdown("#### Distribución por Tipo de Actividad")
                    for tipo, cantidad in actividades_por_tipo.items():
                        st.progress(cantidad / total_actividades, f"{tipo}: {cantidad} actividades")
                    
                    # Vista previa de los datos
                    st.markdown("#### Vista Previa de los Datos")
                    st.dataframe(resultado.head(10), 
                                use_container_width=True,
                                column_config={
                                    'fecha': 'Fecha',
                                    'jugador': 'Jugador',
                                    'tipo': 'Tipo de Actividad',
                                    'descripcion': 'Descripción'
                                })
                    
                    # Botones de confirmación y cancelación
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirmar Importación", type="primary", key="confirmar_importacion"):
                            try:
                                st.info("Procesando la importación, por favor espere...")
                                raw_df_to_save = st.session_state.get("training_import_raw_df")

                                if raw_df_to_save is None or raw_df_to_save.empty:
                                    uploaded_file = st.session_state.get("archivo_cargado")
                                    if uploaded_file is None:
                                        raise ValueError("No hay archivo validado para importar.")
                                    uploaded_file.seek(0)
                                    raw_df_to_save = pd.read_excel(uploaded_file)

                                destination = _save_sessions_raw_df(raw_df_to_save)
                                st.success(f"✅ ¡Datos guardados exitosamente en {destination}!")

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
                                    label="📥 Descargar Reporte en CSV",
                                    data=csv,
                                    file_name=f"reporte_actividades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime='text/csv',
                                    help="Descargar un reporte detallado de las actividades importadas"
                                )

                                st.session_state['archivo_validado'] = False
                                st.session_state['resultado_validacion'] = None
                                st.session_state['archivo_cargado'] = None
                                st.session_state['training_import_raw_df'] = None
                                st.success("✅ ¡Importación completada con éxito!")

                            except Exception as e:
                                st.error(f"❌ Error al guardar la importación: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                    
                    with col2:
                        if st.button("❌ Cancelar", key="cancelar_importacion"):
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
                        st.error("❌ Error en la validación del archivo")
                    
                    # Mostrar consejos para solucionar el problema
                    st.markdown("### ¿Cómo solucionar el problema?")
                    st.markdown("""
                    1. **Verifica el formato del archivo**: Asegúrate de que el archivo tenga al menos 5 columnas.
                    2. **Revisa las fechas**: Las fechas deben estar en un formato reconocible (ej: DD/MM/YYYY).
                    3. **Comprueba los nombres de los jugadores**: La segunda columna debe contener nombres de jugadores.
                    4. **Asegúrate de que haya datos**: Al menos una de las columnas 3-5 debe contener información.
                    5. **Descarga la plantilla de ejemplo** si necesitas una referencia.
                    
                    - Debe ser un archivo Excel (.xlsx o .xls)
                    - Debe contener las columnas: 'Mes', 'Player', 'Individual Training', 'Meeting', 'Review Clips'
                    - La columna 'Mes' debe contener fechas válidas
                    - Debe haber al menos un jugador y una actividad registrada
                    """)
                
                # Manejo de errores general
                if 'error_importacion' in st.session_state:
                    st.error(f"Ocurrió un error al procesar el archivo: {st.session_state['error_importacion']}")
                    del st.session_state['error_importacion']
