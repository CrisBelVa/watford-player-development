import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image
from pathlib import Path

# --- Configuración de página ---
# Obtener la ruta absoluta al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')

# Configuración de la página
st.set_page_config(
    page_title="Watford - Individual Development",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
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
    """Carga y preprocesa los datos de entrenamiento desde el archivo Excel."""
    try:
        # Ruta al archivo de entrenamiento
        data_dir = Path(BASE_DIR) / 'data'
        file_path = data_dir / 'Individuals - Training.xlsx'
        
        # Cargar el archivo Excel omitiendo la primera fila (nombre del jugador)
        df = pd.read_excel(file_path, header=1)
        
        # Renombrar columnas para mejor manejo
        df.columns = [str(col).strip() for col in df.columns]
        
        # Verificar que las columnas esperadas existen
        if len(df.columns) < 5:
            st.error("El archivo Excel no tiene el formato esperado. Se esperan al menos 5 columnas.")
            return pd.DataFrame()
        
        # Renombrar columnas a nombres más manejables
        df = df.rename(columns={
            df.columns[0]: 'Date',
            df.columns[1]: 'Player',
            df.columns[2]: 'Training_Type',
            df.columns[3]: 'Meeting',
            df.columns[4]: 'Review_Clips'
        })
        
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
    # Logo y título
    if logo:
        st.image(logo, width=100)
    st.title("Desarrollo Individual")
    
    # Navegación
    st.markdown("### Navegación")
    page = st.radio(
        "Seleccione una sección:",
        ["📊 Dashboard General", "👤 Perfil Individual", "📝 Registro Actividades", "📈 Reportes"],
        label_visibility="collapsed"
    )
    
    # Información del usuario
    st.markdown("---")
    st.markdown("### Información del Usuario")
    if "staff_info" in st.session_state:
        st.write(f"**Nombre:** {st.session_state.staff_info['full_name']}")
        st.write(f"**Rol:** {st.session_state.staff_info['role']}")
    
    if st.button("Cerrar Sesión", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Contenido principal ---

# 1. Dashboard General
if page == "📊 Dashboard General":
    st.header("📊 Dashboard General")
    
    # Mostrar indicador de carga mientras se obtienen los datos
    with st.spinner('Cargando datos del dashboard...'):
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio", datetime.now() - timedelta(days=30))
        with col2:
            fecha_fin = st.date_input("Fecha de fin", datetime.now())
        
        # Métricas
        st.subheader(f"Métricas del Período: {fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')}")
        
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
        st.subheader(f"Evolución de Actividades ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
        
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
        st.subheader(f"Resumen por Jugador ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
        
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
            st.subheader(f"Evolución de Actividades por Jugador")
            
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
                tab1, tab2, tab3 = st.tabs(["📊 Entrenamientos", "💬 Meetings", "🎥 Review Clips"])
                
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
        **Notas:**
        - Los datos se actualizan automáticamente cada hora.
        - Para ver información más detallada, utilice la sección de Reportes.
        - Contacte con el administrador si necesita ayuda o detecta algún error.
        """)

# 2. Perfil Individual
elif page == "👤 Perfil Individual":
    st.header("👤 Perfil Individual")
    
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
            
            # Filtrar por jugador y rango de fechas
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filtrar actividades del jugador en el rango de fechas
            actividades = df[
                (df['Player'] == jugador_nombre) &
                (df['Date'] >= start_date) &
                (df['Date'] <= end_date)
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
    
    # Selector de jugador
    jugador_id = st.selectbox(
        "Seleccionar Jugador",
        options=list(jugadores.keys()),
        format_func=lambda x: jugadores[x],
        index=0
    )
    jugador_seleccionado = jugadores[jugador_id]
    
    # Filtros de fecha (últimos 90 días por defecto)
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=90)
    
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input("Fecha de inicio", fecha_inicio)
    with col2:
        fecha_fin = st.date_input("Fecha de fin", fecha_fin)
    
    # Validar fechas
    if fecha_inicio > fecha_fin:
        st.error("La fecha de inicio no puede ser posterior a la fecha de fin")
        st.stop()
    
    # Cargar actividades del jugador
    with st.spinner('Cargando datos del jugador...'):
        df_actividades = load_player_activities(jugador_id, fecha_inicio, fecha_fin)
        
        # Obtener información del jugador seleccionado
        jugador_info = next((v for k, v in jugadores.items() if k == jugador_id), "")
        
        # Mostrar información del jugador
        st.subheader("Información del Jugador")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre", jugador_info)
        with col2:
            # Obtener posición del jugador (asumiendo que está en la base de datos)
            st.metric("Posición", "No especificada")  # Se puede mejorar con datos reales
        with col3:
            st.metric("Equipo", "Watford FC")  # Se puede mejorar con datos reales
        
        # Calcular métricas
        total_entrenamientos = 0
        total_meetings = 0
        total_review_clips = 0
        
        if not df_actividades.empty:
            total_entrenamientos = len(df_actividades[df_actividades['tipo'] == 'Entrenamiento'])
            total_meetings = len(df_actividades[df_actividades['tipo'] == 'Meeting'])
            total_review_clips = len(df_actividades[df_actividades['tipo'] == 'Review Clip'])
        
        total_actividades = total_entrenamientos + total_meetings + total_review_clips
        
        # Mostrar métricas
        st.subheader("Métricas del Período")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Entrenamientos", total_entrenamientos)
        with col2:
            st.metric("Meetings", total_meetings)
        with col3:
            st.metric("Review Clips", total_review_clips)
        with col4:
            st.metric("Total Actividades", total_actividades)
        
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
                
            fig = px.scatter(
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
                title=f"Actividades de {jugador_info} ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Fecha',
                yaxis_title='',
                showlegend=True,
                legend_title='Tipo de Actividad',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Lista de actividades con filtros
            st.subheader("Lista de Actividades")
            
            # Filtro por tipo de actividad
            tipos_disponibles = df_actividades['tipo'].unique().tolist()
            tipos_seleccionados = st.multiselect(
                "Filtrar por tipo de actividad",
                options=tipos_disponibles,
                default=tipos_disponibles,
                key=f"filtro_tipo_{jugador_id}"
            )
            
            # Aplicar filtros
            actividades_filtradas = df_actividades[df_actividades['tipo'].isin(tipos_seleccionados)]
            
            if not actividades_filtradas.empty:
                # Ordenar por fecha descendente
                actividades_filtradas = actividades_filtradas.sort_values('fecha', ascending=False)
                
                # Mostrar cada actividad en un expander
                for _, actividad in actividades_filtradas.iterrows():
                    # Crear título del expander con información disponible
                    titulo_expander = f"{actividad['tipo']} - {actividad['fecha'].strftime('%d/%m/%Y %H:%M')}"
                    
                    # Agregar información adicional si está disponible
                    if 'subtipo' in actividad and pd.notna(actividad['subtipo']):
                        titulo_expander += f" - {actividad['subtipo']}"
                    elif 'descripcion' in actividad and pd.notna(actividad['descripcion']) and len(actividad['descripcion']) > 0:
                        # Usar los primeros 30 caracteres de la descripción si no hay subtítulo
                        desc = (actividad['descripcion'][:30] + '...') if len(actividad['descripcion']) > 30 else actividad['descripcion']
                        titulo_expander += f" - {desc}"
                    
                    with st.expander(titulo_expander):
                        # Mostrar información específica según el tipo de actividad
                        st.write(f"**Tipo:** {actividad['tipo']}")
                        st.write(f"**Fecha y hora:** {actividad['fecha'].strftime('%d/%m/%Y %H:%M')}")
                        
                        if 'descripcion' in actividad and pd.notna(actividad['descripcion']):
                            st.write(f"**Descripción:** {actividad['descripcion']}")
                            
                        if actividad['tipo'] == 'Entrenamiento':
                            st.write(f"**Duración:** {actividad.get('duracion_minutos', 'N/A')} minutos")
                            st.write(f"**Objetivo:** {actividad.get('objetivo', 'No especificado')}")
                            st.write(f"**Resultado:** {actividad.get('resultado', 'No especificado')}")
                        elif actividad['tipo'] == 'Meeting':
                            st.write(f"**Tipo de reunión:** {actividad.get('tipo_reunion', 'No especificado')}")
                            st.write(f"**Duración:** {actividad.get('duracion_minutos', 'N/A')} minutos")
                        elif actividad['tipo'] == 'Review Clip':
                            st.write(f"**Duración:** {actividad.get('duracion_segundos', 'N/A')} segundos")
                            if 'enlace_video' in actividad and actividad['enlace_video']:
                                st.write(f"**Enlace:** [{actividad['enlace_video']}]({actividad['enlace_video']})")
                        
                        # Mostrar notas si existen
                        if 'notas' in actividad and actividad['notas'] and pd.notna(actividad['notas']):
                            st.markdown("---")
                            st.write("**Notas:**")
                            st.write(actividad['notas'])
            else:
                st.info("No hay actividades que coincidan con los filtros seleccionados.")
        else:
            st.info("No hay actividades registradas para este jugador en el período seleccionado.")

# 3. Registro de Actividades
elif page == "📝 Registro Actividades":
    st.header("📝 Registro de Actividades")
    st.write("Registra nuevas actividades de desarrollo individual para los jugadores.")
    
    
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
    with st.expander("📤 Importar Datos desde Excel", expanded=True):
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
                            
                            # Si hay un error de validación, mostrarlo
                            if not es_valido and isinstance(resultado, str):
                                st.error(f"❌ {resultado}")
                            
                            # No hacemos rerun aquí para evitar problemas con el estado
                        except Exception as e:
                            st.error(f"❌ Error inesperado al validar el archivo: {str(e)}")
                            st.session_state['archivo_validado'] = False
                            st.session_state['resultado_validacion'] = str(e)
                
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
                                import os
                                import shutil
                                from datetime import datetime
                                
                                # Obtener los datos de la sesión
                                uploaded_file = st.session_state['archivo_cargado']
                                
                                st.info("Procesando la importación, por favor espere...")
                                
                                # Ruta al directorio data (un nivel arriba del directorio actual)
                                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
                                os.makedirs(data_dir, exist_ok=True)
                                
                                # Ruta completa del archivo de destino
                                file_path = os.path.join(data_dir, 'Individuals - Training.xlsx')
                                
                                # Crear archivo temporal primero
                                temp_path = os.path.join(data_dir, 'temp_training.xlsx')
                                
                                # Reiniciar el puntero del archivo uploaded
                                uploaded_file.seek(0)
                                
                                # Guardar primero en archivo temporal
                                with open(temp_path, 'wb') as f:
                                    f.write(uploaded_file.getvalue())
                                
                                # Verificar que el archivo temporal se guardó correctamente
                                if not os.path.exists(temp_path):
                                    st.error("❌ Error: No se pudo crear el archivo temporal.")
                                    raise Exception("No se pudo crear el archivo temporal")
                                
                                # Si existe el archivo original, crear backup
                                backup_path = None
                                if os.path.exists(file_path):
                                    backup_path = os.path.join(data_dir, f'backup_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
                                    shutil.copy2(file_path, backup_path)
                                    st.info(f"✓ Backup creado: {os.path.basename(backup_path)}")
                                
                                # Mover el archivo temporal al destino final
                                shutil.move(temp_path, file_path)
                                
                                # Verificar que el archivo se movió correctamente
                                if os.path.exists(file_path):
                                    file_size = os.path.getsize(file_path) / 1024  # Tamaño en KB
                                    st.success(f"✅ ¡Archivo guardado exitosamente!")
                                    st.success(f"   📁 Ubicación: {file_path}")
                                    st.success(f"   📊 Tamaño: {file_size:.1f} KB")
                                    
                                    # Limpiar el caché de Streamlit
                                    st.cache_data.clear()
                                    
                                    # Limpiar cachés específicos
                                    if 'load_training_data' in globals():
                                        load_training_data.clear()
                                    if 'get_current_month_metrics' in globals():
                                        get_current_month_metrics.clear()
                                    if 'get_monthly_summary' in globals():
                                        get_monthly_summary.clear()
                                    if 'get_players_summary' in globals():
                                        get_players_summary.clear()
                                    
                                    st.success("✅ ¡Caché actualizado! Los nuevos datos estarán disponibles.")
                                    
                                    # Mostrar opción para descargar un reporte
                                    csv = resultado.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Descargar Reporte en CSV",
                                        data=csv,
                                        file_name=f"reporte_actividades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime='text/csv',
                                        help="Descargar un reporte detallado de las actividades importadas"
                                    )
                                    
                                    # Mensaje para refrescar la página
                                    st.info("💡 **Sugerencia:** Refresca la página o navega a otra sección para ver los nuevos datos reflejados en el dashboard.")
                                    
                                    # Limpiar el estado de la sesión para permitir nueva carga
                                    st.session_state['archivo_validado'] = False
                                    st.session_state['resultado_validacion'] = None
                                    st.session_state['archivo_cargado'] = None
                                    
                                    # Mostrar mensaje de éxito y botón para cargar otro archivo
                                    st.success("✅ ¡Importación completada con éxito!")
                                    if st.button("🔄 Cargar otro archivo", key="cargar_otro_archivo"):
                                        # Limpiar el estado y forzar recarga
                                        st.session_state.clear()
                                        st.rerun()
                                    
                                    # Detener la ejecución para mostrar el mensaje de éxito
                                    st.stop()
                                    
                                else:
                                    st.error("❌ Error: No se pudo verificar el archivo guardado.")
                                    # Restaurar backup si existe
                                    if backup_path and os.path.exists(backup_path):
                                        shutil.copy2(backup_path, file_path)
                                        st.info("✓ Archivo original restaurado desde backup.")
                                
                            except Exception as e:
                                st.error(f"❌ Error al guardar el archivo: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                                
                                # Limpiar archivo temporal si existe
                                if 'temp_path' in locals() and os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                    except:
                                        pass
                                
                                # Restaurar backup si existe
                                if 'backup_path' in locals() and backup_path and os.path.exists(backup_path):
                                    try:
                                        if os.path.exists(file_path):
                                            os.remove(file_path)
                                        shutil.copy2(backup_path, file_path)
                                        st.info("✓ Archivo original restaurado desde backup después del error.")
                                    except Exception as restore_error:
                                        st.error(f"Error al restaurar el backup: {str(restore_error)}")
                    
                    with col2:
                        if st.button("❌ Cancelar", key="cancelar_importacion"):
                            st.session_state['archivo_validado'] = False
                            st.session_state['resultado_validacion'] = None
                            st.session_state['archivo_cargado'] = None
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

# 4. Reportes
elif page == "📈 Reportes":
    st.header("📈 Reportes")
    
    # Obtener lista de jugadores activos
    @st.cache_data(ttl=3600)
    def load_active_players():
        try:
            from db_utils import get_active_players
            df = get_active_players()
            if not df.empty:
                return df.set_index('id')['nombre_completo'].to_dict()
            return {}
        except Exception as e:
            st.error(f"Error al cargar jugadores: {e}")
            return {}
    
    # Obtener resumen mensual por jugador
    @st.cache_data(ttl=600)
    def get_monthly_summary_by_player(start_date, end_date):
        try:
            from db_utils import get_connection
            import pandas as pd
            
            conn = get_connection()
            query = """
            SELECT 
                j.id as jugador_id,
                j.nombre_completo as jugador,
                COALESCE(e.entrenamientos, 0) as entrenamientos,
                COALESCE(m.meetings, 0) as meetings,
                COALESCE(r.review_clips, 0) as review_clips,
                (COALESCE(e.duracion_total, 0) / 60.0) as horas_entrenamiento,
                (COALESCE(r.duracion_total, 0) / 3600.0) as horas_revision
            FROM jugadores j
            LEFT JOIN (
                SELECT 
                    jugador_id,
                    COUNT(*) as entrenamientos,
                    SUM(duracion_minutos) as duracion_total
                FROM entrenamientos_individuales
                WHERE fecha BETWEEN %s AND %s
                GROUP BY jugador_id
            ) e ON j.id = e.jugador_id
            LEFT JOIN (
                SELECT 
                    jugador_id,
                    COUNT(*) as meetings
                FROM meetings
                WHERE fecha BETWEEN %s AND %s
                GROUP BY jugador_id
            ) m ON j.id = m.jugador_id
            LEFT JOIN (
                SELECT 
                    jugador_id,
                    COUNT(*) as review_clips,
                    SUM(duracion_segundos) as duracion_total
                FROM review_clips
                WHERE fecha BETWEEN %s AND %s
                GROUP BY jugador_id
            ) r ON j.id = r.jugador_id
            WHERE j.activo = 1
            ORDER BY j.nombre_completo
            """
            
            params = (start_date, end_date, start_date, end_date, start_date, end_date)
            df = pd.read_sql(query, conn, params=params)
            
            # Calcular total de actividades y horas totales
            df['total_actividades'] = df['entrenamientos'] + df['meetings'] + df['review_clips']
            df['horas_totales'] = df['horas_entrenamiento'] + df['horas_revision']
            
            return df
            
        except Exception as e:
            st.error(f"Error al obtener resumen mensual: {e}")
            return pd.DataFrame()
    
    # Obtener actividades para exportar
    def get_activities_for_export(start_date, end_date):
        try:
            from db_utils import get_connection
            import pandas as pd
            
            conn = get_connection()
            
            # Obtener entrenamientos
            query_entrenamientos = """
            SELECT 
                e.id,
                j.nombre_completo as jugador,
                'Entrenamiento' as tipo,
                e.fecha,
                e.objetivo,
                e.resultado,
                e.duracion_minutos as duracion,
                e.notas,
                NULL as enlace_video,
                NULL as etiquetas
            FROM entrenamientos_individuales e
            JOIN jugadores j ON e.jugador_id = j.id
            WHERE e.fecha BETWEEN %s AND %s
            """
            
            # Obtener meetings
            query_meetings = """
            SELECT 
                m.id,
                j.nombre_completo as jugador,
                CONCAT('Meeting - ', m.tipo) as tipo,
                m.fecha,
                m.titulo as objetivo,
                m.descripcion as resultado,
                NULL as duracion,
                m.notas,
                NULL as enlace_video,
                NULL as etiquetas
            FROM meetings m
            JOIN jugadores j ON m.jugador_id = j.id
            WHERE m.fecha BETWEEN %s AND %s
            """
            
            # Obtener review clips
            query_clips = """
            SELECT 
                r.id,
                j.nombre_completo as jugador,
                'Review Clip' as tipo,
                r.fecha,
                r.titulo as objetivo,
                r.descripcion as resultado,
                r.duracion_segundos as duracion,
                r.notas,
                r.enlace_video,
                r.etiquetas
            FROM review_clips r
            JOIN jugadores j ON r.jugador_id = j.id
            WHERE r.fecha BETWEEN %s AND %s
            """
            
            params = (start_date, end_date)
            
            df_entrenamientos = pd.read_sql(query_entrenamientos, conn, params=params)
            df_meetings = pd.read_sql(query_meetings, conn, params=params)
            df_clips = pd.read_sql(query_clips, conn, params=params)
            
            return {
                'entrenamientos': df_entrenamientos,
                'meetings': df_meetings,
                'clips': df_clips
            }
            
        except Exception as e:
            st.error(f"Error al obtener datos para exportar: {e}")
            return {}
    
    # Obtener fechas según el período seleccionado
    def get_dates_from_period(period):
        end_date = datetime.now().date()
        if period == "Último mes":
            start_date = end_date - timedelta(days=30)
        elif period == "Últimos 3 meses":
            start_date = end_date - timedelta(days=90)
        elif period == "Último año":
            start_date = end_date - timedelta(days=365)
        else:  # Todo
            start_date = datetime(2020, 1, 1).date()
        return start_date, end_date
    
    # Cargar jugadores activos
    jugadores = load_active_players()
    
    if not jugadores:
        st.warning("No se encontraron jugadores activos.")
        st.stop()
    
    # Selector de tipo de reporte
    report_type = st.selectbox(
        "Tipo de Reporte:",
        ["Resumen Mensual", "Comparativo Jugadores", "Exportar Datos"]
    )
    
    # Sección de Resumen Mensual
    if report_type == "Resumen Mensual":
        st.subheader("Resumen Mensual")
        
        # Obtener los últimos 6 meses
        current_date = datetime.now()
        months = []
        for i in range(6):
            month = (current_date - pd.DateOffset(months=i)).strftime("%B %Y")
            months.append(month)
        
        # Selector de mes
        mes_seleccionado = st.selectbox("Seleccionar Mes:", months, index=0)
        
        # Calcular fechas para el mes seleccionado
        from datetime import datetime
        import calendar
        
        mes_num = datetime.strptime(mes_seleccionado, "%B %Y").month
        anio = datetime.strptime(mes_seleccionado, "%B %Y").year
        
        # Obtener el primer y último día del mes
        start_date = datetime(anio, mes_num, 1).strftime('%Y-%m-%d')
        last_day = calendar.monthrange(anio, mes_num)[1]
        end_date = datetime(anio, mes_num, last_day).strftime('%Y-%m-%d')
        
        # Obtener datos del resumen mensual
        df_resumen = get_monthly_summary_by_player(start_date, end_date)
        
        if not df_resumen.empty:
            # Mostrar tabla con resumen por jugador
            st.dataframe(
                df_resumen[['jugador', 'entrenamientos', 'meetings', 'review_clips', 'total_actividades']],
                column_config={
                    'jugador': 'Jugador',
                    'entrenamientos': 'Entrenamientos',
                    'meetings': 'Meetings',
                    'review_clips': 'Review Clips',
                    'total_actividades': 'Total Actividades'
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Agregar gráfico de evolución por jugador
            st.subheader(f"Evolución de Actividades por Jugador ({start_date} - {end_date})")
            
            # Obtener datos para el gráfico
            df_actividades = load_training_data()
            
            # Filtrar por rango de fechas
            date_mask = (df_actividades['Date'].dt.date >= datetime.strptime(start_date, '%Y-%m-%d').date()) & (df_actividades['Date'].dt.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
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
                
                # Crear gráfico de líneas
                fig_jugadores = px.line(
                    df_agrupado, 
                    x='Mes', 
                    y='Entrenamientos',
                    color='Jugador',
                    title='Evolución de Entrenamientos por Jugador',
                    labels={'Entrenamientos': 'Cantidad de Entrenamientos', 'Mes': 'Mes'},
                    markers=True
                )
                
                fig_jugadores.update_layout(
                    xaxis_title='Mes',
                    yaxis_title='Cantidad de Actividades',
                    legend_title='Jugador',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_jugadores, use_container_width=True)
                
                # Crear pestañas para cada tipo de actividad
                tab1, tab2, tab3 = st.tabs([" Entrenamientos", " Meetings", " Review Clips"])
                
                with tab1:
                    fig_entrenamientos = px.line(
                        df_agrupado, 
                        x='Mes', 
                        y='Entrenamientos',
                        color='Jugador',
                        title='Entrenamientos por Mes',
                        labels={'Entrenamientos': 'Cantidad', 'Mes': 'Mes'},
                        markers=True
                    )
                    st.plotly_chart(fig_entrenamientos, use_container_width=True)
                
                with tab2:
                    fig_meetings = px.line(
                        df_agrupado, 
                        x='Mes', 
                        y='Meetings',
                        color='Jugador',
                        title='Meetings por Mes',
                        labels={'Meetings': 'Cantidad', 'Mes': 'Mes'},
                        markers=True
                    )
                    st.plotly_chart(fig_meetings, use_container_width=True)
                
                with tab3:
                    fig_clips = px.line(
                        df_agrupado, 
                        x='Mes', 
                        y='Review_Clips',
                        color='Jugador',
                        title='Review Clips por Mes',
                        labels={'Review_Clips': 'Cantidad', 'Mes': 'Mes'},
                        markers=True
                    )
                    st.plotly_chart(fig_clips, use_container_width=True)
            else:
                st.warning("No hay datos de actividades para mostrar en el rango de fechas seleccionado.")
        else:
            st.warning("No hay datos disponibles para el rango de fechas seleccionado.")
            
            if not df_resumen.empty:
                # Mostrar gráfico de barras
                fig = px.bar(
                    df_resumen.melt(
                        id_vars=['jugador'],
                        value_vars=['entrenamientos', 'meetings', 'review_clips'],
                        var_name='tipo_actividad',
                        value_name='cantidad'
                    ),
                    x='jugador',
                    y='cantidad',
                    color='tipo_actividad',
                    title=f'Actividades por Jugador - {mes_seleccionado}',
                    labels={'jugador': 'Jugador', 'cantidad': 'Cantidad', 'tipo_actividad': 'Tipo de Actividad'},
                    barmode='group'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos disponibles para el período seleccionado.")
    
    # Sección de Comparativo de Jugadores
    elif report_type == "Comparativo Jugadores":
        st.subheader("Comparativo entre Jugadores")
        
        # Selector de período
        periodo = st.selectbox(
            "Período:",
            ["Último mes", "Últimos 3 meses", "Último año", "Todo"],
            index=0
        )
        
        # Obtener fechas según el período seleccionado
        start_date, end_date = get_dates_from_period(periodo)
        
        # Obtener datos comparativos
        df_comparativo = get_monthly_summary_by_player(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not df_comparativo.empty:
            # Mostrar métricas comparativas
            st.write(f"### Métricas Comparativas ({periodo})")
            
            # Mostrar métricas en columnas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jugadores", len(df_comparativo))
            with col2:
                st.metric("Total Actividades", int(df_comparativo['total_actividades'].sum()))
            with col3:
                st.metric("Promedio Actividades/Jugador", f"{df_comparativo['total_actividades'].mean():.1f}")
            with col4:
                st.metric("Total Horas", f"{df_comparativo['horas_totales'].sum():.1f}")
            
            # Mostrar tabla comparativa
            st.dataframe(
                df_comparativo[['jugador', 'total_actividades', 'horas_totales']],
                column_config={
                    'jugador': 'Jugador',
                    'total_actividades': 'Total Actividades',
                    'horas_totales': 'Horas Totales',
                },
                use_container_width=True
            )
            
            # Mostrar gráfico de comparación
            fig = px.bar(
                df_comparativo,
                x='jugador',
                y='total_actividades',
                title=f'Total de Actividades por Jugador\n({periodo})',
                labels={'jugador': 'Jugador', 'total_actividades': 'Total de Actividades'},
                color='jugador'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No hay datos disponibles para el período seleccionado.")
    
    # Sección de Exportar Datos
    elif report_type == "Exportar Datos":
        st.subheader("Exportar Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Selector de período
            periodo = st.selectbox(
                "Período:",
                ["Último mes", "Últimos 3 meses", "Último año", "Todo"],
                key="export_period"
            )
        
        with col2:
            # Selector de formato
            formato = st.selectbox(
                "Formato:",
                ["Excel (.xlsx)", "CSV (.csv)"],
                key="export_format"
            )
        
        # Obtener fechas según el período seleccionado
        start_date, end_date = get_dates_from_period(periodo)
        
        # Botón para generar reporte
        if st.button("🔽 Generar y Descargar Reporte", type="primary"):
            with st.spinner('Generando reporte...'):
                # Obtener datos para exportar
                actividades = get_activities_for_export(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not actividades or all(df.empty for df in actividades.values()):
                    st.warning("No hay datos para exportar en el período seleccionado.")
                else:
                    if formato == "Excel (.xlsx)":
                        import io
                        from datetime import datetime
                        
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            # Hoja de Entrenamientos
                            if not actividades['entrenamientos'].empty:
                                actividades['entrenamientos'].to_excel(
                                    writer, 
                                    sheet_name='Entrenamientos', 
                                    index=False
                                )
                            
                            # Hoja de Meetings
                            if not actividades['meetings'].empty:
                                actividades['meetings'].to_excel(
                                    writer, 
                                    sheet_name='Meetings', 
                                    index=False
                                )
                            
                            # Hoja de Review Clips
                            if not actividades['clips'].empty:
                                actividades['clips'].to_excel(
                                    writer, 
                                    sheet_name='Review Clips', 
                                    index=False
                                )
                        
                        # Crear botón de descarga
                        st.download_button(
                            label="📥 Descargar archivo Excel",
                            data=buffer.getvalue(),
                            file_name=f"reporte_actividades_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    else:  # CSV
                        import base64
                        import zipfile
                        from io import BytesIO
                        
                        # Crear archivo ZIP con los CSVs
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Añadir archivos CSV al ZIP
                            if not actividades['entrenamientos'].empty:
                                entrenamientos_csv = actividades['entrenamientos'].to_csv(index=False, encoding='utf-8-sig')
                                zip_file.writestr('entrenamientos.csv', entrenamientos_csv)
                            
                            if not actividades['meetings'].empty:
                                meetings_csv = actividades['meetings'].to_csv(index=False, encoding='utf-8-sig')
                                zip_file.writestr('meetings.csv', meetings_csv)
                            
                            if not actividades['clips'].empty:
                                clips_csv = actividades['clips'].to_csv(index=False, encoding='utf-8-sig')
                                zip_file.writestr('review_clips.csv', clips_csv)
                        
                        # Crear botón de descarga para el ZIP
                        zip_buffer.seek(0)
                        b64 = base64.b64encode(zip_buffer.read()).decode()
                        st.download_button(
                            label="📥 Descargar archivos CSV",
                            data=zip_buffer,
                            file_name=f"reporte_actividades_{datetime.now().strftime('%Y%m%d')}.zip",
                            mime="application/zip"
                        )
    
    # Contenido del reporte según el tipo seleccionado
    if report_type == "Resumen Mensual":
        st.subheader("Resumen Mensual")
        
        # Gráfico de actividades por tipo
        data = {
            'Tipo': ['Entrenamientos', 'Meetings', 'Review Clips'],
            'Cantidad': [24, 12, 8]
        }
        
        fig = px.pie(
            data,
            values='Cantidad',
            names='Tipo',
            title='Distribución de Actividades',
            color_discrete_sequence=['#fcec03', '#1f77b4', '#2ca02c']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Botón para exportar
        if st.button("Exportar a Excel", type="primary"):
            # Crear un archivo Excel en memoria
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(data).to_excel(writer, sheet_name='Resumen', index=False)
                
                # Agregar más hojas si es necesario
                pd.DataFrame({
                    'Jugador': jugadores,
                    'Entrenamientos': [8, 5, 6, 5],
                    'Meetings': [4, 3, 3, 2],
                    'Review Clips': [3, 2, 2, 1]
                }).to_excel(writer, sheet_name='Por Jugador', index=False)
            
            # Descargar el archivo
            st.download_button(
                label="Descargar Reporte",
                data=output.getvalue(),
                file_name=f"reporte_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    elif tipo_reporte == "Comparativo entre Jugadores":
        st.subheader("Comparativo entre Jugadores")
        
        # Selección de jugadores a comparar
        jugadores_comp = st.multiselect(
            "Seleccionar Jugadores",
            options=jugadores,
            default=jugadores[:2],
            max_selections=4
        )
        
        if len(jugadores_comp) >= 2:
            # Datos de ejemplo para la comparación
            data_comp = {
                'Jugador': jugadores_comp * 3,
                'Tipo': ['Entrenamientos']*len(jugadores_comp) + \
                        ['Meetings']*len(jugadores_comp) + \
                        ['Review Clips']*len(jugadores_comp),
                'Cantidad': [8, 5, 6, 5, 4, 3, 3, 2, 3, 2, 2, 1][:len(jugadores_comp)*3]
            }
            
            fig = px.bar(
                data_comp,
                x='Jugador',
                y='Cantidad',
                color='Tipo',
                barmode='group',
                title='Comparativo de Actividades por Jugador',
                color_discrete_map={
                    'Entrenamientos': '#fcec03',
                    'Meetings': '#1f77b4',
                    'Review Clips': '#2ca02c'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Seleccione al menos dos jugadores para comparar.")
    
    else:  # Evolución de Métricas
        st.subheader("Evolución de Métricas")
        
        # Datos de ejemplo para la evolución
        fechas = pd.date_range(end=datetime.now(), periods=6, freq='M')
        data_evol = {
            'Fecha': list(fechas) * 3,
            'Tipo': ['Entrenamientos']*6 + ['Meetings']*6 + ['Review Clips']*6,
            'Cantidad': [15, 18, 20, 22, 24, 24, 8, 9, 10, 11, 12, 12, 5, 6, 7, 7, 8, 8]
        }
        
        fig = px.line(
            data_evol,
            x='Fecha',
            y='Cantidad',
            color='Tipo',
            markers=True,
            title='Evolución de Actividades en el Tiempo',
            color_discrete_map={
                'Entrenamientos': '#fcec03',
                'Meetings': '#1f77b4',
                'Review Clips': '#2ca02c'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
