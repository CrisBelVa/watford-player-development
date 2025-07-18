import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import List, Tuple, Dict, Any, Optional, Union
import streamlit as st
import altair as alt
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

def connect_to_db():
    """
    Establece conexión a la base de datos con manejo de errores mejorado.
    
    Returns:
        sqlalchemy.engine.Engine: Objeto de conexión a la base de datos o None en caso de error
    """
    try:
        load_dotenv()

        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = os.getenv("DB_NAME")

        if not all([user, password, host, port, database]):
            st.error("❌ Error: Faltan credenciales de la base de datos.")
            st.warning("Por favor, verifica tu archivo .env")
            return None

        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"connect_timeout": 5}
        )
        
        # Verificar la conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        return engine
        
    except Exception as e:
        st.error(f"❌ Error al conectar a la base de datos: {str(e)}")
        return None

# [Resto de las funciones existentes...]

@st.cache_data(ttl=3600)  # Cachear por 1 hora
async def get_active_players(cached: bool = True) -> pd.DataFrame:
    """
    Obtiene la lista de jugadores activos con manejo de caché y errores mejorado.
    
    Args:
        cached: Si es True, usa datos en caché si están disponibles
        
    Returns:
        pd.DataFrame: DataFrame con los jugadores activos o DataFrame vacío en caso de error
    """
    if cached and 'cached_active_players' in st.session_state:
        return st.session_state.cached_active_players
        
    engine = connect_to_db()
    if not engine:
        st.warning("No se pudo conectar a la base de datos para obtener jugadores activos.")
        return pd.DataFrame()
        
    query = """
    SELECT 
        id, 
        COALESCE(CONCAT(nombre, ' ', apellido), nombre, 'Jugador ' || id) as nombre_completo, 
        COALESCE(posicion, 'Sin posición') as posicion, 
        COALESCE(equipo_actual, 'Sin equipo') as equipo_actual
    FROM jugadores 
    WHERE activo = 1 
    ORDER BY nombre, apellido
    """
    
    try:
        with st.spinner("Cargando jugadores activos..."):
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
                
                # Validar datos mínimos
                if df.empty:
                    st.warning("No se encontraron jugadores activos en la base de datos.")
                    return pd.DataFrame()
                    
                # Almacenar en caché de sesión
                st.session_state.cached_active_players = df
                return df
                
    except Exception as e:
        st.error(f"❌ Error al cargar jugadores activos: {str(e)}")
        if hasattr(e, 'orig') and hasattr(e.orig, 'args'):
            st.error(f"Detalles: {e.orig.args}")
        return pd.DataFrame()


@st.cache_data(ttl=600)  # Cachear por 10 minutos
def get_player_activities(player_id: int, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Obtiene las actividades de un jugador con filtros opcionales por fechas.
    
    Args:
        player_id: ID del jugador
        start_date: Fecha de inicio en formato 'YYYY-MM-DD' (opcional)
        end_date: Fecha de fin en formato 'YYYY-MM-DD' (opcional)
        
    Returns:
        pd.DataFrame: DataFrame con las actividades del jugador o DataFrame vacío en caso de error
    """
    # Verificar si hay caché disponible
    cache_key = f"player_activities_{player_id}_{start_date}_{end_date}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Validar fechas
    try:
        if start_date:
            datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        st.error(f"❌ Formato de fecha inválido. Use YYYY-MM-DD. Error: {str(e)}")
        return pd.DataFrame()
        
    # Conectar a la base de datos
    engine = connect_to_db()
    if not engine:
        st.warning("⚠️ No se pudo conectar a la base de datos.")
        return pd.DataFrame()
    
    # Construir condiciones de consulta
    where_conditions = ["j.id = :player_id"]
    params = {"player_id": player_id}
    
    if start_date:
        where_conditions.append("a.fecha >= :start_date")
        params['start_date'] = start_date
    if end_date:
        where_conditions.append("a.fecha <= :end_date")
        params['end_date'] = end_date
        
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    # Construir consultas individuales para cada tipo de actividad
    entrenamientos_query = ("""
    SELECT 
        j.id as jugador_id,
        CONCAT(j.nombre, ' ', j.apellido) as jugador,
        'Entrenamiento Individual' as tipo,
        e.fecha,
        e.objetivo,
        e.resultado,
        e.duracion_minutos as duracion,
        e.notas,
        NULL as enlace_video,
        NULL as etiquetas,
        'min' as duracion_unidad
    FROM entrenamientos_individuales e
    JOIN jugadores j ON e.jugador_id = j.id
    """ + f"""
    WHERE {where_clause.replace('a.', 'e.')}""").strip()
    
    reuniones_query = ("""
    SELECT 
        j.id as jugador_id,
        CONCAT(j.nombre, ' ', j.apellido) as jugador,
        CONCAT('Reunión - ', m.tipo) as tipo,
        m.fecha,
        m.titulo as objetivo,
        m.descripcion as resultado,
        NULL as duracion,
        m.notas,
        NULL as enlace_video,
        NULL as etiquetas,
        NULL as duracion_unidad
    FROM meetings m
    JOIN jugadores j ON m.jugador_id = j.id
    """ + f"""
    WHERE {where_clause.replace('a.', 'm.')}""").strip()
    
    review_clips_query = ("""
    SELECT 
        j.id as jugador_id,
        CONCAT(j.nombre, ' ', j.apellido) as jugador,
        'Review Clip' as tipo,
        r.fecha,
        r.titulo as objetivo,
        r.descripcion as resultado,
        r.duracion_segundos as duracion,
        r.notas,
        r.enlace_video,
        r.etiquetas,
        'seg' as duracion_unidad
    FROM review_clips r
    JOIN jugadores j ON r.jugador_id = j.id
    """ + f"""
    WHERE {where_clause.replace('a.', 'r.')}""").strip()
    
    # Combinar todas las consultas
    query = f"""
    {entrenamientos_query}
    UNION ALL
    {reuniones_query}
    UNION ALL
    {review_clips_query}
    ORDER BY fecha DESC
    """
    
    try:
        with st.spinner("📊 Cargando actividades..."):
            with engine.connect() as conn:
                # Ejecutar consulta con parámetros
                df = pd.read_sql(query, conn, params=params)
                
                if df.empty:
                    st.info("ℹ️ No se encontraron actividades para el período seleccionado.")
                    return pd.DataFrame()
                
                # Procesar fechas
                df['fecha'] = pd.to_datetime(df['fecha'])
                
                # Convertir duración a minutos para consistencia
                if 'duracion' in df.columns:
                    df['duracion_min'] = df.apply(
                        lambda x: x['duracion'] / 60 if x['duracion_unidad'] == 'seg' else x['duracion'], 
                        axis=1
                    )
                
                # Almacenar en caché
                st.session_state[cache_key] = df
                return df
                
    except Exception as e:
        st.error(f"❌ Error al cargar actividades: {str(e)}")
        if hasattr(e, 'orig') and hasattr(e.orig, 'args'):
            st.error(f"Detalles: {e.orig.args}")
        return pd.DataFrame()


def get_monthly_summary_all_players(months: int = 12) -> pd.DataFrame:
    """
    Obtiene un resumen mensual de actividades para todos los jugadores.
    
    Args:
        months: Número de meses hacia atrás para incluir en el resumen
        
    Returns:
        pd.DataFrame: DataFrame con el resumen de actividades por mes
    """
    engine = connect_to_db()
    if not engine:
        st.warning("⚠️ No se pudo conectar a la base de datos.")
        return pd.DataFrame()
    
    query = """
    WITH fechas AS (
        SELECT DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL n MONTH), '%Y-%m-01') as fecha_inicio
        FROM (
            SELECT 0 as n UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION
            SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION
            SELECT 10 UNION SELECT 11
        ) nums
        WHERE n <= :months
    ),
    actividades AS (
        SELECT 
            DATE_FORMAT(fecha, '%Y-%m') as mes,
            COUNT(CASE WHEN tipo = 'entrenamiento' THEN 1 END) as entrenamientos,
            COUNT(CASE WHEN tipo = 'meeting' THEN 1 END) as meetings,
            COUNT(CASE WHEN tipo = 'review_clip' THEN 1 END) as review_clips
        FROM (
            SELECT fecha, 'entrenamiento' as tipo FROM entrenamientos_individuales
            UNION ALL
            SELECT fecha, 'meeting' as tipo FROM meetings
            UNION ALL
            SELECT fecha, 'review_clip' as tipo FROM review_clips
        ) t
        WHERE fecha >= (SELECT MIN(fecha_inicio) FROM fechas)
        GROUP BY DATE_FORMAT(fecha, '%Y-%m')
    )
    SELECT 
        f.fecha_inicio as mes,
        COALESCE(a.entrenamientos, 0) as entrenamientos,
        COALESCE(a.meetings, 0) as meetings,
        COALESCE(a.review_clips, 0) as review_clips
    FROM fechas f
    LEFT JOIN actividades a ON f.fecha_inicio = CONCAT(a.mes, '-01')
    ORDER BY f.fecha_inicio
    """
    
    try:
        with st.spinner("📊 Generando resumen mensual..."):
            with engine.connect() as conn:
                result = pd.read_sql(
                    text(query), 
                    conn, 
                    params={"months": months-1}
                )
                return result
    except Exception as e:
        st.error(f"❌ Error al obtener resumen mensual: {e}")
        if hasattr(e, 'orig') and hasattr(e.orig, 'args'):
            st.error(f"Detalles: {e.orig.args}")
        return pd.DataFrame()


def get_monthly_summary_by_player(month_year: str) -> pd.DataFrame:
    """
    Obtiene un resumen de actividades por jugador para un mes específico.
    
    Args:
        month_year: Mes y año en formato 'YYYY-MM'
        
    Returns:
        pd.DataFrame: DataFrame con el resumen de actividades por jugador
    """
    # Validar formato de fecha
    try:
        datetime.strptime(month_year, '%Y-%m')
    except ValueError:
        st.error("❌ Formato de fecha inválido. Use YYYY-MM")
        return pd.DataFrame()
    
    engine = connect_to_db()
    if not engine:
        st.warning("⚠️ No se pudo conectar a la base de datos.")
        return pd.DataFrame()
    
    query = """
    SELECT 
        j.id as jugador_id,
        CONCAT(j.nombre, ' ', j.apellido) as jugador,
        COALESCE(j.posicion, 'Sin posición') as posicion,
        COALESCE(j.equipo_actual, 'Sin equipo') as equipo,
        COALESCE(e.entrenamientos, 0) as entrenamientos,
        COALESCE(m.meetings, 0) as meetings,
        COALESCE(r.review_clips, 0) as review_clips,
        (COALESCE(e.entrenamientos, 0) + 
         COALESCE(m.meetings, 0) + 
         COALESCE(r.review_clips, 0)) as total_actividades
    FROM jugadores j
    LEFT JOIN (
        SELECT jugador_id, COUNT(*) as entrenamientos
        FROM entrenamientos_individuales
        WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year
        GROUP BY jugador_id
    ) e ON j.id = e.jugador_id
    LEFT JOIN (
        SELECT jugador_id, COUNT(*) as meetings
        FROM meetings
        WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year
        GROUP BY jugador_id
    ) m ON j.id = m.jugador_id
    LEFT JOIN (
        SELECT jugador_id, COUNT(*) as review_clips
        FROM review_clips
        WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year
        GROUP BY jugador_id
    ) r ON j.id = r.jugador_id
    WHERE j.activo = 1
    HAVING total_actividades > 0
    ORDER BY total_actividades DESC, jugador
    """
    
    try:
        with st.spinner("📊 Generando resumen por jugador..."):
            with engine.connect() as conn:
                result = pd.read_sql(
                    text(query), 
                    conn, 
                    params={"month_year": month_year}
                )
                if result.empty:
                    st.info(f"ℹ️ No hay actividades registradas para {month_year}")
                return result
    except Exception as e:
        st.error(f"❌ Error al obtener resumen por jugador: {e}")
        if hasattr(e, 'orig') and hasattr(e.orig, 'args'):
            st.error(f"Detalles: {e.orig.args}")
        return pd.DataFrame()


def insert_entrenamiento(jugador_id: int, fecha: str, objetivo: str, resultado: str, 
                         duracion_minutos: int, notas: str = None) -> bool:
    """Insertar nuevo entrenamiento individual"""
    engine = connect_to_db()
    if not engine:
        return False
    
    query = """
    INSERT INTO entrenamientos_individuales 
    (jugador_id, fecha, objetivo, resultado, duracion_minutos, notas, created_at)
    VALUES (:jugador_id, :fecha, :objetivo, :resultado, :duracion_minutos, :notas, NOW())
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {
                'jugador_id': jugador_id,
                'fecha': fecha,
                'objetivo': objetivo,
                'resultado': resultado,
                'duracion_minutos': duracion_minutos,
                'notas': notas
            })
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al insertar entrenamiento: {e}")
        return False


def insert_meeting(jugador_id: int, fecha: str, tipo: str, titulo: str, 
                  descripcion: str = None, notas: str = None) -> bool:
    """Insertar nuevo meeting"""
    engine = connect_to_db()
    if not engine:
        return False
    
    query = """
    INSERT INTO meetings 
    (jugador_id, fecha, tipo, titulo, descripcion, notas, created_at)
    VALUES (:jugador_id, :fecha, :tipo, :titulo, :descripcion, :notas, NOW())
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {
                'jugador_id': jugador_id,
                'fecha': fecha,
                'tipo': tipo,
                'titulo': titulo,
                'descripcion': descripcion,
                'notas': notas
            })
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al insertar meeting: {e}")
        return False


def insert_review_clip(jugador_id: int, fecha: str, titulo: str, descripcion: str, 
                      enlace_video: str, duracion_segundos: int, 
                      etiquetas: str = None, notas: str = None) -> bool:
    """Insertar nuevo review clip"""
    engine = connect_to_db()
    if not engine:
        return False
    
    query = """
    INSERT INTO review_clips 
    (jugador_id, fecha, titulo, descripcion, enlace_video, 
     duracion_segundos, etiquetas, notas, created_at)
    VALUES (:jugador_id, :fecha, :titulo, :descripcion, :enlace_video, 
            :duracion_segundos, :etiquetas, :notas, NOW())
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {
                'jugador_id': jugador_id,
                'fecha': fecha,
                'titulo': titulo,
                'descripcion': descripcion,
                'enlace_video': enlace_video,
                'duracion_segundos': duracion_segundos,
                'etiquetas': etiquetas,
                'notas': notas
            })
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al insertar review clip: {e}")
        return False


def get_department_metrics(month_year: str) -> Dict[str, Any]:
    """Métricas generales del departamento para un mes"""
    engine = connect_to_db()
    if not engine:
        return {}
    
    query = """
    SELECT 
        (SELECT COUNT(*) FROM entrenamientos_individuales 
         WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year) as total_entrenamientos,
        
        (SELECT COUNT(*) FROM meetings 
         WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year) as total_meetings,
        
        (SELECT COUNT(*) FROM review_clips 
         WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year) as total_review_clips,
        
        (SELECT COUNT(DISTINCT jugador_id) FROM (\n            SELECT jugador_id FROM entrenamientos_individuales WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year\n            UNION\n            SELECT jugador_id FROM meetings WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year\n            UNION\n            SELECT jugador_id FROM review_clips WHERE DATE_FORMAT(fecha, '%Y-%m') = :month_year\n        ) t) as jugadores_activos,
        
        (SELECT COUNT(*) FROM jugadores WHERE activo = 1) as total_jugadores_activos
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), {'month_year': month_year}).fetchone()
            if result:
                return {
                    'total_entrenamientos': result[0],
                    'total_meetings': result[1],
                    'total_review_clips': result[2],
                    'jugadores_activos': result[3],
                    'total_jugadores_activos': result[4],
                    'porcentaje_participacion': round((result[3] / result[4] * 100) if result[4] > 0 else 0, 1)
                }
            return {}
    except Exception as e:
        st.error(f"Error al obtener métricas del departamento: {e}")
        return {}
