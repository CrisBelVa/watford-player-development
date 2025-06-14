#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para migrar datos de los archivos Excel de Alberto a la base de datos.
"""

import os
import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path para poder importar db_utils
sys.path.append(str(Path(__file__).parent.parent))
from db_utils import get_connection

# Configuración
EXCEL_FILES = {
    'training': 'Alberto Individuals - Training 4.xlsx',
    'meetings': 'Alberto Individuals 9.xlsx'
}

# Mapeo de meses en inglés a números
MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def clean_player_name(sheet_name):
    """
    Limpia el nombre del jugador del nombre de la hoja.
    
    Args:
        sheet_name (str): Nombre de la hoja del Excel
        
    Returns:
        str: Nombre del jugador limpio
    """
    # Eliminar 'Player - ' del inicio si existe
    if sheet_name.startswith('Player - '):
        return sheet_name[9:].strip()
    return sheet_name.strip()

def month_to_number(month_name):
    """
    Convierte el nombre de un mes a su número correspondiente.
    
    Args:
        month_name (str): Nombre del mes en inglés (case insensitive)
        
    Returns:
        int: Número del mes (1-12)
        
    Raises:
        ValueError: Si el nombre del mes no es válido
    """
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    month_lower = month_name.lower()
    if month_lower not in months:
        raise ValueError("Mes no válido: {}".format(month_name))
        
    return months[month_lower]

def generate_random_dates(year, month, count):
    """
    Genera fechas aleatorias dentro de un mes específico.
    
    Args:
        year (int): Año
        month (int): Mes (1-12)
        count (int): Número de fechas a generar
        
    Returns:
        list: Lista de fechas generadas (objetos datetime)
    """
    if not (1 <= month <= 12):
        raise ValueError("El mes debe estar entre 1 y 12")
    
    # Determinar el último día del mes
    if month == 12:
        last_day = 31
    else:
        last_day = (datetime(year, month + 1, 1) - timedelta(days=1)).day
    
    # Generar fechas aleatorias
    dates = set()
    while len(dates) < count:
        day = random.randint(1, last_day)
        try:
            # Verificar si la fecha es válida (por ejemplo, 30 de febrero)
            date = datetime(year, month, day)
            # Asegurarse de que no sea fin de semana (sábado=5, domingo=6)
            if date.weekday() < 5:  # Lunes a viernes
                dates.add(date)
        except ValueError:
            continue
    
    # Ordenar las fechas
    return sorted(list(dates))

def migrate_players(sheet_names, conn):
    """
    Migra los jugadores a la base de datos.
    
    Args:
        sheet_names (list): Lista de nombres de hojas del Excel
        conn: Conexión a la base de datos
        
    Returns:
        dict: Diccionario con el nombre del jugador y su ID
    """
    print("\n🔍 Migrando jugadores...")
    player_ids = {}
    
    with conn.cursor() as cursor:
        # Obtener jugadores existentes
        cursor.execute("SELECT id, nombre FROM jugadores")
        existing_players = {name.lower(): player_id for player_id, name in cursor.fetchall()}
        
        for sheet_name in sheet_names:
            # Limpiar el nombre del jugador
            name = clean_player_name(sheet_name)
            name_lower = name.lower()
            
            if name_lower in existing_players:
                player_ids[name] = existing_players[name_lower]
                print("  ✓ Jugador existente: {} (ID: {})".format(name, existing_players[name_lower]))
            else:
                # Insertar nuevo jugador
                try:
                    cursor.execute(
                        "INSERT INTO jugadores (nombre, posicion, fecha_nacimiento, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (name, 'Desconocida', '2000-01-01', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    )
                    conn.commit()
                    
                    # Obtener el ID del jugador insertado
                    cursor.execute("SELECT last_insert_rowid()")
                    player_id = cursor.fetchone()[0]
                    player_ids[name] = player_id
                    print("  ✓ Nuevo jugador insertado: {} (ID: {})".format(name, player_id))
                    
                except Exception as e:
                    print("  ❌ Error insertando jugador {}: {}".format(name, str(e)))
                    conn.rollback()
    
    return player_ids

def insert_meeting(player_id, date, conn):
    """
    Inserta una reunión en la base de datos.
    
    Args:
        player_id (int): ID del jugador
        date (datetime): Fecha de la reunión
        conn: Conexión a la base de datos
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO meetings 
            (jugador_id, fecha, tipo, titulo, descripcion, notas)
            VALUES (?, ?, 'Individual', 'Reunión de seguimiento', 'Reunión de seguimiento', 'Migrado automáticamente')
            """,
            (player_id, date.strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()

def insert_training(player_id, date, conn):
    """
    Inserta un entrenamiento en la base de datos.
    
    Args:
        player_id: ID del jugador
        date: Fecha del entrenamiento
        conn: Conexión a la base de datos
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO entrenamientos_individuales 
            (jugador_id, fecha, objetivo, resultado, duracion_minutos, notas)
            VALUES (?, ?, 'Entrenamiento individual', 'Completado', 60, 'Migrado automáticamente')
            """,
            (player_id, date.strftime('%Y-%m-%d'))
        )
        conn.commit()

def migrate_training_data(file_path, player_ids, conn):
    """
    Migra los datos de entrenamiento del archivo Excel.
    
    Args:
        file_path: Ruta al archivo Excel
        player_ids: Diccionario con los IDs de los jugadores
        conn: Conexión a la base de datos
    """
    print("\nMigrando datos de entrenamiento...")
    
    # Leer todas las hojas del Excel
    xls = pd.ExcelFile(file_path)
    
    for sheet_name in xls.sheet_names:
        if ' - T&M' not in sheet_name:
            continue
            
        player_name = clean_player_name(sheet_name)
        if player_name not in player_ids:
            print("  Jugador no encontrado: {}".format(player_name))
            continue
            
        player_id = player_ids[player_name]
        print("\nProcesando {}...".format(player_name))
        
        try:
            # Leer la hoja del jugador
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            # Buscar fila con los meses
            month_row = None
            for i, row in df.iterrows():
                if any(isinstance(cell, str) and 'september' in str(cell).lower() for cell in row):
                    month_row = i
                    break
            
            if month_row is None:
                print("  No se encontró la fila de meses en la hoja {}".format(sheet_name))
                continue
            
            # Obtener los meses
            months = [str(cell).lower() for cell in df.iloc[month_row] if isinstance(cell, str)]
            
            # Buscar filas con actividades
            activity_rows = {}
            for i, row in df.iterrows():
                if i <= month_row:
                    continue
                    
                cell_value = str(row[0]).lower() if pd.notna(row[0]) else ""
                if 'individual training' in cell_value:
                    activity_rows['training'] = i
                elif 'meeting' in cell_value and 'review' not in cell_value:
                    activity_rows['meeting'] = i
                elif 'review clips' in cell_value:
                    activity_rows['review'] = i
            
            # Procesar cada mes
            for col_idx, month_cell in enumerate(df.iloc[month_row]):
                if not isinstance(month_cell, str):
                    continue
                    
                month_name = month_cell.strip().lower()
                if month_name not in MONTHS:
                    continue
                    
                month_num = MONTHS[month_name]
                year = 2024  # Año por defecto, ajustar según sea necesario
                
                # Procesar cada tipo de actividad
                for activity_type, row_idx in activity_rows.items():
                    try:
                        count = df.iat[row_idx, col_idx + 1]  # +1 porque la columna 0 es el nombre
                        if pd.isna(count) or not str(count).isdigit():
                            continue
                            
                        count = int(float(count))
                        if count <= 0:
                            continue
                            
                        # Generar fechas aleatorias para las actividades
                        dates = generate_random_dates(year, month_num, count)
                        
                        # Insertar actividades
                        with conn.cursor() as cursor:
                            for date in dates:
                                if activity_type == 'training':
                                    cursor.execute(
                                        """
                                        INSERT INTO entrenamientos_individuales 
                                        (jugador_id, fecha, objetivo, resultado, duracion_minutos, notas)
                                        VALUES (?, ?, 'Entrenamiento individual', 'Completado', 60, 'Migrado automáticamente')
                                        """,
                                        (player_id, date.strftime('%Y-%m-%d'))
                                    )
                                elif activity_type == 'meeting':
                                    cursor.execute(
                                        """
                                        INSERT INTO meetings 
                                        (jugador_id, fecha, tipo, titulo, descripcion, notas)
                                        VALUES (?, ?, 'Individual', 'Reunión de seguimiento', 'Reunión de seguimiento', 'Migrado automáticamente')
                                        """,
                                        (player_id, date.strftime('%Y-%m-%d %H:%M:%S'))
                                    )
                                elif activity_type == 'review':
                                    cursor.execute(
                                        """
                                        INSERT INTO review_clips 
                                        (jugador_id, fecha, titulo, descripcion, enlace_video, duracion_segundos, etiquetas, notas)
                                        VALUES (?, ?, 'Revisión de video', 'Revisión de jugadas', 'https://example.com', 300, 'revisión', 'Migrado automáticamente')
                                        """,
                                        (player_id, date.strftime('%Y-%m-%d %H:%M:%S'))
                                    )
                            
                            conn.commit()
                            print("  ✅ {} en {}".format(count, activity_type, month_cell))
                            
                    except Exception as e:
                        print("  ❌ Error procesando {} en {}: {}".format(activity_type, month_cell, str(e)))
                        conn.rollback()
                        
        except Exception as e:
            print("  ❌ Error procesando la hoja {}: {}".format(sheet_name, str(e)))

def migrate_alberto_excel_data():
    """
    Función principal para migrar los datos de los archivos Excel de Alberto.
    """
    print("=" * 50)
    print("  MIGRACIÓN DE DATOS DE ALBERTO")
    print("=" * 50)
    
    # Obtener la ruta del directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Verificar si existe el directorio de datos
    if not os.path.exists(data_dir):
        print("\n❌ No se encontró el directorio de datos: {}".format(data_dir))
        return
    
    # Buscar archivos Excel
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    if not excel_files:
        print("\n❌ No se encontraron archivos Excel en: {}".format(data_dir))
        return
    
    # Conectar a la base de datos
    try:
        conn = get_connection()
        print("\n✅ Conexión a la base de datos establecida")
    except Exception as e:
        print("\n❌ Error al conectar a la base de datos: {}".format(str(e)))
        return
    
    try:
        # Obtener nombres de hojas de todos los archivos Excel
        sheet_names = []
        for file_name in excel_files:
            try:
                file_path = os.path.join(data_dir, file_name)
                xls = pd.ExcelFile(file_path)
                sheet_names.extend(xls.sheet_names)
            except Exception as e:
                print("\n⚠️  Error al leer el archivo {}: {}".format(file_name, str(e)))
        
        # Migrar jugadores
        player_ids = migrate_players(sheet_names, conn)
        
        # Migrar datos de entrenamiento
        for file_name in excel_files:
            file_path = os.path.join(data_dir, file_name)
            migrate_training_data(file_path, player_ids, conn)
        
        print("\n✅ Migración completada exitosamente!")
        
    except Exception as e:
        print("\n❌ Error durante la migración: {}".format(str(e)))
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("\n🔌 Conexión a la base de datos cerrada")

if __name__ == "__main__":
    print("=" * 50)
    print("  MIGRACIÓN DE DATOS DE ALBERTO")
    print("=" * 50)
    # Verificar que estamos en el directorio correcto
    if not any(f in os.listdir() for f in EXCEL_FILES.values()):
        print("\n⚠️  ADVERTENCIA: No se encontraron los archivos de Excel en el directorio actual.")
        print(f"   Asegúrate de que los archivos {', '.join(EXCEL_FILES.values())} estén en el mismo directorio.")
        print("   Ejecuta este script desde el directorio que contiene los archivos de Excel.\n")
    else:
        migrate_alberto_excel_data()
