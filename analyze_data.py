#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os

def analyze_excel_file(filepath):
    print("\nAnalizando archivo: {}".format(os.path.basename(filepath)))
    print("-" * 50)
    
    # Leer todas las hojas del archivo Excel
    xl = pd.ExcelFile(filepath)
    
    print("Hojas disponibles: {}".format(xl.sheet_names))
    
    # Analizar cada hoja
    for sheet_name in xl.sheet_names:
        print("\nHoja: {}".format(sheet_name))
        print("-" * 30)
        
        try:
            # Leer solo las primeras filas para el análisis
            df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=5)
            
            # Mostrar información básica
            print("Número de columnas: {}".format(len(df.columns)))
            print("Número de filas (muestra): {}".format(len(df)))
            print("\nPrimeras filas:")
            print(df.head().to_string())
            
            # Mostrar tipos de datos
            print("\nTipos de datos:")
            print(df.dtypes)
            
        except Exception as e:
            print("Error al analizar la hoja {}: {}".format(sheet_name, str(e)))

if __name__ == "__main__":
    # Ruta a los archivos
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Analizar cada archivo
    files_to_analyze = [
        os.path.join(data_dir, 'Alberto Individuals - Training 4.xlsx'),
        os.path.join(data_dir, 'Alberto Individuals 9.xlsx')
    ]
    
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            analyze_excel_file(file_path)
        else:
            print("Archivo no encontrado: {}".format(file_path))
