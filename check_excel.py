#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sys

try:
    # Leer el archivo Excel
    file_path = '/Users/lucas/Desktop/DPTO PRACTICAS PROF/WATFORD/watford-player-development/data/Individuals - Training bkp.xlsx'
    
    # Leer todas las hojas
    xls = pd.ExcelFile(file_path)
    print("Hojas en el archivo: {}".format(xls.sheet_names))
    
    # Leer la primera hoja
    df = pd.read_excel(file_path, header=1)  # Usar header=1 para saltar la primera fila
    
    # Mostrar información del DataFrame
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    print("\nColumnas:")
    print(df.columns.tolist())
    
    print("\nInformación del DataFrame:")
    print(df.info())
    
    # Verificar si hay valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    # Verificar tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes)
    
except Exception as e:
    sys.stderr.write("Error al leer el archivo: {}\n".format(str(e)))
    import traceback
    traceback.print_exc()
