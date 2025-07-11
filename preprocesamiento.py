# Instalación de librerías necesarias
!pip install tensorflow pandas numpy matplotlib xlrd openpyxl

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cambiar al directorio de trabajo
os.chdir('/content/drive/MyDrive/tesis_2025')

# Lectura del archivo Excel
file_path = "2009-2024.xlsx"
df = pd.read_excel(file_path, engine='openpyxl') 

# Exploración básica del DataFrame
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

# Filtrado de columnas relevantes
columnas_utiles = [
    'ANO_EJE', 'PLIEGO_NOMBRE', 'EJECUTORA', 'EJECUTORA_NOMBRE',
    'PROGRAMA_PPTO_DESCRIPCION', 'GRUPO_FUNCIONAL_DESCRIPCION',
    'TIPO_TRANSACCION', 'GENERICA', 'SUBGENERICA', 'SUBGENERICA_DET',
    'ESPECIFICA', 'ESPECIFICA_DET', 'MTO_PIA', 'MTO_MODIFICACION',
    'MTO_PIM', 'MTO_DEVENGADO_01', 'MTO_DEVENGADO_02', 'MTO_DEVENGADO_03',
    'MTO_DEVENGADO_04', 'MTO_DEVENGADO_05', 'MTO_DEVENGADO_06',
    'MTO_DEVENGADO_07', 'MTO_DEVENGADO_08', 'MTO_DEVENGADO_09',
    'MTO_DEVENGADO_10', 'MTO_DEVENGADO_11', 'MTO_DEVENGADO_12',
    'MTO_DEVENGADO_13'
]

df_filtrado = df[columnas_utiles]

# Guardar el DataFrame filtrado en un nuevo archivo
df_filtrado.to_excel('data_filtrada.xlsx', index=False)

# Crear columna "CLASIFICADOR" a partir de componentes presupuestales
df_filtrado['CLASIFICADOR'] = df_filtrado[
    ['TIPO_TRANSACCION', 'GENERICA', 'SUBGENERICA',
     'SUBGENERICA_DET', 'ESPECIFICA', 'ESPECIFICA_DET']
].astype(str).agg('.'.join, axis=1)

# Calcular el total devengado anual
columnas_devengado = [
    'MTO_DEVENGADO_01', 'MTO_DEVENGADO_02', 'MTO_DEVENGADO_03',
    'MTO_DEVENGADO_04', 'MTO_DEVENGADO_05', 'MTO_DEVENGADO_06',
    'MTO_DEVENGADO_07', 'MTO_DEVENGADO_08', 'MTO_DEVENGADO_09',
    'MTO_DEVENGADO_10', 'MTO_DEVENGADO_11', 'MTO_DEVENGADO_12',
    'MTO_DEVENGADO_13']

df_filtrado['MTO_DEVENGADO'] = df_filtrado[columnas_devengado].fillna(0).sum(axis=1)

# Crear DataFrame final con las columnas más relevantes
df_final = df_filtrado[[
    'ANO_EJE', 'PLIEGO_NOMBRE', 'EJECUTORA_NOMBRE',
    'PROGRAMA_PPTO_DESCRIPCION', 'GRUPO_FUNCIONAL_DESCRIPCION',
    'CLASIFICADOR', 'MTO_PIA', 'MTO_PIM', 'MTO_DEVENGADO'
]]

# Agrupar por año, pliego y ejecutora para obtener sumatorias
df_resultado = df_filtrado.groupby(
    ['ANO_EJE', 'PLIEGO_NOMBRE', 'EJECUTORA_NOMBRE'],
    as_index=False
)[['MTO_PIM', 'MTO_DEVENGADO']].sum()

# Guardar resultados agregados en Excel
df_resultado.to_excel('data_resultado2.xlsx', index=False)
