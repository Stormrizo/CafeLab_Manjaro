# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

ruta_txt = "Absorbancia_VeMN7_5g.txt"   # ajustar el nombre si el archivo se llama distinto
ruta_csv = "Absorbancia_VeMN7_5g_limpio.csv"  # nombre del archivo CSV de salida

# Leer el archivo .txt completo
with open(ruta_txt, 'r') as f:
    lineas = f.readlines()

# Buscar el inicio de los datos (donde aparece 'nm A')
for i, linea in enumerate(lineas):
    if 'nm' in linea and 'A' in linea:
        inicio_datos = i + 1
        break

# Extraer solo los datos
datos = [linea.strip().split() for linea in lineas[inicio_datos:] if linea.strip()]

# Convertir a DataFrame
df = pd.DataFrame(datos, columns=['nm', 'A'])

# Convertir a tipo numérico (XXX.XXX se convertirá en NaN automáticamente)
df['nm'] = pd.to_numeric(df['nm'], errors='coerce')
df['A'] = pd.to_numeric(df['A'], errors='coerce')

# Guardar como CSV
df.to_csv(ruta_csv, index=False)
print(f'Archivo convertido exitosamente a: {ruta_csv}')

# %%
