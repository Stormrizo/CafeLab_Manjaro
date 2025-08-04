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

csv_base = "Absorbancia_hueco_VeMN22_5g.csv"
csv_limpio = "Absorbancia_hueco_VeMN22_5g_limpio.csv"

# Lee el archivo .csv completo
with open(csv_base, 'r') as f:
    lineas = f.readlines()

# Detectar línea de encabezado real (donde están los datos)
for i, linea in enumerate(lineas):
    if "nm" in linea and "A" in linea:
        inicio_datos = i + 1
        break

# Extrae los datos desde ahí
datos = [linea.strip().split(',') for linea in lineas[inicio_datos:] if linea.strip()]

# Crear el DataFrame a usar 
df = pd.DataFrame(datos, columns=["nm", "A"])

# Convierte las columnas a int o float (XXX.XXX se vuelven NaN (not a number) automáticamente)
df["nm"] = pd.to_numeric(df["nm"], errors='coerce')
df["A"] = pd.to_numeric(df["A"], errors='coerce')

# se guarda como nuevo CSV limpio (NO se eliminan NaN)
df.to_csv(csv_limpio, index=False)
print(f"Archivo limpio creado como: {csv_limpio}")

# %%
