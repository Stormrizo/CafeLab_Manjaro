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

# %% [markdown]
# # Conversión de AIlly.txt a .csv
# Este notebook convierte los archivos originales de absorbancia de Café Illy - AIlly de `.txt` a `.csv` sin pérdida de información

# %%
import pandas as pd
import os

# Iteramos sobre todos los archivos .txt en el directorio actual
for archivo in os.listdir():
    if archivo.endswith('.txt'):
        try:
            nombre_csv = archivo.replace('.txt', '.csv')
            df = pd.read_csv(archivo, sep='\t', engine='python')
            df.to_csv(nombre_csv, index=False)
            print(f"✅ {archivo} convertido exitosamente a {nombre_csv}")
        except Exception as e:
            print(f"❌ Error al convertir {archivo}: {e}")
