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
import csv

archivo_txt = 'AIlly.txt'
archivo_csv = 'AIlly.csv'

with open(archivo_txt, 'r', encoding='utf-8') as f:
    lineas = f.readlines()

salida_csv = []

for linea in lineas:
    linea = linea.strip()
    if not linea:
        continue  # opcional: ignorar líneas completamente vacías

    # Separar por espacios múltiples o tabulaciones (solo una vez)
    partes = linea.split(None, 1)
    if len(partes) == 2:
        salida_csv.append([partes[0], partes[1]])
    else:
        salida_csv.append([linea])  # línea completa sin separación

# Reescribir con comas sin añadir líneas extras
with open(archivo_csv, 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    for fila in salida_csv:
        writer.writerow(fila)

print(f"Archivo '{archivo_csv}' creado y separado por comas.")





# %%
archivo_txt = 'Línea_base_absorbancia.txt'
archivo_csv = 'Línea_base_absorbancia.csv'

with open(archivo_txt, 'r', encoding='utf-8') as f:
    lineas = f.readlines()

salida_csv = []

for linea in lineas:
    linea = linea.strip()
    if not linea:
        continue  # opcional: ignorar líneas completamente vacías

    # Separar por espacios múltiples o tabulaciones (solo una vez)
    partes = linea.split(None, 1)
    if len(partes) == 2:
        salida_csv.append([partes[0], partes[1]])
    else:
        salida_csv.append([linea])  # línea completa sin separación

# Reescribir con comas sin añadir líneas extras
with open(archivo_csv, 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    for fila in salida_csv:
        writer.writerow(fila)

print(f"Archivo '{archivo_csv}' creado y separado por comas.")

# %%
archivo_txt = 'Línea_base_transmitancia.txt'
archivo_csv = 'Línea_base_transmitancia.csv'

with open(archivo_txt, 'r', encoding='utf-8') as f:
    lineas = f.readlines()

salida_csv = []

for linea in lineas:
    linea = linea.strip()
    if not linea:
        continue  # opcional: ignorar líneas completamente vacías

    # Separar por espacios múltiples o tabulaciones (solo una vez)
    partes = linea.split(None, 1)
    if len(partes) == 2:
        salida_csv.append([partes[0], partes[1]])
    else:
        salida_csv.append([linea])  # línea completa sin separación

# Reescribir con comas sin añadir líneas extras
with open(archivo_csv, 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    for fila in salida_csv:
        writer.writerow(fila)

print(f"Archivo '{archivo_csv}' creado y separado por comas.")

# %%
