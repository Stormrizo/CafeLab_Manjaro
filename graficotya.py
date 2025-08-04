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
#     display_name: ydata-env
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from ydata_profiling import ProfileReport

# Cargar archivos de absorción
file_suffix_A = ['AChMN', 'AGoMG', 'AMiMG', 'AOxMM', 'AIlly']

for file_name in file_suffix_A:
    file_path = f'Data/{file_name}.txt'
    try:
        file = pd.read_csv(file_path, engine='python', sep='\t', skiprows=8)
        file.iloc[:, 1] = pd.to_numeric(file['A'], errors='coerce')
        globals()[file_name] = file
        print(f"{file_name}     CARGADO")
    except FileNotFoundError:
        print(f"No se encontró el archivo {file_name}.txt")

    for porce in ['05', '20', '50']:
        try:
            file_path = f'Data/{file_name}{porce}.txt'
            file = pd.read_csv(file_path, engine='python', sep='\t', skiprows=8)
            file.iloc[:, 1] = pd.to_numeric(file['A'], errors='coerce')
            globals()[file_name + porce] = file
            print(f"{file_name+porce}     CARGADO")
        except FileNotFoundError:
            print(f"No se encontró el archivo {file_name+porce}.txt")


# %%
# Cargar archivos de transmitancia
file_suffix_T = ['TChMN', 'TGoMG', 'TMiMG', 'TOxMM', 'TIlly']

for file_name in file_suffix_T:
    file_path = f'Data/{file_name}.txt'
    try:
        file = pd.read_csv(file_path, engine='python', sep='\t', skiprows=8)
        file.iloc[:, 1] = pd.to_numeric(file['%T'], errors='coerce')
        globals()[file_name] = file
        print(f"{file_name}     CARGADO")
    except FileNotFoundError:
        print(f"No se encontró el archivo {file_name}.txt")

    for porce in ['05', '20', '50']:
        try:
            file_path = f'Data/{file_name}{porce}.txt'
            file = pd.read_csv(file_path, engine='python', sep='\t', skiprows=8)
            file.iloc[:, 1] = pd.to_numeric(file['%T'], errors='coerce')
            globals()[file_name + porce] = file
            print(f"{file_name+porce}     CARGADO")
        except FileNotFoundError:
            print(f"No se encontró el archivo {file_name+porce}.txt")


# %%
AIlly.head()


# %%
TIlly.head()

# %%
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

cmap_A = cm.get_cmap('tab10', len(file_suffix_A))
cmap_T = cm.get_cmap('Set1', len(file_suffix_T))

# Absorbancia
for i, name in enumerate(file_suffix_A):
    df = globals()[name]
    ax1.plot(df['nm'], df['A'], color=cmap_A(i), label=name)

ax1.set_title("Absorbancia", fontsize=20)
ax1.set_xlabel('Longitud de Onda (nm)')
ax1.set_ylabel('Absorbancia')
ax1.legend(loc='upper right')

# Transmitancia
for i, name in enumerate(file_suffix_T):
    df = globals()[name]
    ax2.plot(df['nm'], df['%T'], color=cmap_T(i), label=name)

ax2.set_title("Transmitancia", fontsize=20)
ax2.set_xlabel('Longitud de Onda (nm)')
ax2.set_ylabel('Transmitancia (%)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()


# %%
