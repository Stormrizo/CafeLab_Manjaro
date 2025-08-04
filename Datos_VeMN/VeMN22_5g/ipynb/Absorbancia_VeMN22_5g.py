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

# %%
datos = pd.read_csv("Absorbancia_VeMN22_5g_limpio.csv")

# %%
datos.info()

# %%
datos.head()

# %%
datos.hist(figsize=(5,5), bins=20, edgecolor="black")

# %%
x=datos["nm"]
y=datos["A"]

# %%
x


# %%
y

# %%
x.value_counts()

# %%
x.values

# %%
type(x)

# %%
type(y.value_counts())

# %%
type(y)

# %%
y.values

# %%
qué hace? y.values.shape


# %%
y.values.shape

# %%
import seaborn as sb
sb.scatterplot(x=x,y=y, data=datos, hue="A", palette="coolwarm")

# %%
from scipy.signal import savgol_filter
import seaborn as sb
import matplotlib.pyplot as plt

# Crear columna suavizada
datos['A_suave'] = savgol_filter(datos['A'], window_length=19, polyorder=3)

# Graficar scatterplot con suavizado y gradiente de color
plt.figure(figsize=(10, 6))
sb.scatterplot(x='nm', y='A_suave', data=datos, hue='A_suave', palette='coolwarm', s=20, edgecolor=None)
plt.title("Absorbancia - Café de Veracruz (Muestra 22.5g - ratio 1:8.88)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia")
plt.legend(title='Absorbancia', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
