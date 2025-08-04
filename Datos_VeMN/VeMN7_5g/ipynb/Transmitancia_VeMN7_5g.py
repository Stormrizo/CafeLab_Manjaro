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
datos=pd.read_csv("Transmitancia_VeMN7_5g_limpio.csv")

# %%
datos.info()

# %%
datos.head()

# %%
x=datos["nm"]

# %%
y=datos["%T"]

# %%
x

# %%
y

# %%
import seaborn as sb
sb.scatterplot(x=x,y=y, data=datos, hue=y, palette="coolwarm")

# %%
from scipy.signal import savgol_filter
import seaborn as sb
import matplotlib.pyplot as plt


# %%
#Creamos una columna suavizada
datos["T_suavizada"]=savgol_filter(datos["%T"], window_length=19, polyorder=3)

# %%
#Graficamos el scatterplot con suavizado y gradiente de color
plt.figure(figsize=(10,6))
sb.scatterplot(x=x, y="T_suavizada", data=datos, hue="T_suavizada", palette="coolwarm", s=20, edgecolor=None)
plt.title("Transmitancia - Café de Veracruz (Muestra 7.5g - ratio 1:26.66)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Transmitancia (%)")
plt.legend(title="Transmitancia (%)", loc='best')
plt.grid(True)
plt.show()

# %%
