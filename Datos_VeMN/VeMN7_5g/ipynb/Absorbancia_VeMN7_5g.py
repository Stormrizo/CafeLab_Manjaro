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
datos = pd.read_csv("Absorbancia_VeMN7_5g_limpio.csv")

# %%
datos.info()

# %%
datos.head()

# %%
x=datos["nm"]
y=datos["A"]

# %%
x

# %%
y

# %%
datos.hist(figsize=(5,5), bins=20, edgecolor="black")

# %%
import seaborn as sb
sb.scatterplot(x=x, y=y, data=datos, hue=y, palette="coolwarm")


# %%
y

# %%
from scipy.signal import savgol_filter
import seaborn as sb
import matplotlib.pyplot as plt

# Creamos una columna suavizada usando Savitzky–Golay filter

datos["A_suave"] = savgol_filter(datos["A"], window_length=19, polyorder=3)

#Graficamos el scatterplot con suavizado y gradiente de color

plt.figure(figsize=(10,6))
sb.scatterplot(x=x, y="A_suave", data=datos, hue="A_suave", palette="coolwarm", s=20, edgecolor=None )
plt.title("Absorbancia - Café de Veracruz (Muestra 7.5g - ratio 1:26.66 )", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia")
plt.legend(title="Absorbancia", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
for w in [11, 15, 19, 21]:
    datos["A_suave"] = savgol_filter(datos["A"], window_length=w, polyorder=3)
    plt.figure()
    sb.scatterplot(x='nm', y="A_suave", data=datos, hue="A_suave", palette="coolwarm", s=15)
    plt.title(f"Filtro con ventana = {w}")
    plt.show()


# %% [markdown]
#

# %%
