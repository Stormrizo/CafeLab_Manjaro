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
datos = pd.read_csv("Absorbancia_Illy_limpio.csv")

# %%
datos.head()

# %%
datos.info()

# %%
datos.hist(figsize=(5,5), bins=30, edgecolor="black")

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
y.value_counts()

# %%
import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(10,6))
sb.scatterplot(x=x, y=y, data=datos, hue="A", palette="coolwarm", s=50, edgecolor="none")
plt.title("Absorbancia - Café Illy´s")
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia (A)")
plt.legend(title="A", loc="best")
plt.grid(None)
plt.show()

# %%
#Vamos a usar el filtro de Savitzky-Golay para suavizar la curva

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sb

#Creamos la columna suavizada para los datos de absorbancia

datos["A_suavizada"] = savgol_filter(datos["A"], window_length=19, polyorder=3)
plt.figure(figsize=(10,6))
sb.scatterplot(x=x, y="A_suavizada", data=datos, hue="A_suavizada", palette="coolwarm", s=50, edgecolor="none")
plt.title("Absorbancia - Café Illy´s")
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia (A)")
plt.legend(title="A", loc="best")
plt.grid(None)
plt.show()


# %%
