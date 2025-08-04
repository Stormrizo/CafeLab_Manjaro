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
datos = pd.read_csv("Absorbancia_VeMN15g_limpio.csv")

# %%
datos.head()

# %%
datos.info()

# %%
x=datos["nm"]
y=datos["A"]

x
y

# %%
type(x)

# %%
type(y)

# %%
y

# %%
x


# %%
x.describe()

# %%
y.describe()

# %%
datos.hist(figsize=(5,5), bins=20, edgecolor="black", grid=False)

# %%
import seaborn as sb 
sb.scatterplot(x=x, y=y, data=datos, hue="A", palette="coolwarm")

# %%
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sb

# creamos columna suavizada

datos["A_suave"]=savgol_filter(datos["A"], window_length=19, polyorder=3)

#graficamos el scatterplot con suavizado y gradiente de color

plt.figure(figsize=(10,6))
sb.scatterplot(x="nm", y="A_suave", data =datos, hue="A_suave", palette="coolwarm", s=20, edgecolor=None)
plt.title("Absorbancia - Café de Veracruz (Muestra 15g - ratio 1:13.24)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia")
plt.legend(title='Absorbancia', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
