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
datos=pd.read_csv("Transmitancia_VeMN22_5g_limpio.csv")

# %%
datos.info()

# %%
datos.head()

# %%
x=datos["nm"]
y=datos["%T"]

# %%
x

# %%
y

# %%
datos.hist(figsize=(5,5), bins=20, edgecolor="black")

# %%
type(x)

# %%
type(y)

# %%
x.describe()

# %%
y.describe()

# %%
import seaborn as sb
sb.scatterplot(x=x, y=y, data=datos, hue="%T", palette="coolwarm")

# %%
from scipy.signal import savgol_filter
import seaborn as sb
import matplotlib.pyplot as plt

#creamos una columna suavizada
datos["T_suave"] = savgol_filter(datos["%T"],window_length=19, polyorder=3)

#Graficamos scatterplot con sauvizado y gradiente de color 
plt.figure(figsize=(10,6))
sb.scatterplot(x="nm", y="T_suave", data=datos, hue="T_suave", palette="coolwarm", s=20, edgecolor=None)
plt.title("Transmitancia - Café de Veracruz (Muestra 22.5g - ratio 1:8.88)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Transmitancia (%T)")
plt.legend(title="Transmitancia", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
