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
datos = pd.read_csv("Transmitancia_VeMN15g_limpio.csv")

# %%
datos.info()

# %%
datos.head()

# %%
x=datos["nm"]
y=datos["%T"]

# %%
x.value_counts()

# %%
y.value_counts()


# %%
type(x)

# %%
type(y)

# %%
datos.head()

# %%
datos.hist(figsize=(5,5), bins=40, grid=False, edgecolor="black", linewidth=1.2, color="green", alpha=0.7)

# %%
import matplotlib.pyplot as plt
import seaborn as sb

#graficamos los datos
plt.figure(figsize=(10,6))
sb.scatterplot(x=x, y=y, data=datos, hue="%T", palette="coolwarm", s=20, edgecolor=None)
plt.title("Transmitancia - Café de Veracruz (Muestra 15g - ratio 1:13.24)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Transmitancia (%T)")
plt.legend(title="%T", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sb

# creamos columna suavizada

datos["T_suave"] = savgol_filter(datos["%T"], window_length=19, polyorder=3)

# creamos scatterplot con columna suavizada de datos de %T y gradiente de color

plt.figure(figsize=(10,6))
sb.scatterplot(x=x, y="T_suave", data=datos, hue="T_suave", palette="coolwarm", s=20, edgecolor=None)
plt.title("Transmitancia - Café de Veracruz (Muestra 15g - ratio 1:13.24)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Transmitancia (%T)")
plt.legend(title="%T", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()



# %%

# %%
