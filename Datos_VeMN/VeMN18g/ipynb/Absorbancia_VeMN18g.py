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

datos = pd.read_csv("ABS_VEMN_18G.csv", skiprows=8)

# %%
#primeros 5 datos
datos.head()

# %%
datos.info()

# %%
import seaborn as sb
print(sb.__version__)

# %%
sb.scatterplot(x="nm", y="A", data=datos, hue="A", palette="coolwarm")

# %%
x=datos["nm"]
y=datos["A"]

# %%
print(x)


# %%
print(y)

# %%
type(x)

# %%
type(y)

# %%
type(x.values)

# %%
type(y.values)

# %%
x.values

# %%
y.values 

# %%
x.value_counts()

# %%
y.value_counts()

# %%
datos.info()

# %%
y.info()

# %%
datos.describe()

# %%
y.describe()

# %%
x.describe()

# %%
x.hist()

# %%
datos.hist()

# %%
datos.hist(figsize=(15,8), bins=30, edgecolor="black")

 # %%
 #graficas agradables
import seaborn as sb
sb.scatterplot(x="nm", y="A", data=datos[(datos.nm > 2)], hue="A", palette="coolwarm") 
#con hue el color de cada uno de los puntos depende de el valor de A)

# %%
#mejoramos el set de datos
#usamos la función drop.na
datos.info()
datos.head()

# %%
y.info()

# %%
y.describe()

# %%
x.describe()

# %%
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# importamos en csv datos ignorando las primeras 8 filas porque son encabezados
datos = pd.read_csv("ABS_VEMN_18G.csv", skiprows=8)

#ntenta convertir todos los valores de la columna 'A' a números (float o int). Si hay alguno que no se puede convertir (como 'XXX.XXX'), ponlo como NaN (valor nulo'
datos["A"] = pd.to_numeric(datos["A"], errors='coerce')
#usar errors='coerce', esos errores se "suavizan" convirtiendo

#eliminar cualquier fila donde 'A' no pudo convertirse y quedó como NaN
datos = datos.dropna(subset=["A"])

#filtrar valores absurdos (por ejemplo, nm > 2 parece raro para espectroscopía UV-VIS)
datos = datos[datos["nm"] > 100]  # O el rango que te interese, como > 300 si es NIR

# 🎨 Graficar
plt.figure(figsize=(10, 6))
sb.scatterplot(x="nm", y="A", data=datos, hue="A", palette="coolwarm")
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia")
plt.title("Espectro de Absorbancia del Café")
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sb

# Configurar el tamaño y estilo
plt.figure(figsize=(10, 6))
sb.set(style="whitegrid")

# Gráfica
sb.scatterplot(
    x="nm", y="A", data=datos, hue="A", palette="coolwarm", edgecolor=None
)

# Etiquetas
plt.title("Espectro de Absorbancia del Café", fontsize=16)
plt.xlabel("Longitud de onda (nm)", fontsize=12)
plt.ylabel("Absorbancia", fontsize=12)
plt.legend(title="A", loc="best")
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.signal import savgol_filter

# Leer el archivo (ajusta el nombre si es necesario)
datos = pd.read_csv("ABS_VEMN_18G.csv", skiprows=8)

# Convertir la columna 'A' a tipo numérico, forzando errores como NaN
datos["A"] = pd.to_numeric(datos["A"], errors='coerce')

# Eliminar filas con NaN (por ejemplo, donde 'A' era 'XXX.XXX')
datos = datos.dropna(subset=["A"])

# Aplicar suavizado Savitzky-Golay
datos["A_suave"] = savgol_filter(datos["A"], window_length=19, polyorder=3)

# Configurar estilo
sb.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Gráfica con gradiente de color
sb.scatterplot(
    x="nm", 
    y="A_suave", 
    data=datos, 
    hue="A_suave", 
    palette="coolwarm", 
    s=20, 
    edgecolor=None
)

# Etiquetas
plt.title("Absorbancia - Café de Veracruz (Muestra 18g - ratio 1:11.11)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Absorbancia")
plt.legend(title="Absorbancia", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
