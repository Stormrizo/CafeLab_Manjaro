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

# Rutas de archivo
csv_base = "Transmitancia_VeMN18g.csv"
csv_limpio = "Transmitancia_VeMN18g_limpio.csv"

# Leer el archivo .csv completo
with open(csv_base, 'r') as f:
    lineas = f.readlines()

# Detectar línea de encabezado real (donde están los datos)
for i, linea in enumerate(lineas):
    if "nm" in linea and "%T" in linea:
        inicio_datos = i + 1
        break

# Extraer los datos desde ahí
datos = [linea.strip().split(',') for linea in lineas[inicio_datos:] if linea.strip()]

# Crear DataFrame
df = pd.DataFrame(datos, columns=["nm", "%T"])

# Convertir columnas a numérico (XXX.XXX se vuelve NaN automáticamente)
df["nm"] = pd.to_numeric(df["nm"], errors='coerce')
df["%T"] = pd.to_numeric(df["%T"], errors='coerce')

# Guardar como nuevo CSV limpio (NO se eliminan NaN)
df.to_csv(csv_limpio, index=False)
print(f"✅ Archivo limpio creado como: {csv_limpio}")


# %%
import pandas as pd

# %%
datos=pd.read_csv("Transmitancia_VeMN18g_limpio.csv")

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
x.value_counts()

# %%
y.value_counts()

# %%
import seaborn as sb
sb.scatterplot(x="nm", y="%T", data=datos, hue="%T", palette="coolwarm")

# %%
from scipy.signal import savgol_filter
import seaborn as sb
import matplotlib.pyplot as plt

# Crear columna suavizada
datos['T_suave'] = savgol_filter(datos['%T'], window_length=19, polyorder=3)

# Graficar scatterplot con suavizado y gradiente de color
plt.figure(figsize=(10, 6))
sb.scatterplot(x='nm', y='T_suave', data=datos, hue='T_suave', palette='coolwarm', s=20, edgecolor=None)
plt.title("Transmitancia - Café de Veracruz (Muestra 18g - ratio 1:11.11)", fontsize=14)
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Transmitancia (%)")
plt.legend(title="Transmitancia (%)", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
