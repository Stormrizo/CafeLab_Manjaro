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
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Leer todos los CSV del directorio
archivos_csv = [f for f in os.listdir() if f.endswith('_data.csv')]

# Definir regiones espectrales
regiones = {
    '300–400 nm': (300, 400),
    '400–500 nm': (400, 500),
    '500–600 nm': (500, 600),
    '600–700 nm': (600, 700),
    '700–800 nm': (700, 800),
    '800–900 nm': (800, 900)
}

# Procesar cada archivo
for archivo in archivos_csv:
    nombre_base = archivo.replace("_data.csv", "")
    df = pd.read_csv(archivo)
    
    # Limpieza de datos: convertir a numérico y eliminar NaNs
    df["CPS"] = pd.to_numeric(df["CPS"], errors="coerce")
    df = df.dropna()

    nm = df["nm"].values
    cps = df["CPS"].values

    # Métricas
    max_val = np.max(cps)
    min_val = np.min(cps)
    avg = np.mean(cps)
    std = np.std(cps)
    auc = np.trapezoid(cps, nm)
    lambda_max = nm[np.argmax(cps)]
    lambda_min = nm[np.argmin(cps)]

    promedios_regiones = {}
    for nombre, (a, b) in regiones.items():
        datos_region = df[(df["nm"] >= a) & (df["nm"] <= b)]["CPS"]
        promedio = np.mean(datos_region) if not datos_region.empty else np.nan
        promedios_regiones[nombre] = promedio

    # Tabla resumen
    tabla = pd.DataFrame([{
        "Muestra": nombre_base,
        "Máx. CPS": round(max_val, 3),
        "Mín. CPS": round(min_val, 3),
        "Promedio": round(avg, 3),
        "Desv. Estándar": round(std, 3),
        "AUC": round(auc, 2),
        "λ máx (nm)": round(lambda_max, 1),
        "λ mín (nm)": round(lambda_min, 1),
        **{k: round(v, 3) if not np.isnan(v) else None for k, v in promedios_regiones.items()}
    }])

    display(tabla)

    # Gráfica interactiva
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["nm"],
        y=df["CPS"],
        mode='lines',
        name=nombre_base
    ))
    fig.update_layout(
        title=f"Espectro de fluorescencia - {nombre_base}",
        xaxis_title="Longitud de onda (nm)",
        yaxis_title="Intensidad (CPS)",
        template="plotly_white"
    )
    fig.show()



# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# === Ruta base relativa
ruta_base = os.getcwd()

# === Archivos de fluorescencia
archivos_fluo = [f for f in os.listdir(ruta_base) if f.endswith("_data.csv")]

# === Regiones espectrales para fluorescencia
regiones = {
    '300–400 nm': (300, 400),
    '400–500 nm': (400, 500),
    '500–600 nm': (500, 600),
    '600–700 nm': (600, 700),
    '700–800 nm': (700, 800),
    '800–900 nm': (800, 900)
}

# === Función para leer y limpiar
def leer_fluo(filepath):
    df = pd.read_csv(filepath)
    df["CPS"] = pd.to_numeric(df["CPS"], errors="coerce")
    return df.dropna()

# === Colores para curvas
colores_plotly = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
]

# === Figura principal
fig = make_subplots(
    rows=1, cols=1,
    subplot_titles=("Curvas de fluorescencia",),
    shared_xaxes=False,
    shared_yaxes=False
)

metricas = []

for i, archivo in enumerate(archivos_fluo):
    nombre_base = archivo.replace("_data.csv", "")
    df = leer_fluo(os.path.join(ruta_base, archivo))

    fig.add_trace(go.Scatter(
        x=df["nm"],
        y=df["CPS"],
        mode='lines',
        name=nombre_base,
        line=dict(color=colores_plotly[i % len(colores_plotly)])
    ), row=1, col=1)

    max_val = df["CPS"].max()
    min_val = df["CPS"].min()
    avg = df["CPS"].mean()
    std = df["CPS"].std()
    auc = np.trapezoid(df["CPS"], df["nm"])
    lambda_max = df["nm"][df["CPS"].idxmax()]
    lambda_min = df["nm"][df["CPS"].idxmin()]

    proms_reg = {}
    for region, (a, b) in regiones.items():
        reg_vals = df.loc[(df["nm"] >= a) & (df["nm"] <= b), "CPS"]
        proms_reg[region] = f"{reg_vals.mean():.2f}" if not reg_vals.empty else "—"

    metricas.append({
        'Tipo': 'Fluorescencia',
        'Color de curva': colores_plotly[i % len(colores_plotly)],
        'Curva': nombre_base,
        'Máx': round(max_val, 2),
        'Mín': round(min_val, 2),
        'Promedio': round(avg, 2),
        'Desv. Std': round(std, 2),
        'AUC': round(auc, 2),
        'λ máx (nm)': round(lambda_max, 2),
        'λ mín (nm)': round(lambda_min, 2),
        **proms_reg
    })

# === Tabla HTML
df_metricas = pd.DataFrame(metricas)
tabla_html = "<table class='styled-table'><thead><tr>"
for col in df_metricas.columns:
    tabla_html += f"<th>{col}</th>"
tabla_html += "</tr></thead><tbody>"
for _, row in df_metricas.iterrows():
    tabla_html += "<tr>"
    for col in df_metricas.columns:
        if col == "Color de curva":
            tabla_html += f"<td style='background-color:{row[col]};'></td>"
        else:
            tabla_html += f"<td>{row[col]}</td>"
    tabla_html += "</tr>"
tabla_html += "</tbody></table>"

# === Configuración de layout
fig.update_layout(
    title="Curvas de fluorescencia - Muestras de café",
    title_font_size=24,
    title_x=0.05,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=14, color="black"),
    margin=dict(l=80, r=50, t=80, b=60),
    height=700,
    xaxis_title="Longitud de onda (nm)",
    yaxis_title="Intensidad (CPS)"
)

# === HTML Final
fecha = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
grafica_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

html_final = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte UV-Vis - Café (Fluorescencia)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: 'Segoe UI', Tahoma, sans-serif;
      margin: 40px;
      background-color: #fdfdfd;
      color: #222;
    }}
    h1 {{
      color: #111;
    }}
    h2 {{
      margin-top: 40px;
    }}
    .info {{
      margin-bottom: 20px;
    }}
    .lab {{
      margin-top: 40px;
      font-size: 0.95em;
      background-color: #f0f8ff;
      padding: 15px;
      border-left: 6px solid #009879;
    }}
    .styled-table {{
      border-collapse: collapse;
      margin-top: 20px;
      font-size: 1em;
      width: 100%;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
    }}
    .styled-table thead tr {{
      background-color: #009879;
      color: #ffffff;
      text-align: left;
    }}
    .styled-table th, .styled-table td {{
      padding: 12px 15px;
      border: 1px solid #ddd;
      text-align: center;
    }}
    .styled-table tbody tr:nth-child(even) {{
      background-color: #f3f3f3;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      height: auto !important;
    }}
  </style>
</head>
<body>

  <h1>Curvas de fluorescencia</h1> 

  <div class="info">
    <p><strong>Fecha:</strong> {fecha}</p>
  </div>

  {grafica_html}

  <h2>Métricas espectrales por curva</h2>
  {tabla_html}

  <div>
    <p>Las curvas permiten identificar regiones espectrales relevantes para el estudio de compuestos activos en muestras de café. Las métricas estadísticas complementan esta visualización proporcionando información clave sobre el comportamiento espectral en distintas regiones.</p>
  </div>

  <div class="lab">
    <strong>Nombre del proyecto: CafeLab</strong><br><br>
    <strong>Colaboradores:</strong><br>
    Martín Rodolfo Palomino Merino – Profesor investigador, jefe responsable del laboratorio de caracterización de materiales (FCFM-BUAP).<br>
    Lizeth Jazmín Orozco García – Colaboradora principal.<br>
    Julio Alfredo Ballinas García – Colaborador del proyecto.
  </div>

</body>
</html>
"""

# === Guardar archivo
ruta_salida = os.path.join(ruta_base, "Reporte_Cafe_Fluorescencia.html")
with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(html_final)

ruta_salida


print(f"Archivo HTML guardado en: {ruta_salida}")
print("Rango de longitudes de onda:", df['nm'].min(), "a", df['nm'].max())
fig.show()


# %%

# %%
