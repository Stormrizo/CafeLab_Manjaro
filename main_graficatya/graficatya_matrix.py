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
#     display_name: ydata-env
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# === Archivos ===
archivos_abs = [
    'AChMN_matrix.csv', 'AGoMG_matrix.csv', 'AIlly_matrix.csv',
    'AMiMG_matrix.csv', 'AOxMM_matrix.csv', 'AVeMN_matrix.csv'
]
archivos_trans = [
    'TChMN_matrix.csv', 'TGoMG_matrix.csv', 'TIlly_matrix.csv',
    'TMiMG_matrix.csv', 'TOxMM_matrix.csv', 'TVeMN_matrix.csv'
]

# === Función para leer datos y limpiar ===
def leer_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['nm'] + list(df.columns[1:])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

# === Diccionarios de datos ===
datos_abs = {nombre[:-11]: leer_csv(nombre) for nombre in archivos_abs}
datos_trans = {nombre[:-11]: leer_csv(nombre) for nombre in archivos_trans}

# === Crear figura interactiva con subplots ===
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Curvas de Absorbancia", "Curvas de Transmitancia"),
    shared_xaxes=True
)

# === Colores para distinguir ===
colores = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# === Gráficas de absorbancia ===
for i, (nombre, df) in enumerate(datos_abs.items()):
    for col in df.columns[1:]:
        fig.add_trace(go.Scatter(
            x=df['nm'],
            y=df[col],
            mode='lines',
            name=f"{nombre} - {col}",
            line=dict(color=colores[i % len(colores)])
        ), row=1, col=1)

# === Gráficas de transmitancia ===
for i, (nombre, df) in enumerate(datos_trans.items()):
    for col in df.columns[1:]:
        fig.add_trace(go.Scatter(
            x=df['nm'],
            y=df[col],
            mode='lines',  # Línea continua, no puntos
            name=f"{nombre} - {col}",
            line=dict(color=colores[i % len(colores)])
        ), row=2, col=1)


# === Estilo general ===
fig.update_layout(
    height=800,
    width=1000,
    title_text="Curvas de Absorbancia y Transmitancia - Café",
    xaxis_title="Longitud de onda (nm)",
    yaxis_title="Absorbancia / Transmitancia",
    legend_title="Muestras",
    template="plotly_white"
)

# === Guardar como archivo HTML ===
fig.write_html("curvas_cafe_interactivas.html")
fig.show()


# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# === Archivos ===
archivos_abs = [
    'AChMN_matrix.csv', 'AGoMG_matrix.csv', 'AIlly_matrix.csv',
    'AMiMG_matrix.csv', 'AOxMM_matrix.csv', 'AVeMN_matrix.csv'
]
archivos_trans = [
    'TChMN_matrix.csv', 'TGoMG_matrix.csv', 'TIlly_matrix.csv',
    'TMiMG_matrix.csv', 'TOxMM_matrix.csv', 'TVeMN_matrix.csv'
]

# === Regiones espectrales ===
regiones = {
    '100–200 nm': (100, 200),
    '200–300 nm': (200, 300),
    '300–400 nm': (300, 400),
    '400–500 nm': (400, 500),
    '500–600 nm': (500, 600),
    '600–700 nm': (600, 700),
    '700–800 nm': (700, 800)
}

# === Ruta base relativa
ruta_base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# === Función para leer datos y limpiar ===
def leer_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['nm'] + list(df.columns[1:])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

# === Diccionarios de datos ===
datos_abs = {nombre[:-11]: leer_csv(os.path.join(ruta_base, nombre)) for nombre in archivos_abs}
datos_trans = {nombre[:-11]: leer_csv(os.path.join(ruta_base, nombre)) for nombre in archivos_trans}

# === Colores (hex) para las curvas ===
colores_plotly = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
]

# === Figura principal
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Curvas de absorbancia", "Curvas de transmitancia"),
    shared_xaxes=False,  # ✅ Cada gráfico tiene su eje x propio
    shared_yaxes=False,   # Opcional: también su eje y
    vertical_spacing=0.1  # puedes ajustar a 0.03 si lo quieres aún más compacto
)


metricas = []

# === Absorbancia
for i, (nombre, df) in enumerate(datos_abs.items()):
    for col in df.columns[1:]:
        datos = df[['nm', col]].dropna().rename(columns={col: 'valor'})
        fig.add_trace(go.Scatter(
            x=datos['nm'],
            y=datos['valor'],
            mode='lines',
            name=f"{nombre} - {col}",
            line=dict(color=colores_plotly[i % len(colores_plotly)])
        ), row=1, col=1)

        # === Métricas
        max_val = datos['valor'].max()
        min_val = datos['valor'].min()
        avg = datos['valor'].mean()
        std = datos['valor'].std()
        auc = np.trapezoid(datos['valor'], datos['nm'])
        lambda_max = datos['nm'][datos['valor'].idxmax()]
        lambda_min = datos['nm'][datos['valor'].idxmin()]

        proms_reg = {}
        for region, (a, b) in regiones.items():
            reg_vals = datos.loc[(datos['nm'] >= a) & (datos['nm'] <= b), 'valor']
            proms_reg[region] = f"{reg_vals.mean():.2f}" if not reg_vals.empty else "—"

        metricas.append({
            'Tipo': 'Absorbancia',
            'Color de curva': colores_plotly[i % len(colores_plotly)],
            'Curva': f"{nombre} - {col}",
            'Máx': round(max_val, 2),
            'Mín': round(min_val, 2),
            'Promedio': round(avg, 2),
            'Desv. Std': round(std, 2),
            'AUC': round(auc, 2),
            'λ máx (nm)': round(lambda_max, 2),
            'λ mín (nm)': round(lambda_min, 2),
            **proms_reg
        })

# === Transmitancia
for i, (nombre, df) in enumerate(datos_trans.items()):
    for col in df.columns[1:]:
        datos = df[['nm', col]].dropna().rename(columns={col: 'valor'})
        fig.add_trace(go.Scatter(
            x=datos['nm'],
            y=datos['valor'],
            mode='lines',
            name=f"{nombre} - {col}",
            line=dict(color=colores_plotly[i % len(colores_plotly)])
        ), row=2, col=1)

        # === Métricas
        max_val = datos['valor'].max()
        min_val = datos['valor'].min()
        avg = datos['valor'].mean()
        std = datos['valor'].std()
        auc = np.trapezoid(datos['valor'], datos['nm'])
        lambda_max = datos['nm'][datos['valor'].idxmax()]
        lambda_min = datos['nm'][datos['valor'].idxmin()]

        proms_reg = {}
        for region, (a, b) in regiones.items():
            reg_vals = datos.loc[(datos['nm'] >= a) & (datos['nm'] <= b), 'valor']
            proms_reg[region] = f"{reg_vals.mean():.2f}" if not reg_vals.empty else "—"

        metricas.append({
            'Tipo': 'Transmitancia',
            'Color de curva': colores_plotly[i % len(colores_plotly)],
            'Curva': f"{nombre} - {col}",
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

fig.update_layout(
    title="Curvas de absorbancia y transmitancia - Muestras de café",
    title_font_size=24,
    title_x=0.05,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=14, color="black"),
    margin=dict(l=80, r=50, t=80, b=60),
    height=1300,
)


# Etiquetas de los ejes
fig.update_xaxes(
    title_text="Longitud de onda (nm)",
    title_font=dict(size=16),
    tickfont=dict(size=12),
    showline=True,
    linecolor='black',
    mirror=True
)

fig.update_yaxes(
    title_text="Absorbancia",
    title_font=dict(size=16),
    tickfont=dict(size=12),
    showline=True,
    linecolor='black',
    mirror=True,
    row=1, col=1
)

fig.update_yaxes(
    title_text="Transmitancia (%)",
    title_font=dict(size=16),
    tickfont=dict(size=12),
    showline=True,
    linecolor='black',
    mirror=True,
    row=2, col=1
)

fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')



# === HTML Final
fecha = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
grafica_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

html_final = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte UV-Vis - Café</title>
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

  <h1>Curvas de absorbancia y transmitancia</h1> 

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
ruta_salida = os.path.join(ruta_base, "Reporte_Cafe_UVVis.html")
with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(html_final)

print(f"Archivo HTML guardado en: {ruta_salida}")
print("Rango de longitudes de onda:", df['nm'].min(), "a", df['nm'].max())
fig.show()
