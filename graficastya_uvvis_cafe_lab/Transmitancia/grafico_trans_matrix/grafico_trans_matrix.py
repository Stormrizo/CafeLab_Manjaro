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
from datetime import datetime
import os

# === Definir regiones espectrales por nombre y rangos (en nm) ===
regiones = {
    '100–200 nm': (100, 200),
    '200–300 nm': (200, 300),
    '300–400 nm': (300, 400),
    '400–500 nm': (400, 500),
    '500–600 nm': (500, 600),
    '600–750 nm': (600, 750),
    '750–800 nm': (750, 800),
    '800–900 nm': (800, 900),
    '900–1000 nm': (900, 1000),
}

# === Buscar todos los archivos que terminan con '_matrix.csv' ===
archivos_csv = [f for f in os.listdir() if f.endswith('_matrix.csv')]

# === Iterar sobre cada archivo para generar su reporte ===
for archivo in archivos_csv:
    df = pd.read_csv(archivo)

    # === Limpiar valores no numéricos y filtrar datos extremos razonables (%T: 0–100) ===
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where((df[col] >= 0) & (df[col] <= 100))

    df_clean = df.copy()
    resultados = []

    for col in df_clean.columns[1:]:
        datos = df_clean[['nm', col]].dropna()

        if datos.empty:
            print(f"⚠️  Columna vacía o inválida en {archivo}: {col}, se omite.")
            continue

        nm = datos['nm'].values
        transmitancia = datos[col].values

        max_val = np.max(transmitancia)
        min_val = np.min(transmitancia)
        avg = np.mean(transmitancia)
        std = np.std(transmitancia)
        auc = np.trapezoid(transmitancia, nm)
        lambda_max = nm[np.argmax(transmitancia)]
        lambda_min = nm[np.argmin(transmitancia)]

        promedios_regiones = {}
        for nombre, (a, b) in regiones.items():
            datos_region = datos[(datos['nm'] >= a) & (datos['nm'] <= b)][col]
            promedio = np.mean(datos_region) if not datos_region.empty else np.nan
            promedios_regiones[nombre] = (
                f"{promedio:.6f}" if not np.isnan(promedio) else "—"
            )

        resultados.append({
            "Muestra": col,
            "Máx. Trans.": f"{max_val:.3f}",
            "Mín. Trans.": f"{min_val:.3f}",
            "Promedio": f"{avg:.3f}",
            "Desv. Estándar": f"{std:.3f}",
            "AUC": f"{auc:.2f}",
            "λ máx (nm)": f"{lambda_max:.1f}",
            "λ mín (nm)": f"{lambda_min:.1f}",
            **promedios_regiones
        })

    # === Crear tabla de métricas ===
    tabla = pd.DataFrame(resultados)
    if tabla.empty:
        print(f"No se generó reporte para {archivo}, todas las columnas estaban vacías.")
        continue

    tabla["Máx. Trans. (num)"] = pd.to_numeric(tabla["Máx. Trans."], errors='coerce')
    columnas_ordenadas = tabla.sort_values("Máx. Trans. (num)", ascending=False)["Muestra"]

    # === Crear gráfica Plotly ===
    fig = go.Figure()
    for col in columnas_ordenadas:
        datos_validos = df_clean[['nm', col]].dropna()
        fig.add_trace(go.Scatter(
            x=datos_validos['nm'],
            y=datos_validos[col],
            mode='lines',
            name=col
        ))

    nombre_base = archivo.replace('_matrix.csv', '')
    titulo = f"Curvas de transmitancia – Café {nombre_base}"

    fig.update_layout(
        title=titulo,
        xaxis_title="Longitud de onda (nm)",
        yaxis_title="Transmitancia (%T)",
        legend_title="Muestras (gramos)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # === Generar HTML ===
    fecha = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    grafica_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={"responsive": True})
    tabla_html = tabla.drop(columns=["Máx. Trans. (num)"]).to_html(index=False, classes="styled-table", border=0)

    html_final = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <title>Reporte de transmitancia UV-Vis</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{
          font-family: 'Segoe UI', Tahoma, sans-serif;
          margin: 40px;
          background-color: #fdfdfd;
          color: #222;
        }}
        h1 {{ color: #111; }}
        h2 {{ margin-top: 40px; }}
        .info {{ margin-bottom: 20px; }}
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

      <h1>{titulo}</h1>
      <div class="info"><p><strong>Fecha:</strong> {fecha}</p></div>

      {grafica_html}

      <h2>Métricas espectrales por muestra</h2>
      {tabla_html}

      <div class="lab">
        <strong>Nombre del proyecto: CafeLab</strong><br><br>
        <strong>Colaboradores:</strong><br>
        Martín Rodolfo Palomino Merino – Profesor investigador (FCFM-BUAP)<br>
        Lizeth Jazmín Orozco García – Colaboradora principal<br>
        Julio Alfredo Ballinas García – Colaborador del proyecto
      </div>
    </body>
    </html>
    """

    # Guardar el archivo
    nombre_salida = f"reporte_transmitancia_{nombre_base}.html"
    with open(nombre_salida, "w", encoding="utf-8") as f:
        f.write(html_final)

    print(f"✅ Reporte generado: {nombre_salida}")



# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# === Preparación de archivos CSV en el mismo nivel ===
archivos_csv = [f for f in os.listdir() if f.endswith("_matrix.csv")]

# === Regiones espectrales UV-Vis extendidas ===
regiones = {
    '100–200 nm': (100, 200),
    '200–300 nm': (200, 300),
    '300–400 nm': (300, 400),
    '400–500 nm': (400, 500),
    '500–600 nm': (500, 600),
    '600–700 nm': (600, 700),
    '700–800 nm': (700, 800)
}

# === Inicializar figura Plotly
fig = go.Figure()
fig.update_layout(
    title="Comparativa de curvas de transmitancia",
    xaxis_title="Longitud de onda (nm)",
    yaxis_title="Transmitancia (%T)",
    legend_title="Curva",
    template="plotly_white",
    margin=dict(l=40, r=40, t=80, b=40),
)

# === Colores únicos para todas las curvas
total_curvas = 0
curvas_info = {}
for archivo in archivos_csv:
    df = pd.read_csv(archivo)
    col_nm = [c for c in df.columns if 'nm' in c.lower()][0]
    col_trans = [c for c in df.columns if c != col_nm]
    curvas_info[archivo] = (df, col_nm, col_trans)
    total_curvas += len(col_trans)

colormap = plt.get_cmap('tab20')(np.linspace(0, 1, total_curvas))
colores_hex = [mcolors.to_hex(c) for c in colormap]

# === Extraer métricas por curva
metricas = []
idx_color = 0

for archivo, (df, col_nm, columnas) in curvas_info.items():
    for col in columnas:
        datos = df[[col_nm, col]].dropna().rename(columns={col_nm: 'nm', col: 'transmitancia'})
        datos['transmitancia'] = pd.to_numeric(datos['transmitancia'], errors='coerce')
        datos = datos.dropna()
        if datos.empty:
            continue

        fig.add_trace(go.Scatter(
            x=datos['nm'],
            y=datos['transmitancia'],
            mode='lines',
            name=col,
            line=dict(color=colores_hex[idx_color])
        ))

        max_val = datos['transmitancia'].max()
        min_val = datos['transmitancia'].min()
        avg = datos['transmitancia'].mean()
        std = datos['transmitancia'].std()
        auc = np.trapezoid(datos['transmitancia'], datos['nm'])
        lambda_max = datos['nm'][datos['transmitancia'].idxmax()]
        lambda_min = datos['nm'][datos['transmitancia'].idxmin()]

        proms_reg = {}
        for region, (a, b) in regiones.items():
            reg_vals = datos.loc[(datos['nm'] >= a) & (datos['nm'] <= b), 'transmitancia']
            proms_reg[region] = f"{reg_vals.mean():.2f}" if not reg_vals.empty else "—"

        metricas.append({
            'Color de curva': colores_hex[idx_color],
            'Curva': col,
            'Máx (%T)': round(max_val, 2),
            'Mín (%T)': round(min_val, 2),
            'Promedio (%)': round(avg, 2),
            'Desv. Std': round(std, 2),
            'AUC': round(auc, 2),
            'λ máx (nm)': round(lambda_max, 2),
            'λ mín (nm)': round(lambda_min, 2),
            **proms_reg
        })

        idx_color += 1

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

# === Generar HTML final
fecha = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
grafica_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

html_final = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Comparativa de curvas de transmitancia UV-Vis</title>
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

  <h1>Curvas de transmitancia - CafeLab</h1> 

  <div class="info">
    <p><strong>Fecha:</strong> {fecha}</p>
  </div>

  {grafica_html}

  <h2>Métricas espectrales por curva</h2>
  {tabla_html}

  <div>
    <p>Las curvas de transmitancia permiten identificar la transparencia relativa de la muestra a lo largo del espectro. Una menor transmitancia puede sugerir absorción fuerte en ciertas regiones, lo cual es útil para análisis de compuestos activos presentes en el café.</p>
  </div>

  <div class="lab">
    <strong>Nombre del proyecto: CafeLab</strong><br><br>
    <strong>Colaboradores:</strong><br>
    Martín Rodolfo Palomino Merino – Profesor investigador, jefe responsable del laboratorio de caracterización de materiales (FCFM-BUAP).<br>
    Lizeth Jazmín Orozco García – Colaborador principal.<br>
    Julio Alfredo Ballinas García – Colaborador del proyecto.
  </div>

</body>
</html>
"""

# === Guardar archivo HTML
with open("Reporte_transmitancia.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("✅ HTML generado: Reporte_transmitancia.html")
print("Rango de longitudes de onda:", df['nm'].min(), "a", df['nm'].max())
fig.show()

# %%
