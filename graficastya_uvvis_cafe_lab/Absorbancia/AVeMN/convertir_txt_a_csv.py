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

# %% [markdown]
# # Conversión de AVeMN7_5g.txt a CSV
# Este notebook convierte el archivo original de absorbancia de Veracruz (molienda normal, 7.5g) de `.txt` a `.csv` sin pérdida de información.
#

# %%
import pandas as pd


# %%
ruta_txt = 'AVeMN7_5g.txt'
ruta_csv = 'AVeMN7_5g.csv'


# %%
try:
    df = pd.read_csv(ruta_txt, sep='\t', engine='python')  # sin skiprows
    df.to_csv(ruta_csv, index=False)
    print(f"✅ Conversión completa. CSV guardado en: {ruta_csv}")
except Exception as e:
    print(f"❌ Error durante la conversión: {e}")


# %%
