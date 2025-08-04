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
datos= pd.read_csv("ACh.csv")

# %%
datos.head()

# %%
datos.info()

# %%
import os

print("Directorio actual:", os.getcwd())
print("Archivos en Data/:", os.listdir("Data"))



# %%
os.listdir(".")  # El punto representa el directorio actual



# %%
