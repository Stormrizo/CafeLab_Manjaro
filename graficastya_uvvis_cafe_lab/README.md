# Gráficas UV-Vis de muestras de café 

Este directorio agrupa los análisis espectrales de diferentes muestras de café mediante **espectroscopía UV-Vis**, organizados por tipo de dato (absorbancia y transmitancia). Forma parte del trabajo realizado en el proyecto **CafeLab** liderado por `Lizeth Jazmin Orozco` (colaborador principal), orientado a caracterizar compuestos presentes en el café en función de su región de origen, tipo de molienda y preparación

---

##  Estructura del directorio

- [`Absorbancia/`](./Absorbancia): Contiene espectros de absorbancia organizados por muestra. Se analizan los picos de absorción para identificar compuestos fenólicos, cafeína, ácidos clorogénicos, entre otros.
- [`Transmitancia/`](./Transmitancia): Sección destinada a los espectros de transmitancia (en desarrollo).

Cada subcarpeta dentro de estas secciones representa una muestra codificada por origen y tipo de molienda. Además, se incluyen notebooks para visualizar y procesar múltiples espectros simultáneamente.

---

## Objetivos

Analizar, comparar y visualizar el comportamiento espectral del café:

- Comparación de perfiles espectrales por región y molienda

---

##  Requisitos técnicos

Recomendado para reproducir los notebooks:

```bash
pip install pandas matplotlib jupyter

