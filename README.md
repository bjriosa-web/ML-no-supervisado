# DimensioVision AI
Plataforma web de reducción de dimensionalidad (PCA) con Streamlit, scikit-learn y Plotly.
Dataset: Wine Quality (UCI, https://archive.ics.uci.edu/dataset/186/wine+quality) o tu propio CSV.

## Instalación y ejecución en Windows (CMD o PowerShell)
1. Instala Python 3.10+ desde python.org (marca "Add Python to PATH").
2. En la carpeta del proyecto:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
3. Se abrirá http://localhost:8501. Flujo: Dataset → Preprocesamiento → Reducción → Visualización → Comparación → Conclusiones (descarga del informe HTML).

Nota: la descarga desde UCI requiere internet. Sin conexión, descarga `winequality-red.csv` / `winequality-white.csv` y cárgalos en la sección Dataset (separador `;` detectado automáticamente).
