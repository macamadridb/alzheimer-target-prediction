# alzheimer-target-prediction

# GNN-GO -- Pipeline reproducible para múltiples redes PPI -- ES
Este repositorio contiene el código, datos y resultados asociados a la memoria de título de Macarena Madrid (Ingeniería Civil Informática, Universidad de Concepción).
[English](README.md)

## Objetivo
Esta plantilla permite ejecutar tu pipeline sobre cualquier red PPI, desde los datos brutos y caracterización de módulos funcionales:
1. Preparación de datos y atributos
2. Búsqueda de hiperparámetros para la GNN
3. Entranamiento y evaluación de la GNN
4. Búsqueda del mejor clustering sobre los embeddings
5. Análisis y caracterización de módulos funcionales



## 1. Estructura

- `input/`
  - `Edge.csv`
  - `GO.csv`
  - `metadata_GO.csv`
  - `metadata_proteins.csv`
- `outputs/`
- `configs/`
- `src/`: 
  - `01_data_preprocessing.ipynb/`             # Paso 1
  - `02_tuning.ipynb/`                         # Paso 2
  - `03_model_training.ipynb/`                 # Paso 3
  - `04_clustering_search.ipynb/`              # Paso 4
  - `05_module_analysis.ipynb/`                # Paso 5
- `requirements.txt/`
- `README.md/`

## 2. Requisitos e instalación

- Python 
- Librerías clave: `pandas, numpy, networkx, scikit-learn, torch, torch-geometric, optuna, hdbscan, umap-learn, matplotlib`

Instala todo con

```bash
python -m venv .venv && source .venv/bin/activate # (Linux/Mac)
# .venv\Scripts\activate # (Windows PowerShell)
pip install -r requirements.txt
```

## 3. Archivos de entrada (carpeta `input/`)
:warning: **¡Asegúrate de que los archivos de tus nuevos datos se ubiquen en esta carpeta!**

1. **Edge.csv** (separador: tab `\t`, con encabezado)
- Columnas **aceptadas** : `protein1/proteina1, protein2/proteina2, interaction_score.`
- Ejemplo:
```bash
protein1 protein2 interaction_score
CALM3 PRKACA 0.82
PRKACB PRKACG 0.77
```

2. **GO.csv** (separador: tab `\t`, con encabezado)
- Columnas **aceptadas** : `protein/proteina, GO_term.`
- Ejemplo:
```bash
protein GO_term
CALM3 GO:0005515
```

3. **metadata_GO.csv** (separador: coma `,`, con encabezado)
- Columnas **aceptadas** : `GO_term` , `Term_Name_Clean` , `Ontology` 

4. **metadata_proteins.csv** (separador: coma `,`, con encabezado)
- Columnas **aceptadas** : `protein/proteina` + **atributos** adicionales por columna (p. ej.,`Target_group`, `DEG`, etc)

- ## 4. Modificación y Reproducibilidad (Manual)

### 📌 01_data_preprocessing.ipynb

El archivo clave para la entrada de nuevos datos es **`01_data_preprocessing.ipynb`**.  
Para ejecutar el pipeline con una red PPI diferente o nuevos datos, sigue estos pasos:

#### A. Modificar Rutas de Archivo (Celda 2)

El único lugar donde se deben cambiar las rutas de los archivos es en la **Celda 2** del notebook `01_data_preprocessing.ipynb`.

```python
# BASE_INPUT_DIR = './input'  # <-- Modifica esta ruta si tus datos no están en ./input

# Los nombres de archivo se definen aquí:
# EDGE_FILENAME = "Edge.csv"
# GO_FILENAME = "GO.csv"
# ...
```

####  B. Ajuste de Separadores de Archivo (`def load_files`)

La función `load_files` es la encargada de cargar los archivos.  
Es crucial que el parámetro `sep` coincida con el separador de tus archivos CSV/TSV.

En el código actual, los separadores están configurados así:

| Archivo                 | Separador (`sep`) | Valor |
|------------------------|------------------|:-----:|
| `Edge.csv`             | Tabulador        | `'\t'` |
| `GO.csv`               | Tabulador        | `'\t'` |
| `metadata_proteins.csv` | Coma             | `','`  |
| `metadata_GO.csv`       | Coma             | `','`  |


### 📌 02_tuning.ipynb

### 📌 03_model_training.ipynb

### 📌 04_clustering_search.ipynb

### 📌 05_module_analysis.ipynb










Este proyecto busca identificar blancos terapéuticos potenciales para el Alzheimer mediante el uso de redes de interacción proteína-proteína (PPI), información funcional (GO) y modelos de Deep Learning.

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

