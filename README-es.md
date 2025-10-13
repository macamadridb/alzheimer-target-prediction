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

Si, por ejemplo, tus archivos `Edge.csv` y `GO.csv` usan **coma ( , )** en lugar de **tabulador ( \t )**, debes modificar las líneas correspondientes en la definición de la función `load_files` (Celda 3) de esta manera:

```python
def load_files(edge_path, go_path, protein_metadata_path, go_metadata_path):
    # Antes: edges_df = pd.read_csv(edge_path, sep='\t')
    edges_df = pd.read_csv(edge_path, sep=',')  # <-- CAMBIADO

    # Antes: go_terms_df = pd.read_csv(go_path, sep='\t')
    go_terms_df = pd.read_csv(go_path, sep=',')  # <-- CAMBIADO

    protein_metadata_df = pd.read_csv(protein_metadata_path, sep=',')
    go_metadata_df = pd.read_csv(go_metadata_path, sep=',')

    return edges_df, go_terms_df, protein_metadata_df, go_metadata_df
```
#### C. Ajuste de la Codificación de Características (`def create_node_features`)

La función `create_node_features` es responsable de transformar los atributos categóricos de las proteínas en características numéricas (**features**) para el modelo GNN.

El código actual utiliza **dos métodos de codificación:**


**1. Label Encoding (Codificación de Etiquetas):**

- Se usa para variables categóricas **ordinales o binarias**, como `Target_type` (`single, complex, unknown`) y `DEG` (`up, down, none`).
- Asigna un valor entero único a cada categoría (e.g., 0, 1, 2).

**2. Multi-Hot Encoding (Codificación Multi-Etiqueta):**

- Se usa para atributos donde una proteína puede tener **múltiples valores simultáneamente** (separados por comas), como `Target_group` y los términos **GO (Gene Ontology)**.
- Crea una columna binaria (0 o 1) para cada posible valor único. Si una proteína posee ese valor, la columna es 1; de lo contrario, es 0.  
  👉 **Esto genera la mayoría de las features de entrada `X` para la GNN.**

**Para modificar el código si añades nuevas columnas:**
- Si añades una nueva columna categórica simple (ej: `Is_Essential` con valores `Yes / No`), utiliza **Label Encoding** si deseas una codificación simple:

```python
# Ejemplo de adición de Label Encoding para una nueva columna 'New_Category'
le_new = LabelEncoder()
protein_features['New_Category_encoded'] = le_new.fit_transform(protein_features['New_Category'].fillna('default_value'))
# Luego, añade 'New_Category_encoded' al pd.concat final.
```

- Si añades una nueva columna con múltiples valores separados por comas (ej: `Disease_Associations`), utiliza **Multi-Hot Encoding** (similar a cómo se procesa `Target_group` y `GO_term`).

```python
# Ejemplo de adición de Multi-Hot Encoding para 'New_Multilabel_Feature'
mlb_new = MultiLabelBinarizer()
new_encoded = mlb_new.fit_transform(
    protein_features['New_Multilabel_Feature'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
)
new_df = pd.DataFrame(new_encoded, index=protein_features.index, columns=mlb_new.classes_)
# Luego, añade new_df al pd.concat final.
```

Ejecuta el resto de las celdas del notebook (`01_data_preprocessing.ipynb`) en orden para generar el objeto `data` de PyTorch Geometric listo para el entrenamiento.

### 📌 02_tuning.ipynb

### 📌 03_model_training.ipynb

### 📌 04_clustering_search.ipynb

### 📌 05_module_analysis.ipynb









---

Este proyecto busca identificar blancos terapéuticos potenciales para el Alzheimer mediante el uso de redes de interacción proteína-proteína (PPI), información funcional (GO) y modelos de Deep Learning.


## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

