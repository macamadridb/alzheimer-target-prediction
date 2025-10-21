[English](README.md)
# alzheimer-target-prediction

# GNN-GO -- Pipeline reproducible para múltiples redes PPI -- ES
Este repositorio contiene el código, datos y resultados asociados a la memoria de título de Macarena Madrid (Ingeniería Civil Informática, Universidad de Concepción).
Específicamente, el **código que se generó para ser replicable** se encuentra en la carpeta `gnngo`.

## Objetivo
Esta plantilla permite ejecutar tu pipeline sobre cualquier red PPI, desde los datos brutos y caracterización de módulos funcionales:
1. Preparación de datos y atributos
2. Búsqueda de hiperparámetros para la GNN
3. Entranamiento y evaluación de la GNN
4. Búsqueda del mejor clustering sobre los embeddings
5. Análisis y caracterización de módulos funcionales

---

## 1. Estructura

- `input/`
  - `Edge.csv`
  - `GO.csv`
  - `metadata_GO.csv`
  - `metadata_proteins.csv`

- `output/`  

- `01_data_preprocessing.ipynb` — Paso 1  
- `02_tuning.ipynb` — Paso 2  
- `03_model_training.ipynb` — Paso 3  
- `04_clustering_search.ipynb` — Paso 4  
- `05_module_analysis.ipynb` — Paso 5  

- `requirements.txt`  



---
## 2. Requisitos e instalación
El proyecto fue desarrollado en **Python 3.10 o 3.11** y requiere la instalación de varias librerías adicionales para el procesamiento de datos y el entrenamiento del modelo GNN.

Instala todas las librerías necesarias con:
```bash
# Instalar dependencias
pip install -r requirements.txt
```

:bulb: Nota: Si cuentas con una GPU, puedes aprovecharla para acelerar el entrenamiento del modelo instalando la versión de PyTorch compatible con CUDA.
Consulta las instrucciones oficiales en: https://pytorch.org/get-started/locally

---

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

El archivo **`02_tuning.ipynb`** es la fase de optimización de hiperparámetros. Su propósito es encontrar la combinación de hiperparémetros que maximice el rendimiento del modelo.

#### A. Carga de componentes
Al inicio, el notebook realiza los siguientes pasos clave:

1. **Carga de Datos:** Se llama al objeto `data` y `metadata` guardados previamente en `output/`.

2. **Definición de Arquitectura:** Se definen las clases `GNNEncoder` y `LinkPredictor`, estableciendo la estructura base del modelo.

3. **Evaluación de Entrenamiento:** Se definen las funciones `def train` y `def test`, que contienen la lógica para la propagación y el cálculo de métricas (AUC, F1, Acc, etc.).

#### B. División de enlaces
En una celda dedicada, se utiliza `RandomLinkSplit` de PyTorch Geometric para dividir los enlaces del grafo cargado en tres conjuntos:

- **80% para Entrenamiento (`train_data`)**: Se usa para la propagación GNN y la generación de *embeddings*.
  
- **10% para Validación (`val_data`)**: Se usa para evaluar el rendimiento de cada *trial* y guiar el proceso de poda (*pruning*) de Optuna.

- **10% para Prueba (`test_data`)**: Se mantiene intocable hasta el final, pero se evalúa en cada *trial* para guardar la métrica final de la mejor arquitectura.

#### C. Búsqueda de hiperparámetros con Optuna
La función `def objective` es el núcleo de este notebook. Optuna la llama repetidamente con diferentes combinaciones de hiperparámetros:

1. **Sugerencia de Hiperparámetros:**  
   En Optuna debes proponer valores para parámetros clave como:
   - `hidden_channels`, `out_channels` (Dimensiones del *embedding*).
   - `num_heads` (Cabezas de atención del GATv2).
   - `learning_rate`, `dropout_rate`.
   - `epochs`.

2. **Ejecución del Trial:**  
   Para cada combinación sugerida, el modelo se inicializa y entrena, llamando repetidamente a `def train` y `def test`.

3. **Poda (*Pruning*):**  
   Se utiliza el podador `MedianPruner` para detener anticipadamente los *trials* que muestran un rendimiento de validación consistentemente bajo, ahorrando tiempo de cómputo.

4. **Guardado de Resultados:**  
   Al finalizar, el estudio guarda un registro completo de todos los *trials* y la mejor arquitectura encontrada en `output/optuna_study_results.csv`.  
   Este archivo se usará como entrada para el entrenamiento final.

:warning: **Consideraciones de Tiempo de Cómputo**
La ejecución de la búsqueda de hiperparámetros es la fase **más intensiva en cómputo**.

| **Factor**           | **Consideración** |
|---------------------|------------------|
| **Número de Trials** | Un mayor número de *trials* aumenta la probabilidad de encontrar el óptimo global, pero incrementa el tiempo de ejecución. El valor por defecto es `20`. |
| **Épocas**           | El número de épocas sugerido por `trial.suggest_int("epochs", 50, 200, step=50)` también impacta directamente. |
| **Hardware**         | El tiempo de ejecución varía drásticamente. Para grandes redes (como la red de AD), la ejecución de `20` trials puede tomar **20 horas o más** en hardware con GPU dedicada. Se recomienda ajustar el número de *trials* y las *épocas* según la disponibilidad de recursos. |

```python
# Modificación en la celda de ejecución de Optuna (si es necesario):
# n_trials = 20  # <-- Cambia este valor para reducir o aumentar el tiempo de cómputo.
```

### 📌 03_model_training.ipynb
Este es el **último notebook** del pipeline, dedicado al entrenamiento final, a la generación de embeddings y a la evaluacion rigurosa del modelo GNN

#### A. Propósito

1. **Carga Óptima:**  
   Carga los hiperparámetros óptimos (HPs) — *Learning Rate, Dropout, Canales, Épocas* — encontrados por Optuna desde el archivo `output/optuna_study_results.csv`.

2. **Entrenamiento Final:**  
   Entrena la arquitectura GNN seleccionada durante el número de épocas óptimo, **guardando el mejor modelo basado en el rendimiento del conjunto de Validación**.

3. **Evaluación en Prueba:**  
   Evalúa el modelo entrenado **una sola vez** en el conjunto de Prueba (`test_data`), que se ha mantenido completamente aislado durante las fases de entrenamiento y tuning. Este resultado proporciona la **métrica final y no sesgada** del modelo.

####  B. Flujo de Datos

El notebook utiliza:

- **Modelo:** `GNNEncoder` (GAT con `edge_attr`) y `LinkPredictor` (MLP con concatenación).

- **Datos:** El grafo `data` cargado desde `output/processed_graph_data.pt` y dividido de nuevo (80% Train, 10% Val, 10% Test) usando la **misma semilla** para replicabilidad.
s
- **Hiperparámetros:** Los valores de los HPs son asignados a las variables `HIDDEN_CHANNELS`, `LEARNING_RATE`, `FINAL_EPOCHS`, etc., directamente desde el resultado de Optuna.

#### C. Resultados Finales

Al finalizar la ejecución, este notebook genera **dos artefactos clave** en el directorio `output/`:

#### 1. Reporte Completo de Entrenamiento
- **Archivo:** `output/resultados_metricas_entrenamiento.csv`
- **Contenido:** Contiene todas las métricas (*Loss, AUC, Accuracy, F1*, etc.) para los conjuntos de **Entrenamiento**, **Validación** y **Prueba** en **CADA ÉPOCA** del entrenamiento final.

#### 2. Embeddings Finales de Nodos
- **Archivo:** `output/embeddings.csv`
- **Contenido:** Los vectores de características (*embeddings*) generados por el mejor `GNNEncoder` para todos los nodos del grafo.

### 📌 04_clustering_search.ipynb
Este notebook realiza la optimización de hiperparámetros para **tres algoritmos de clustering** (`K-Means`, `DBSCAN` y `HDBSCAN`) utilizando los **embeddings finales generados por el GNN**.

#### A. Flujo General

1. **Normalización:**  
   Los embeddings cargados son sometidos a una **normalización L2** para asegurar la consistencia.

2. **Reducción de Dimensionalidad (UMAP):**  
   Se utiliza **UMAP** para visualizar la estructura de los embeddings y analizar el resultado de los **clusters óptimos**.

3. **Búsqueda Exhaustiva:**  
   Se ejecuta una búsqueda por rejilla o rangos predefinidos para encontrar la **mejor configuración de hiperparámetros** basada en métricas de calidad intrínsecas (`Silhouette`, `Davies-Bouldin`, `Calinski-Harabasz`).

#### B. Algoritmos y Configuraciones (Detalle)

Estos notebooks realizan la optimización de tres algoritmos de **clustering** sobre los **embeddings finales**.

#### 1. K-Means

Se busca el número óptimo de **clusters (K)** mediante evaluación de métricas.

- **Rango de Búsqueda:**  
  El rango máximo de `K` a probar es ajustable y se define por la limitación:  
  `min(50, embeddings_data.shape[0] - 1)`.

- **Artefactos Generados** (en `output/clustering_optimization_results/`):

  | Tipo                      | Archivo generado |
  |--------------------------|------------------|
  | **Gráfico de Evaluación** | `kmeans_k_evaluation_metrics.png` |
  | **Visualización UMAP**    | `umap_kmeans_k_x_plot.png`        |

#### 2. DBSCAN

Se realiza una búsqueda exhaustiva en dos dimensiones: el **radio (`Eps`)** y el **mínimo de muestras (`Min Samples`)**.

- **Rango de Búsqueda de `Eps` (Distancia Coseno):**  
  Es ajustable y se explora un amplio espectro, típicamente cubriendo valores desde muy pequeños (por ejemplo, `0.001`) hasta valores grandes (por ejemplo, `1.5`).

- **Rango de Búsqueda de `Min Samples`:**  
  Es ajustable y recorre valores discretos desde `2` hasta `201` para variar la **densidad de los clusters**.

- **Artefactos Generados** (en `output/clustering_optimization_results/`):

  | Tipo                          | Archivo generado |
  |------------------------------|------------------|
  | **Resultados CSV**           | `dbscan_parameter_search_results_extended.csv` |
  | **Mapas de Calor (Heatmaps)** | `dbscan_silhouette_heatmap.png`, `dbscan_noise_percentage_heatmap.png`, *(otros generados automáticamente)* |
  | **Visualización UMAP**       | `umap_dbscan_eps_x_minS_x_plot.png` |

#### 3. HDBSCAN

Este algoritmo utiliza una búsqueda exhaustiva sobre **cuatro hiperparámetros clave**:  
`min_cluster_size`, `min_samples`, `cluster_selection_epsilon` y `alpha`.

- **Artefactos Generados** (en `output/clustering_optimization_results/`):

  | Tipo                           | Archivo generado |
  |-------------------------------|------------------|
  | **Resultados CSV**            | `hdbscan_parameter_search_results_exhaustive.csv` |
  | **Heatmaps 2D**               | `hdbscan_silhouette_best_heatmap.png`, `hdbscan_noise_best_heatmap.png`, *(otros mapas generados automáticamente)* |
  | **Visualización UMAP**        | `umap_hdbscan_optimal_plot.png` |


#### C. Parámetros óptimos en HDBSCAN
El notebook final requiere que el usuario defina los **Hiperparámetros óptimos de HDBSCAN** (`min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `alpha`) basándose en los resultados del análisis previo.
Se aplica **HDBSCAN una única vez** sobre los embeddings para generar el conjunto definitivo de clusters.

Los siguientes hiperparámetros deben ser **actualizados manualmente** con los valores obtenidos de la mejor combinación en `04_clustering_search.ipynb`:

| Variable                                | Descripción                                           | Valor por Defecto (Ejemplo) |
|----------------------------------------|-------------------------------------------------------|-----------------------------|
| `OPTIMAL_MIN_CLUSTER_SIZE`             | Tamaño mínimo de clúster                              | `20` |
| `OPTIMAL_MIN_SAMPLES`                  | Mínimo de muestras para definir la densidad           | `20` |
| `OPTIMAL_CLUSTER_SELECTION_EPSILON`    | Límite de distancia para fusionar clústeres           | `0.0` |
| `OPTIMAL_ALPHA`                        | Factor de suavidad                                    | `2.0` |

Resultados Finales (en `output/final_hdbscan_clusters/`)

Al finalizar la ejecución, este notebook consolida el resultado del clustering en archivos definitivos:

#### 1. 🏷️ Etiquetas de Cluster
- **Archivo:** `hdbscan_cluster_labels.csv`  
- **Contenido:** Un mapeo de cada ID de proteína a su etiqueta de cluster final (incluyendo el **ruido**, marcado como `-1`).

#### 2. 📊 Resumen de Clústeres
- **Archivo:** `hdbscan_cluster_summary.csv`  
- **Contenido:** Lista de los clusters válidos y el tamaño de cada uno, ordenados por tamaño.

#### 3. 🎨 Visualización UMAP
- **Archivo:** `umap_hdbscan_optimal_plot.png`  
- **Contenido:** Representación 2D de todos los embeddings, coloreados por su cluster final detectado por HDBSCAN.

### 📌 05_module_analysis.ipynb

Este notebook es el **paso final del pipeline**, dedicado a la **validación biológica** y la **caracterización funcional de los módulos de proteínas (clústeres)** obtenidos en la etapa previa de HDBSCAN.

:warning: **Configuración crítica**
Antes de ejecutar este notebook, es necesario **ajustar manualmente** el sufijo de parámetros de HDBSCAN para que coincida con el **nombre del archivo de etiquetas de clúster** generado en la etapa previa.

```python
HDBSCAN_PARAMS_SUFFIX = "mcs20_ms20_cse0.00_alpha2.0"  # <<< AJUSTAR ESTE VALOR >>>
```

####  A. Propósito

1. **Validación Funcional (GO):**  
   Determinar la **significancia biológica** de cada módulo mediante el análisis de **Enriquecimiento de Gene Ontology (GO)**.

2. **Estandarización de IDs (Crítica):**  
   Definir el universo de proteínas (`N`) cargando el mapeo `protein_to_idx` desde `metadata.pt` para asegurar la correcta correspondencia entre los clústeres y las anotaciones.

3. **Caracterización de Módulos:**  
   Cuantificar la distribución de proteínas con **Expresión Diferencial (DEG)** y su pertenencia a **Grupos Objetivo (`Target_group`)**.

####  B. Flujo de Datos y Métodos

- **Método de Enriquecimiento GO:**

  1. **Test Hipergeométrico:**  
     Calcula la sobrerrepresentación de términos GO en cada clúster.

  2. **FDR (Benjamini-Hochberg):**  
     Corrige los *p-values* por múltiples comparaciones para evitar falsos positivos.

  3. **Combined Score:**  
     Métrica final usada para ordenar términos GO (**− log₁₀(p_corregido) × Z-Score**) y seleccionar el GO **más representativo** del módulo.

- **Clúster de Ruido (`-1`):**  
  Se excluye del cálculo de enriquecimiento, pero **se reportan sus metadatos** para un posible análisis posterior.

####  C. Resultados Clave

El notebook genera **dos reportes principales** en el directorio de resultados del clúster:

1. Resumen del Análisis de Módulos
- **Archivo:** `hdbscan_module_analysis_summary.csv`
- **Contenido:**  
  Contiene **una fila por cada módulo** con:
  - Tamaño del clúster.
  - **GO Representativo** (con su *p-values corregido*, **Combined Score** y **Z-Score**).
  - Distribuciones porcentuales de **DEG** y **Target_group**.


2. Lista de Nodos con GO Representativo
- **Archivo:** `hdbscan_proteins_with_representative_go.csv`
- **Contenido:**  
  Una tabla que asigna a cada proteína de un clúster válido **el término GO representativo y su nombre limpio**, facilitando la interpretación funcional de los nodos.

### 🖥️ Visualización en Cytoscape

Los archivos generados permiten crear una **visualización de red funcional** en Cytoscape.

---

#### 📌 Pasos para Importar Resultados en Cytoscape (usando la red original):

1. **Abrir Cytoscape**  
   👉 Descargar desde: https://cytoscape.org/

2. **Importar Red (Estructura):**  
   `File → Import → Network → File`  
   → Seleccionar el archivo original de la red (**Edge.csv** o similar).

3. **Importar Atributos (Metadatos):**  
   `File → Import → Table → File`  
   → Seleccionar el archivo `hdbscan_proteins_with_representative_go.csv`.  
   ✅ Asegurarse de que el ID de la proteína (`proteina`) se mapee correctamente con el ID del nodo de la red.

4. **Aplicar Layout por GO Representativo:**  
   `Layout → Group Attributes Layout`  
   → Seleccionar la columna **Nodes: Term_Name_Clean** (o la columna del GO representativo definida) para **agrupar nodos por su función biológica más significativa**.


---

Este proyecto busca identificar blancos terapéuticos potenciales para el Alzheimer mediante el uso de redes de interacción proteína-proteína (PPI), información funcional (GO) y modelos de Deep Learning.


## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

