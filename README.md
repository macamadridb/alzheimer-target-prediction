# alzheimer-target-prediction

# GNN-GO -- Pipeline reproducible para múltiples redes PPI
Este repositorio contiene el código, datos y resultados asociados a la memoria de título de Macarena Madrid (Ingeniería Civil Informática, Universidad de Concepción).

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
  - `01_data_preprocessing.py/`             # Paso 1
  - `02_tuning.py/`                         # Paso 2
  - `03_model_training.py/`                 # Paso 3
  - `04_clustering_search.py/`              # Paso 4
  - `05_module_analysis.py/`                # Paso 5
- `main.py/`                                # Orquestador CLI (Command Line Interface) con subcomandos 
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






Este proyecto busca identificar blancos terapéuticos potenciales para el Alzheimer mediante el uso de redes de interacción proteína-proteína (PPI), información funcional (GO) y modelos de Deep Learning.

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

