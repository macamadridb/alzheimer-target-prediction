[Español](README-es.md)
# alzheimer-target-prediction

# GNN-GO — Reproducible Pipeline for Multiple PPI Networks — EN

This repository contains the code, data, and results associated with the undergraduate thesis of **Macarena Madrid** (Computer Engineering, Universidad de Concepción).  
Specifically, the **code developed for reproducibility** is located in the `gnngo` directory.

## Objective
This template allows you to run the entire pipeline on any PPI network, starting from raw data to the characterization of functional modules:

1. Data and feature preparation  
2. Hyperparameter search for the GNN  
3. GNN training and evaluation  
4. Clustering optimization on node embeddings  
5. Analysis and characterization of functional modules  

---

## 1. Structure

- `input/`
  - `Edge.csv`
  - `GO.csv`
  - `metadata_GO.csv`
  - `metadata_proteins.csv`
- `outputs/`
- `configs/`
- `src/`:  
  - `01_data_preprocessing.ipynb` — Step 1  
  - `02_tuning.ipynb` — Step 2  
  - `03_model_training.ipynb` — Step 3  
  - `04_clustering_search.ipynb` — Step 4  
  - `05_module_analysis.ipynb` — Step 5  
- `requirements.txt`  
- `README.md`  

---

## 2. Requirements and Installation
- **Python 3.10 or 3.11** (recommended)

```bash
# Install dependencies
pip install -r requirements.txt
```
---

## 3. Input Files (folder `input/`)
:warning: **Make sure your new data files are placed in this folder!**

1. **Edge.csv** (separator: tab `\t`, with header)
- **Accepted columns:** `protein1/proteina1`, `protein2/proteina2`, `interaction_score`  
- **Example:**
```bash
   protein1 protein2 interaction_score
   CALM3 PRKACA 0.82
   PRKACB PRKACG 0.77
```

2. **GO.csv** (separator: tab `\t`, with header)
- **Accepted columns:** `protein/proteina, GO_term.`
- **Example:**
```bash
protein GO_term
CALM3 GO:0005515
```

3. **metadata_GO.csv** (separator: comma `,`, with header)
- **Accepted columns:** : `GO_term` , `Term_Name_Clean` , `Ontology` 

4. **metadata_proteins.csv** (separator: comma `,`, with header)
- **Accepted columns:** : `protein/proteina` + additional **attributes** pr column (e.g.,`Target_group`, `DEG`, etc)

