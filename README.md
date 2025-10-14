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

## 4. Modification and Reproducibility (Manual)

### 📌 01_data_preprocessing.ipynb

The key file for loading new data is **`01_data_preprocessing.ipynb`**.  
To run the pipeline with a different PPI network or new datasets, follow these steps:

#### A. Modify File Paths (Cell 2)

The only place where file paths need to be changed is **Cell 2** of the notebook `01_data_preprocessing.ipynb`.

```python
# BASE_INPUT_DIR = './input'  # <-- Change this path if your data is not located in ./input

# File names are defined here:
# EDGE_FILENAME = "Edge.csv"
# GO_FILENAME = "GO.csv"
# ...
```

#### B. Adjust File Separators (`def load_files`)

The `load_files` function is responsible for loading the files.  
It is crucial that the `sep` parameter matches the separator used in your CSV/TSV files.

In the current code, the separators are configured as follows:

| File                    | Separator (`sep`) | Value  |
|--------------------------|------------------|:------:|
| `Edge.csv`               | Tab              | `'\t'` |
| `GO.csv`                 | Tab              | `'\t'` |
| `metadata_proteins.csv`  | Comma            | `','`  |
| `metadata_GO.csv`        | Comma            | `','`  |

If, for example, your `Edge.csv` and `GO.csv` files use **comma ( , )** instead of **tab ( \t )**, you must modify the corresponding lines in the definition of the `load_files` function (Cell 3) as follows:

```python
def load_files(edge_path, go_path, protein_metadata_path, go_metadata_path):
    # Before: edges_df = pd.read_csv(edge_path, sep='\t')
    edges_df = pd.read_csv(edge_path, sep=',')  # <-- CHANGED

    # Before: go_terms_df = pd.read_csv(go_path, sep='\t')
    go_terms_df = pd.read_csv(go_path, sep=',')  # <-- CHANGED

    protein_metadata_df = pd.read_csv(protein_metadata_path, sep=',')
    go_metadata_df = pd.read_csv(go_metadata_path, sep=',')

    return edges_df, go_terms_df, protein_metadata_df, go_metadata_df
```
#### C. Adjust Feature Encoding (`def create_node_features`)

The `create_node_features` function is responsible for transforming categorical protein attributes into numerical **features** for the GNN model.

The current code uses **two encoding methods:**

**1. Label Encoding**

- Used for **ordinal or binary categorical variables**, such as `Target_type` (`single, complex, unknown`) and `DEG` (`up, down, none`).  
- Assigns a unique integer value to each category (e.g., 0, 1, 2).


**2. Multi-Hot Encoding**

- Used for attributes where a protein can have **multiple simultaneous values** (comma-separated), such as `Target_group` and **GO (Gene Ontology)** terms.  
- Creates a binary column (0 or 1) for each possible unique value.  
  👉 **This produces most of the input features `X` for the GNN.**

**To modify the code if you add new columns:**

- If you add a new simple categorical column (e.g., `Is_Essential` with values `Yes / No`), use **Label Encoding** for a simple encoding:

```python
# Example of adding Label Encoding for a new column 'New_Category'
le_new = LabelEncoder()
protein_features['New_Category_encoded'] = le_new.fit_transform(
    protein_features['New_Category'].fillna('default_value')
)
# Then, add 'New_Category_encoded' to the final pd.concat.
```

- If you add a new column with multiple comma-separated values (e.g., `Disease_Associations`), use **Multi-Hot Encoding** (similar to how `Target_group` and `GO_term` are processed).

```python
# Example of adding Multi-Hot Encoding for 'New_Multilabel_Feature'
mlb_new = MultiLabelBinarizer()
new_encoded = mlb_new.fit_transform(
    protein_features['New_Multilabel_Feature'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
)
new_df = pd.DataFrame(new_encoded, index=protein_features.index, columns=mlb_new.classes_)
# Then, add new_df to the final pd.concat.
```

Run the remaining cells of the notebook (`01_data_preprocessing.ipynb`) in order to generate the PyTorch Geometric `data` object ready for training.

### 📌 02_tuning.ipynb

The **`02_tuning.ipynb`** file corresponds to the hyperparameter optimization phase.  
Its purpose is to find the combination of hyperparameters that maximizes the model’s performance.

---

#### A. Component Loading
At the beginning, the notebook performs the following key steps:

1. **Data Loading:** Loads the previously saved `data` and `metadata` objects from the `output/` folder.

2. **Architecture Definition:** Defines the `GNNEncoder` and `LinkPredictor` classes, which establish the base structure of the model.

3. **Training Evaluation:** Defines the functions `def train` and `def test`, which contain the logic for propagation and metric computation (AUC, F1, Accuracy, etc.).

#### B. Link Splitting
In a dedicated cell, `RandomLinkSplit` from PyTorch Geometric is used to divide the loaded graph’s edges into three datasets:

- **80% for Training (`train_data`)**: Used for GNN propagation and embedding generation.  

- **10% for Validation (`val_data`)**: Used to evaluate each *trial’s* performance and guide Optuna’s pruning process.  

- **10% for Testing (`test_data`)**: Kept untouched until the end but evaluated in each *trial* to record the final metric of the best architecture.

#### C. Hyperparameter Search with Optuna
The `def objective` function is the core of this notebook.  
Optuna repeatedly calls it with different combinations of hyperparameters:

1. **Hyperparameter Suggestion:**  
   In Optuna, you propose values for key parameters such as:
   - `hidden_channels`, `out_channels` (Embedding dimensions)  
   - `num_heads` (Attention heads in GATv2)  
   - `learning_rate`, `dropout_rate`  
   - `epochs`

2. **Trial Execution:**  
   For each suggested combination, the model is initialized and trained, repeatedly calling `def train` and `def test`.

3. **Pruning:**  
   The `MedianPruner` is used to stop *trials* early when their validation performance is consistently poor, saving computation time.

4. **Saving Results:**  
   Once completed, the study saves a full record of all *trials* and the best-performing architecture in `output/optuna_study_results.csv`.  
   This file will later be used as input for the final training phase.

:warning: **Computation Time Considerations**  
The hyperparameter search phase is the **most computationally intensive** stage of the pipeline.

| **Factor**           | **Consideration** |
|----------------------|------------------|
| **Number of Trials** | A higher number of *trials* increases the likelihood of finding the global optimum but also extends runtime. The default value is `20`. |
| **Epochs**           | The number of epochs suggested by `trial.suggest_int("epochs", 50, 200, step=50)` also has a direct impact on execution time. |
| **Hardware**         | Runtime varies significantly depending on hardware. For large networks (such as the AD network), running `20` trials may take **20 hours or more** on a machine with a dedicated GPU. It is recommended to adjust the number of *trials* and *epochs* according to available computational resources. |

```python
# Modification in the Optuna execution cell (if necessary):
# n_trials = 20  # <-- Change this value to reduce or increase computation time.
```

### 📌 03_model_training.ipynb
This is the **final notebook** of the pipeline, dedicated to final training, embedding generation, and rigorous evaluation of the GNN model.

---

#### A. Purpose

1. **Optimal Loading:**  
   Loads the optimal hyperparameters (HPs) — *Learning Rate, Dropout, Channels, Epochs* — found by Optuna from the file `output/optuna_study_results.csv`.

2. **Final Training:**  
   Trains the selected GNN architecture for the optimal number of epochs, **saving the best model based on Validation set performance**.

3. **Test Evaluation:**  
   Evaluates the trained model **only once** on the Test set (`test_data`), which has remained completely isolated during the training and tuning phases.  
   This result provides the **final and unbiased metric** of the model.

#### B. Data Flow

The notebook uses:

- **Model:** `GNNEncoder` (GAT with `edge_attr`) and `LinkPredictor` (MLP with concatenation).  

- **Data:** The graph `data` loaded from `output/processed_graph_data.pt`, then split again (80% Train, 10% Val, 10% Test) using the **same random seed** to ensure reproducibility.  

- **Hyperparameters:** The HP values are assigned to variables such as `HIDDEN_CHANNELS`, `LEARNING_RATE`, `FINAL_EPOCHS`, etc., directly from the Optuna results.

#### C. Final Results

At the end of execution, this notebook produces **two key artifacts** in the `output/` directory:

---

#### 1. Full Training Report
- **File:** `output/resultados_metricas_entrenamiento.csv`  
- **Content:** Contains all metrics (*Loss, AUC, Accuracy, F1*, etc.) for the **Training**, **Validation**, and **Test** sets at **each epoch** of the final training.

---

#### 2. Final Node Embeddings
- **File:** `output/embeddings.csv`  
- **Content:** The feature vectors (*embeddings*) generated by the best `GNNEncoder` for all graph nodes.

### 📌 04_clustering_search.ipynb

### 📌 05_module_analysis.ipynb

---

This project aims to identify potential therapeutic targets for Alzheimer’s disease by integrating protein–protein interaction (PPI) networks, functional information (GO), and Deep Learning models.

## 📄 License

This project is distributed under the MIT License.
