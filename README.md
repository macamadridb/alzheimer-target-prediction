[Español](README-es.md)
# alzheimer-target-prediction

# GNN-GO — Replicable Pipeline for Multiple PPI Networks — EN


This repository contains the code, data, and results associated with the undergraduate thesis of **Macarena Madrid** (Computer Engineering, Universidad de Concepción).  
Specifically, the **code developed for replication** is located in the `gnngo` directory.

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

- `output/`  

- `01_data_preprocessing.ipynb` — Step 1  
- `02_tuning.ipynb` — Step 2  
- `03_model_training.ipynb` — Step 3  
- `04_clustering_search.ipynb` — Step 4  
- `05_module_analysis.ipynb` — Step 5  

- `requirements.txt`  


---

## 2. Requirements and Installation
The project was developed using **Python 3.10 or 3.11** and requires the installation of several additional libraries for data processing and GNN model training.

Install all required libraries with:

```bash
# Install dependencies
pip install -r requirements.txt
```
:bulb: Note: If you have a GPU, you can speed up model training by installing the CUDA-compatible version of PyTorch.
Follow the official installation guide here: https://pytorch.org/get-started/locally

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

## 4. Modification and Replicability (Manual)


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

:warning: **Note on device selection (CPU/GPU):**
This notebook automatically detects if a GPU is available and assigns the device as:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
However, on systems with limited GPU memory (e.g., lab workstations or shared servers), this may cause out of memory errors during training.
In such cases, it is recommended to force CPU usage by replacing that line with:
```python
device = torch.device('cpu')
```

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
At this stage, the graph is re-divided using **`RandomLinkSplit`**, a method that separates edges into **Training**, **Validation**, and **Test** subsets.  
This division is an **integral part of the evaluation process**, as all performance metrics — *AUC, Accuracy, Precision, Recall, and F1-score* — are computed over these subsets.  
This ensures a consistent and reproducible evaluation of the model’s predictive performance on unseen data.

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
This notebook performs hyperparameter optimization for **three clustering algorithms** (`K-Means`, `DBSCAN`, and `HDBSCAN`) using the **final embeddings generated by the GNN**.

#### A. General Workflow

1. **Normalization:**  
   The loaded embeddings undergo **L2 normalization** to ensure consistency.

2. **Dimensionality Reduction (UMAP):**  
   **UMAP** is used to visualize the structure of the embeddings and analyze the results of the **optimal clusters**.

3. **Exhaustive Search:**  
   A grid or range-based search is performed to find the **best hyperparameter configuration** based on intrinsic quality metrics (`Silhouette`, `Davies-Bouldin`, `Calinski-Harabasz`).

   #### B. Algorithms and Configurations (Details)

This notebook performs optimization of three **clustering** algorithms on the **final embeddings**.

---

#### 1. K-Means

The optimal number of **clusters (K)** is determined through metric evaluation.

- **Search Range:**  
  The maximum `K` value to test is adjustable and defined by the constraint:  
  `min(50, embeddings_data.shape[0] - 1)`.

- **Generated Artifacts** (in `output/clustering_optimization_results/`):

  | Type                     | Generated File |
  |---------------------------|----------------|
  | **Evaluation Plot**       | `kmeans_k_evaluation_metrics.png` |
  | **UMAP Visualization**    | `umap_kmeans_k_x_plot.png`        |

#### 2. DBSCAN

An exhaustive search is performed over two dimensions: **radius (`Eps`)** and **minimum samples (`Min Samples`)**.

- **`Eps` Search Range (Cosine Distance):**  
  Adjustable, typically exploring a wide range of values — from very small (e.g., `0.001`) to large (e.g., `1.5`).

- **`Min Samples` Search Range:**  
  Adjustable, iterating through discrete values from `2` to `201` to vary the **cluster density**.

- **Generated Artifacts** (in `output/clustering_optimization_results/`):

  | Type                           | Generated File |
  |--------------------------------|----------------|
  | **CSV Results**                | `dbscan_parameter_search_results_extended.csv` |
  | **Heatmaps**                   | `dbscan_silhouette_heatmap.png`, `dbscan_noise_percentage_heatmap.png`, *(others generated automatically)* |
  | **UMAP Visualization**         | `umap_dbscan_eps_x_minS_x_plot.png` |

#### 3. HDBSCAN

This algorithm performs an exhaustive search over **four key hyperparameters**:  
`min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, and `alpha`.

- **Generated Artifacts** (in `output/clustering_optimization_results/`):

  | Type                           | Generated File |
  |--------------------------------|----------------|
  | **CSV Results**                | `hdbscan_parameter_search_results_exhaustive.csv` |
  | **2D Heatmaps**                | `hdbscan_silhouette_best_heatmap.png`, `hdbscan_noise_best_heatmap.png`, *(other maps generated automatically)* |
  | **UMAP Visualization**         | `umap_hdbscan_optimal_plot.png` |

#### C. Optimal Parameters in HDBSCAN
The final notebook requires the user to define the **optimal HDBSCAN hyperparameters** (`min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `alpha`) based on the results from the previous analysis.  
**HDBSCAN is applied only once** to the embeddings to generate the final set of clusters.

The following hyperparameters must be **manually updated** with the best values obtained from `04_clustering_search.ipynb`:

| Variable                              | Description                                           | Default Value (Example) |
|--------------------------------------|-------------------------------------------------------|--------------------------|
| `OPTIMAL_MIN_CLUSTER_SIZE`           | Minimum cluster size                                  | `20` |
| `OPTIMAL_MIN_SAMPLES`                | Minimum number of samples to define density           | `20` |
| `OPTIMAL_CLUSTER_SELECTION_EPSILON`  | Distance threshold for cluster merging                | `0.0` |
| `OPTIMAL_ALPHA`                      | Smoothing factor                                      | `2.0` |

---

**Final Results** (in `output/final_hdbscan_clusters/`)

At the end of execution, this notebook consolidates the clustering output into the final result files:
#### 1. Cluster Labels
- **File:** `hdbscan_cluster_labels.csv`  
- **Content:** A mapping of each protein ID to its final cluster label (including **noise**, marked as `-1`).

#### 2. Cluster Summary
- **File:** `hdbscan_cluster_summary.csv`  
- **Content:** A list of valid clusters and their respective sizes, sorted by cluster size.

#### 3. UMAP Visualization
- **File:** `umap_hdbscan_optimal_plot.png`  
- **Content:** 2D representation of all embeddings, colored according to their final cluster assigned by HDBSCAN.


### 📌 05_module_analysis.ipynb
This notebook is the **final step of the pipeline**, dedicated to the **biological validation** and **functional characterization of protein modules (clusters)** obtained in the previous HDBSCAN stage.

:warning: **Critical Configuration**  
Before running this notebook, you must **manually adjust** the HDBSCAN parameter suffix to match the **cluster label file name** generated in the previous stage.

```python
HDBSCAN_PARAMS_SUFFIX = "mcs20_ms20_cse0.00_alpha2.0"  # <<< ADJUST THIS VALUE >>>
```

#### A. Purpose

1. **Functional Validation (GO):**  
   Assess the **biological significance** of each module through **Gene Ontology (GO) enrichment analysis**.

2. **ID Standardization (Critical):**  
   Define the protein universe (`N`) by loading the mapping `protein_to_idx` from `metadata.pt` to ensure correct correspondence between clusters and annotations.

3. **Module Characterization:**  
   Quantify the distribution of proteins with **Differential Expression (DEG)** and their membership in **Target Groups (`Target_group`)**.

#### B. Data Flow and Methods

- **GO Enrichment Method:**

  1. **Hypergeometric Test:**  
     Calculates the overrepresentation of GO terms within each cluster.

  2. **FDR (Benjamini–Hochberg):**  
     Adjusts *p-values* for multiple comparisons to prevent false positives.

  3. **Combined Score:**  
     Final metric used to rank GO terms (**−log₁₀(corrected p) × Z-Score**) and select the **most representative** GO term for each module.

- **Noise Cluster (`-1`):**  
  Excluded from the enrichment analysis, but **its metadata is still reported** for potential later exploration.

#### C. Key Results

The notebook generates **two main reports** in the cluster results directory:

---

1. Module Analysis Summary
- **File:** `hdbscan_module_analysis_summary.csv`  
- **Content:**  
  Contains **one row per module** including:
  - Cluster size  
  - **Representative GO term** (with corrected *p-value*, **Combined Score**, and **Z-Score**)  
  - Percentage distributions of **DEG** and **Target_group**


2. Node List with Representative GO Term
- **File:** `hdbscan_proteins_with_representative_go.csv`  
- **Content:**  
  A table mapping each protein in a valid cluster to its **representative GO term and cleaned name**, facilitating the functional interpretation of nodes.

### 🖥️ Visualization in Cytoscape

The generated files allow for the creation of a **functional network visualization** in Cytoscape.

---

#### 📌 Steps to Import Results into Cytoscape (using the original network):

1. **Open Cytoscape**  
   👉 Download from: https://cytoscape.org/

2. **Import Network (Structure):**  
   `File → Import → Network → File`  
   → Select the original network file (**Edge.csv** or similar).

3. **Import Attributes (Metadata):**  
   `File → Import → Table → File`  
   → Select the file `hdbscan_proteins_with_representative_go.csv`.  
   ✅ Make sure the protein ID (`proteina`) correctly maps to the node ID in the network.

4. **Apply Layout by Representative GO Term:**  
   `Layout → Group Attributes Layout`  
   → Select the column **Nodes: Term_Name_Clean** (or the defined representative GO column) to **group nodes according to their most significant biological function**.


---

This project aims to identify potential therapeutic targets for Alzheimer’s disease by integrating protein–protein interaction (PPI) networks, functional information (GO), and Deep Learning models.

## 📄 License

This project is distributed under the MIT License.
