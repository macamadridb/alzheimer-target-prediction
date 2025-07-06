# data_loader.py
# ------------------------------------------
# Carga de datos PPI y GO desde archivos MTGO
# Construye edge_index y matriz de features X
# ------------------------------------------

import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data


def load_graph_data():
    """
    Carga datos de entrada desde ../data/Input/
    - Edge.csv (Protein_A<TAB>Protein_B<TAB>Score)
    - GO.csv (Protein_ID<TAB>GO_term)
    Devuelve:
    - data: PyG Data object
    - protein_to_idx: dict
    - go_to_idx: dict
    """
    # Rutas de entrada
    data_dir = os.path.join("..", "data", "Input")
    edge_path = os.path.join(data_dir, "Edge.csv")
    go_path = os.path.join(data_dir, "GO.csv")

    # Leer archivos
    edge_df = pd.read_csv(edge_path, sep='\t', header=None, names=["Prot_A", "Prot_B", "Score"])
    go_df = pd.read_csv(go_path, sep='\t', header=None, names=["Protein_ID", "GO_term"])

    # Proteínas únicas (nodos)
    proteins = sorted(set(edge_df["Prot_A"]).union(edge_df["Prot_B"]))
    go_terms = sorted(go_df["GO_term"].unique())

    # Mapas
    protein_to_idx = {prot: idx for idx, prot in enumerate(proteins)}
    go_to_idx = {go: i for i, go in enumerate(go_terms)}

    # edge_index
    edge_index = torch.tensor([
        [protein_to_idx[a] for a in edge_df["Prot_A"]],
        [protein_to_idx[b] for b in edge_df["Prot_B"]],
    ], dtype=torch.long)

    # Matriz de features X (proteínas x GO terms)
    X = torch.zeros((len(proteins), len(go_terms)), dtype=torch.float)
    for _, row in go_df.iterrows():
        p = row["Protein_ID"]
        g = row["GO_term"]
        if p in protein_to_idx and g in go_to_idx:
            X[protein_to_idx[p], go_to_idx[g]] = 1.0
            
    # Crear máscara de entrenamiento (10% de los nodos)
    num_nodes = len(proteins)
    train_ratio = 0.1
    train_indices = np.random.choice(num_nodes, size=int(num_nodes * train_ratio), replace=False)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    # Objeto de PyG
    data = Data(x=X, edge_index=edge_index)
    data.train_mask = train_mask

    return data, protein_to_idx, go_to_idx

if __name__ == "__main__":
    data, protein_to_idx, go_to_idx = load_graph_data()

    print(f"[INFO] Número de nodos: {data.num_nodes}")
    print(f"[INFO] Número de aristas: {data.num_edges}")
    print(f"[INFO] Dimensión de X: {data.x.shape}")
