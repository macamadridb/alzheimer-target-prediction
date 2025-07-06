# utils.py
# ------------------------------------------
# Funciones auxiliares: guardar embeddings, clustering, etc.
# ------------------------------------------

import pandas as pd
from sklearn.cluster import KMeans


def save_embeddings(embeddings, idx_to_protein, output_path="embeddings.csv"):
    """
    Guarda embeddings en un archivo CSV con nombre de proteína.
    """
    df = pd.DataFrame(embeddings.numpy())
    df.index = [idx_to_protein[i] for i in range(len(idx_to_protein))]
    df.to_csv(output_path)
    print(f"[INFO] Embeddings guardados en: {output_path}")


def cluster_embeddings(embeddings, n_clusters=10):
    """
    Aplica KMeans para agrupar nodos en módulos.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings.numpy())
    return labels
