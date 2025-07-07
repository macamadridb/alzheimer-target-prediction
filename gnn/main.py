# main.py
# ------------------------------------------
# Script principal: carga, modelo, entrenamiento, guardado
# ------------------------------------------

import os
import time
import platform
import psutil
import torch
from datetime import datetime

from data_loader import load_graph_data
from alz_gnn import GCN
from train import train
from utils import save_embeddings


def print_system_info():
    print(f"[INFO] Sistema operativo: {platform.system()} {platform.release()}")
    print(f"[INFO] CPU: {platform.processor()}")
    print(f"[INFO] Núcleos CPU: {psutil.cpu_count(logical=True)}")
    print(f"[INFO] Memoria RAM total: {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    print(f"[INFO] GPU disponible: {'Sí' if torch.cuda.is_available() else 'No'}")
    if torch.cuda.is_available():
        print(f"[INFO] Nombre GPU: {torch.cuda.get_device_name(0)}")


def main():
    print("[INFO] Información del sistema:")
    print_system_info()

    start = time.time()

    print("\n[INFO] Cargando datos...")
    data, protein_to_idx, go_to_idx = load_graph_data()

    print(f"[INFO] Nodos totales: {data.num_nodes}")
    print(f"[INFO] Nodos en train_mask: {data.train_mask.sum().item()}\n")

    print("[INFO] Inicializando modelo GCN...")
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=64
    )

    print("[INFO] Entrenando modelo...")
    model, embeddings = train(model, data)

    print("[INFO] Entrenamiento finalizado.")
    print(f"[INFO] Embeddings generados con forma: {embeddings.shape}")

    results_dir = os.path.join("..","gnn","results")
    os.makedirs(results_dir, exist_ok=True)

    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"embeddings_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)

    print(f"[INFO] Guardando embeddings en: {filepath}")
    idx_to_protein = {idx: prot for prot, idx in protein_to_idx.items()}
    save_embeddings(embeddings, idx_to_protein, output_path=filepath)

    print(f"[INFO] Entrenamiento finalizado en {time.time() - start:.2f} segundos.")


if __name__ == "__main__":
    main()
