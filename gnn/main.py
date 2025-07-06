# main.py
# ------------------------------------------
# Script principal: carga, modelo, entrenamiento, guardado
# ------------------------------------------

from data_loader import load_graph_data
from alz_gnn import GCN
from train import train
from utils import save_embeddings


def main():
    print("[INFO] Cargando datos...")
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

    print("[INFO] Guardando embeddings en archivo CSV...")
    idx_to_protein = {idx: prot for prot, idx in protein_to_idx.items()}
    save_embeddings(embeddings, idx_to_protein, output_path="embeddings.csv")


if __name__ == "__main__":
    main()
