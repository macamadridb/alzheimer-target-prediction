import os
import platform
import psutil
import torch
import time
import pandas as pd
import numpy as np 
import random 

# Importar funciones de los otros archivos
from data_preprocessing import *
from model_architecture import *
from training_evaluation import *
from module_analysis import *
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Para todas las GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False # Para mejor rendimiento, pero menos determinista
    torch.backends.cudnn.deterministic = True # ¡Importante para determinismo en CUDA!
    torch.use_deterministic_algorithms(True) # Para ciertas operaciones en CPU


set_seed(42) 

# --- Configuración de Rutas de Datos ---
BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "input") 
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output") 
# Crear paths individuales
edge_path = os.path.join(BASE_INPUT_DIR, "Edge.csv")
go_path = os.path.join(BASE_INPUT_DIR, "Go.csv")
protein_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_proteins.csv")
go_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_GO.csv")

# --- Parámetros del Modelo y Entrenamiento ---
IN_CHANNELS = None 
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64 
NUM_HEADS = 4 

PREDICTOR_HIDDEN_CHANNELS = 64

LEARNING_RATE = 0.001
EPOCHS = 100 # Número de épocas de entrenamiento

# --- Configuración de Ontología GO ---
GO_ONTOLOGY_FILTER = 'all' # Opciones: 'all', 'BP', 'MF', 'CC'

# --- FLujo Principal ---
def main():
    start_time = time.time()

    # --- 0. Información del Sistema ---
    print("Iniciando Pipeline de Detección de Módulos Funcionales con GNN-GO...")
    print("\n--- Información del Sistema ---")
    print(f"Sistema Operativo: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Procesador (CPU): {platform.processor()}")
    print(f"Núcleos de CPU (físicos/lógicos): {psutil.cpu_count(logical=False)}/{os.cpu_count()}")
    
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Memoria RAM Total: {total_ram_gb:.2f} GB")

    if torch.cuda.is_available():
        print(f"GPU Disponible: Sí")
        print(f"Nombre de GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        device = torch.device('cuda')
    else:
        print("GPU Disponible: No (Usando CPU)")
        device = torch.device('cpu')
    print(f"Dispositivo de Cómputo: {device}")

    print("\n--- Verificación de Archivos de Datos ---")
    if not os.path.isdir(BASE_INPUT_DIR):
        print(f"🚨 Error: El directorio de entrada '{BASE_INPUT_DIR}' no existe.")
        exit()

    EXPECTED_FILES = [
        "Edge.csv",
        "GO.csv",
        "metadata_proteins.csv", 
        "metadata_GO.csv"       
    ]

    
    for filename in EXPECTED_FILES:
        filepath = os.path.join(BASE_INPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"🚨 Error: El archivo '{filepath}' no se encontró.")
            print(f"Asegúrate de que todos los archivos CSV estén en '{BASE_INPUT_DIR}'.")
            exit()
    print("✔️ Todos los archivos de datos encontrados en el directorio de entrada.")

    # --- 1. Carga y Preprocesamiento de Datos ---
    print("\n--- Fase 1: Carga y Preprocesamiento de Datos ---")
    print("Cargando datasets CSV...")
    edges_df, go_terms_df, protein_metadata_df, go_metadata_df = load_files(
        edge_path, go_path, protein_metadata_path, go_metadata_path)
    print("Datasets cargados.")

    print("Creando mapeos de proteínas a índices numéricos...")
    protein_to_idx, idx_to_protein, all_proteins = create_node_mappings(edges_df, go_terms_df, protein_metadata_df)
    
    print(f"Creando características de nodos (features) con filtro GO: '{GO_ONTOLOGY_FILTER}'...")
    x, num_nodes_covered_by_go, num_go_terms_used, _, _, _, _ = create_node_features(
        protein_to_idx,
        go_terms_df,
        protein_metadata_df,
        go_metadata_df,
        go_ontology_filter=GO_ONTOLOGY_FILTER
    )
    global IN_CHANNELS
    IN_CHANNELS = x.shape[1] 
    print(f"Dimensión de las características de nodo (input para GNN): {IN_CHANNELS}")

    print("Creando índices de aristas y atributos de aristas (interaction_score)...")
    edge_index, edge_attr, num_edges_original, num_edges_bidirectional = create_edge_index_and_attributes(edges_df, protein_to_idx)

    print("\n--- Resumen del Grafo Cargado ---")
    print(f"  Total de Nodos (Proteínas únicas): {x.shape[0]}")
    print(f"  Nodos con GO terms cubiertos por la ontología '{GO_ONTOLOGY_FILTER}': {num_nodes_covered_by_go}")
    print(f"  Número de GO terms únicos utilizados (tras filtro '{GO_ONTOLOGY_FILTER}'): {num_go_terms_used}")
    print(f"  Total de Aristas originales (interacciones únicas): {num_edges_original}")
    print(f"  Total de Aristas en el grafo (bidireccional): {num_edges_bidirectional}")
    print(f"  Dimensión de atributos de arista (`interaction_score`): {edge_attr.shape[1]}")
    print(f"  Dimensión de características de nodo: {x.shape[1]}")

    # Crear objeto Data de PyG
    print("Creando objeto Data de PyTorch Geometric...")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # HASTA AQUÍ ES SOLO DE DATA_PREPROCESSING.PY
   
    print("\nDividiendo enlaces para entrenamiento/validación/prueba (predicción de enlaces)...")
    transform = RandomLinkSplit(
        num_val=0.1, #10% de los datos para validación
        num_test=0.1, #10% de los datos para prueba
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True
    )
    
    train_data, val_data, test_data = transform(data)

    data.train_pos_edge_index = train_data.pos_edge_label_index
    data.train_neg_edge_index = train_data.neg_edge_label_index
    data.val_pos_edge_index   = val_data.pos_edge_label_index
    data.val_neg_edge_index   = val_data.neg_edge_label_index
    data.test_pos_edge_index  = test_data.pos_edge_label_index
    data.test_neg_edge_index  = test_data.neg_edge_label_index
    data.x = train_data.x  # se mantiene igual
    data.edge_index = train_data.edge_index
    data.edge_attr = train_data.edge_attr
    
    print(f"  Enlaces Positivos de Entrenamiento: {data.train_pos_edge_index.shape[1]}")
    print(f"  Enlaces Positivos de Validación: {data.val_pos_edge_index.shape[1]}")
    print(f"  Enlaces Positivos de Prueba: {data.test_pos_edge_index.shape[1]}")
    print(f"  Enlaces Negativos de Entrenamiento: {data.train_neg_edge_index.shape[1]}")
    print(f"  Enlaces Negativos de Validación: {data.val_neg_edge_index.shape[1]}")
    print(f"  Enlaces Negativos de Prueba: {data.test_neg_edge_index.shape[1]}")


    # --- 2. Definición y Entrenamiento del Modelo GNN ---
    print("\n--- Fase 2: Entrenamiento del Modelo GNN ---")
    print(f"Inicializando GNNEncoder (in_channels={IN_CHANNELS}, hidden_channels={HIDDEN_CHANNELS}, out_channels={OUT_CHANNELS}, num_heads={NUM_HEADS})...")
    model = GNNEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, num_heads=NUM_HEADS).to(device)
    predictor = LinkPredictor(OUT_CHANNELS, PREDICTOR_HIDDEN_CHANNELS, 1).to(device) 

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss() 

    data = data.to(device)

    results = []
    print(f"Comenzando el entrenamiento por {EPOCHS} épocas...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc, train_acc, train_precision, train_recall, train_f1 = train(model, predictor, data, optimizer, criterion)
        val_loss, test_loss, val_auc, test_auc, val_acc, test_acc, val_precision, test_precision, val_recall, test_recall, val_f1, test_f1 = test(model, predictor, data)

        # Guardar métricas en lista
        results.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss.item(),
        'test_loss': test_loss.item(),
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1
    })
        
        # Mostrar métricas cada 10 épocas (o al inicio y al final)
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f'  Epoch: {epoch:03d} | '
                f'Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | '
                f'Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | '
                f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | '
                f'Train Prec: {train_precision:.4f} | Val Prec: {val_precision:.4f} | Test Prec: {test_precision:.4f} | '
                f'Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f} | Test Recall: {test_recall:.4f} | '
                f'Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}')
    # Guardar métricas en archivo CSV en la carpeta de output
    output_path = os.path.join(BASE_OUTPUT_DIR, "resultados_metricas_entrenamiento.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nMétricas guardadas en: {output_path}")


    print("\nEntrenamiento del modelo GNN completado. Generando embeddings finales...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index, data.edge_attr).cpu().numpy()
    print(f"Embeddings finales generados. Dimensión: {final_embeddings.shape}")

    # Guardar embeddings finales en un archivo CSV
    protein_ids = list(protein_to_idx.keys())  # usa tu diccionario de mapeo de proteínas
    emb_df = pd.DataFrame(final_embeddings, index=protein_ids)

    # Guardar en archivo .csv
    emb_output_path = os.path.join(BASE_OUTPUT_DIR, "embeddings.csv")
    emb_df.to_csv(emb_output_path)

    print(f"[INFO] Embeddings guardados en: {emb_output_path}")


    # --- Tiempo de Ejecución ---
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n--- Pipeline Completado ---")
    print(f"Tiempo total de ejecución del pipeline: {execution_time:.2f} segundos ({execution_time/60:.2f} minutos)")
    print("¡El análisis ha finalizado con éxito!")

if __name__ == "__main__":
    main()