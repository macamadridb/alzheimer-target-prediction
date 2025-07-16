import os
import platform
import psutil
import torch
import time
import pandas as pd
import numpy as np # Importar numpy para np.unique en caso de visualización

# Importar funciones de los otros archivos
from data_preprocessing import *
from model_architecture import *
from training_evaluation import *
from module_analysis import *
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

# --- Configuración de Rutas de Datos ---
BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "input") 

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
EPOCHS = 100 

N_CLUSTERS = 10 
CLUSTERING_METHOD = 'kmeans' 

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
    print(f"  Número de GO terms únicos utilizados (tras filtro): {num_go_terms_used}")
    print(f"  Total de Aristas originales (interacciones únicas): {num_edges_original}")
    print(f"  Total de Aristas en el grafo (bidireccional): {num_edges_bidirectional}")
    print(f"  Dimensión de atributos de arista (`interaction_score`): {edge_attr.shape[1]}")

    # Crear objeto Data de PyG
    print("Creando objeto Data de PyTorch Geometric...")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # HASTA AQUÍ FUNCIONA SÚPER BIEN
    # DE AQUI en adelante no xdd
    print("\nDividiendo enlaces para entrenamiento/validación/prueba (predicción de enlaces)...")
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1, is_undirected=True) 
    
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

    print(f"Comenzando el entrenamiento por {EPOCHS} épocas...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, predictor, data, optimizer, criterion)
        val_auc, test_auc = test(model, predictor, data)
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f'  Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

    print("\nEntrenamiento del modelo GNN completado. Generando embeddings finales...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index, data.edge_attr).cpu().numpy()
    print(f"Embeddings finales generados. Dimensión: {final_embeddings.shape}")

    # --- 3. Detección de Módulos Funcionales ---
    print(f"\n--- Fase 3: Detección de Módulos Funcionales ---")
    print(f"Aplicando {CLUSTERING_METHOD} (K={N_CLUSTERS}) sobre los embeddings de nodos...")
    
    ordered_protein_ids = list(protein_to_idx.keys())
    
    module_labels, protein_module_map = detect_modules(
        final_embeddings, ordered_protein_ids, n_clusters=N_CLUSTERS, clustering_method=CLUSTERING_METHOD
    )
    
    if module_labels is not None and len(np.unique(module_labels[module_labels != -1])) > 1:
        print("Visualizando embeddings de nodos...")
        visualize_embeddings(final_embeddings, module_labels, title=f"Embeddings de Nodos por Módulo (GO: {GO_ONTOLOGY_FILTER})")
    else:
        print("No hay suficientes módulos válidos para visualizar.")

    # --- 4. Análisis y Evaluación de Módulos ---
    print("\n--- Fase 4: Análisis y Evaluación de Módulos ---")
    print("Realizando análisis de enriquecimiento GO y distribución de metadatos por módulo...")
    analysis_results = analyze_modules(
        protein_module_map, go_terms_df, protein_metadata_df, go_metadata_df, all_proteins
    )

    print("\n--- Resultados del Análisis de Módulos Detectados ---")
    for res in analysis_results:
        print(f"\n**Módulo {res['module_id']}**")
        print(f"  Go Representativo: {res['representative_go']}")
        print(f"  p-value: {res['representative_go_p_value']:.2e}")
        
        # Formatear DEG
        deg_str = ", ".join([f"{count}% es {status}" for status, count in res['deg_distribution'].items()])
        print(f"  Proteínas DEG: {deg_str}")
        
        # Formatear Target Group
        target_group_str = ", ".join([f"{count}% es {tg}" for tg, count in res['target_group_distribution'].items()])
        print(f"  Distribución de Target_Group: {target_group_str}")
        print(f"  Total proteínas en módulo: {len(res['module_proteins'])}")

    print("\n--- Lista de Nodos con Términos GO Más Representativos ---")
    protein_go_assignments = assign_most_representative_go(protein_module_map, go_terms_df, go_metadata_df)
    
    # Imprimir los primeros 20 y luego indicar que hay más
    protein_go_output_lines = []
    for protein_id in sorted(protein_go_assignments.keys()): # Ordenar para salida consistente
        go_info = protein_go_assignments[protein_id]
        protein_go_output_lines.append(f"{protein_id}\t{go_info}")

    for i, line in enumerate(protein_go_output_lines):
        if i < 20: # Imprime solo los primeros 20 para la consola
            print(line)
        else:
            break
    if len(protein_go_output_lines) > 20:
        print(f"... y {len(protein_go_output_lines) - 20} más.")

    # Guardar las asignaciones en un archivo csv
    output_df = pd.DataFrame([line.split('\t', 1) for line in protein_go_output_lines], 
                             columns=['Proteina', 'GO_Asignado'])
    
    output_df.to_csv('protein_GO_assignments_formatted.csv', sep='\t', index=False)
    print(f"\nLista de nodos con GOs guardada en 'protein_GO_assignments_formatted.csv'")


    print("\n--- Outputs Adicionales Generados ---")
    embeddings_df = pd.DataFrame(final_embeddings, index=ordered_protein_ids)
    embeddings_df.index.name = 'proteina'
    embeddings_df.to_csv('protein_embeddings.csv', sep='\t', header=True)
    print(f"Embeddings de proteínas guardados en 'protein_embeddings.csv'")

    module_assignments_df = pd.DataFrame(list(protein_module_map.items()), columns=['proteina', 'modulo'])
    module_assignments_df.to_csv('protein_module_assignments.csv', sep='\t', index=False)
    print(f"Asignaciones de módulos guardadas en 'protein_module_assignments.csv'")

    # --- Tiempo de Ejecución ---
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n--- Pipeline Completado ---")
    print(f"Tiempo total de ejecución del pipeline: {execution_time:.2f} segundos ({execution_time/60:.2f} minutos)")
    print("¡El análisis ha finalizado con éxito!")

if __name__ == "__main__":
    main()