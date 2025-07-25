import os
import pandas as pd
import numpy as np # Importar numpy para np.unique en caso de visualización

# Importar funciones de los otros archivos
from data_preprocessing import *
from model_architecture import *
from training_evaluation import *
from module_analysis import *

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "input") 
EMBEDDINGS_PATH = os.path.join(BASE_OUTPUT_DIR, "embeddings.csv")
GO_ONTOLOGY_FILTER = "all"  # ajustar si aplica filtro

# Crear paths individuales
edge_path = os.path.join(BASE_INPUT_DIR, "Edge.csv")
go_path = os.path.join(BASE_INPUT_DIR, "GO.csv")
protein_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_proteins.csv")
go_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_GO.csv")

N_CLUSTERS = 100 # Número de módulos a detectar 
CLUSTERING_METHOD = 'kmeans' 

def main():

    # --- 1. Cargar Embeddings ---
    print(f"🔹 Cargando embeddings desde: {EMBEDDINGS_PATH}")
    embedding_df = pd.read_csv(EMBEDDINGS_PATH, sep=',', index_col=0)
    final_embeddings = embedding_df.values
    ordered_protein_ids = embedding_df.index.tolist()

    # --- 2. Detección de Módulos Funcionales ---
    print(f"🔹 Fase 2: Detección de Módulos Funcionales ({CLUSTERING_METHOD}, K={N_CLUSTERS})")
    labels, protein_module_map = detect_modules(final_embeddings, ordered_protein_ids, n_clusters=N_CLUSTERS, clustering_method=CLUSTERING_METHOD)

    # Visualización t-SNE
    if labels is not None and len(np.unique(labels[labels != -1])) > 1:
        print("Visualizando y guardando embeddings de nodos...")
        emb_output_path = os.path.join(BASE_OUTPUT_DIR, "embeddings_nodos.png")
        visualize_embeddings(final_embeddings, labels,
                             title=f"Embeddings de Nodos por Módulo (GO: {GO_ONTOLOGY_FILTER})",
                             save_path=emb_output_path)
    else:
        print("No hay suficientes módulos válidos para visualizar.")

    # --- 3. Cargar datos adicionales para análisis GO ---
    go_terms_df = pd.read_csv(go_path, sep='\t')
    protein_metadata_df = pd.read_csv(protein_metadata_path, sep=',')
    go_metadata_df = pd.read_csv(go_metadata_path, sep=',')
    all_proteins = ordered_protein_ids

    # --- 4. Análisis y Evaluación de Módulos ---
    print("\\n--- Fase 3: Análisis y Enriquecimiento de Módulos ---")
    analysis_results = analyze_modules(
        protein_module_map, go_terms_df, protein_metadata_df, go_metadata_df, all_proteins
    )

    # Imprimir resultados
    print("\n--- Resultados del Análisis de Módulos Detectados ---")
    module_summary = []
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

        # agregar a resumen
        module_summary.append({
            'module_id': res['module_id'],
            'representative_go': res['representative_go'],
            'representative_go_p_value': res['representative_go_p_value'],
            'deg_distribution': str(res['deg_distribution']),
            'target_group_distribution': str(res['target_group_distribution']),
            'num_proteins': len(res['module_proteins']),
            'proteins': ", ".join(res['module_proteins'])  # Opcional
            })
    # Guardar como CSV
    output_path = os.path.join(BASE_OUTPUT_DIR, "resultados_modulos.csv")
    results_df = pd.DataFrame(module_summary)
    results_df.to_csv(output_path, index=False)

    print(f"\nResumen de módulos guardado en: {output_path}")

    # asignar GO representativo a cada proteína
    print("\n--- Lista de Nodos con Términos GO Más Representativos ---")
    protein_go_df = assign_module_go_to_proteins(analysis_results)
    
    # Imprimir los primeros 20 y luego indicar que hay más
    print(protein_go_df.head(20).to_string(index=False))
    if len(protein_go_df) > 20:
        print(f"... y {len(protein_go_df) - 20} más.")

    # Guardar la tabla en archivo CSV
    output_path = os.path.join(BASE_OUTPUT_DIR, "proteinas_con_go_representativo.csv")
    protein_go_df.to_csv(output_path, index=False)
    print(f"\nLista de nodos con GOs representativos guardada en: {output_path}")

if __name__ == "__main__":
    main()