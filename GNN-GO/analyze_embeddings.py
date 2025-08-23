import os
import pandas as pd
import numpy as np # Importar numpy para np.unique en caso de visualización
from collections import defaultdict

# Importar funciones de los otros archivos
from data_preprocessing import *
from model_architecture import *
from training_evaluation import *
from module_analysis import *

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "input") 
EMBEDDINGS_PATH = os.path.join(BASE_OUTPUT_DIR, "Trial_2", "embeddings_normalizados.csv")
FINAL_CLUSTER_RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "Trial_2")
HDBSCAN_CLUSTER_LABELS_PATH = os.path.join(BASE_OUTPUT_DIR, "Trial_2", "final_hdbscan_clusters", "hdbscan_cluster_labels_mcs20_ms20_cse0.00_alpha2.0.csv")

GO_ONTOLOGY_FILTER = "all"  # ajustar si aplica filtro

# Crear paths individuales
edge_path = os.path.join(BASE_INPUT_DIR, "Edge.csv")
go_path = os.path.join(BASE_INPUT_DIR, "GO.csv")
protein_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_proteins.csv")
go_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_GO.csv")


def main():

    print("Inicializando analyze_embeddings.py para validación biológica ...")
    # --- Cargar la lista de todas las proteínas (del archivo de embeddings o de metadata) ---
    # Es crucial que `all_proteins_list` contenga todas las proteínas que se consideraron
    # en tu espacio de embeddings, para que el análisis de enriquecimiento sea correcto
    # (sirve como el universo de referencia).
    try:
        df_embeddings = pd.read_csv(EMBEDDINGS_PATH)
        all_proteins_list = df_embeddings.iloc[:, 0].tolist()
        print(f"Cargadas {len(all_proteins_list)} IDs de proteínas del archivo de embeddings.")
    except FileNotFoundError:
        print(f"Advertencia: No se encontró {EMBEDDINGS_PATH}. Intentando cargar protein_metadata_df para all_proteins_list.")
        protein_metadata_df_temp = pd.read_csv(protein_metadata_path, sep=',')
        all_proteins_list = protein_metadata_df_temp['proteina'].tolist()
        print(f"Cargadas {len(all_proteins_list)} IDs de proteínas de metadata_proteins.csv.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar la lista de todas las proteínas: {e}")
        print("Asegúrate de que 'ordered_protein_ids' esté definida o que EMBEDDINGS_PATH sea correcto.")
        # Fallback si ordered_protein_ids no está definida globalmente
        # Puedes intentar cargarla desde go_path o edge_path si son fuentes de todas las proteínas
        # Por simplicidad aquí, si falla, usaremos una lista vacía, lo que causará errores posteriores si no se soluciona.
        all_proteins_list = [] 
    
    # --- Cargar datos adicionales para análisis GO ---
    print("\nCargando datos de anotación biológica...")
    go_terms_df = pd.read_csv(go_path, sep='\t')
    protein_metadata_df = pd.read_csv(protein_metadata_path, sep=',')
    go_metadata_df = pd.read_csv(go_metadata_path, sep=',')
    print("Datos de anotación cargados exitosamente.")

   # --- Cargar los resultados de clustering de HDBSCAN ---
    print(f"\nCargando etiquetas de clúster de HDBSCAN desde: {HDBSCAN_CLUSTER_LABELS_PATH}")
    try:
        df_hdbscan_labels = pd.read_csv(HDBSCAN_CLUSTER_LABELS_PATH)
        # Crear el diccionario protein_module_map esperado por analyze_modules
        protein_module_map = df_hdbscan_labels.set_index('protein_id')['cluster_label'].to_dict()
        print("Etiquetas de clúster de HDBSCAN cargadas y mapeadas.")
        print(f"Ejemplo de mapeo: {list(protein_module_map.items())[:5]} (primeras 5)")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de etiquetas de clúster de HDBSCAN en: {HDBSCAN_CLUSTER_LABELS_PATH}")
        print("Asegúrate de que la ruta sea correcta y el archivo 'hdbscan_cluster_labels_mcs50.csv' exista.")
        print("El análisis de módulos no se puede realizar sin las etiquetas de clúster.")
        return # Sale de la función main si no se encuentran las etiquetas
    except Exception as e:
        print(f"ERROR al cargar o procesar las etiquetas de clúster de HDBSCAN: {e}")
        return
    
    # --- Análisis y Evaluación de Módulos ---
    print("\n--- Fase: Análisis y Enriquecimiento de Módulos (HDBSCAN) ---")
    
    # La función analyze_modules ya está diseñada para recibir el 'protein_module_map'
    # y los otros dataframes de anotación.
    analysis_results = analyze_modules(
        protein_module_map, go_terms_df, protein_metadata_df, go_metadata_df, all_proteins_list
    )

    # Imprimir resultados
    print("\n--- Resultados del Análisis de Módulos Detectados (HDBSCAN) ---")
    module_summary = []
    for res in analysis_results:
        print(f"\n**Módulo {res['module_id']}** (Tamaño: {len(res['module_proteins'])} proteínas)") # Añadido el tamaño del módulo
        print(f"  Go Representativo: {res['representative_go']}")
        print(f"  p-value (corregido): {res['representative_go_p_value']:.2e}") # Más claro que es corregido
        print(f"  Combined score: {res['representative_combined_score']:.2f}") # Añadido el score combinado
        print(f"  Z-Score: {res['representative_go_z_score']:.2f}") # Añadido el Z-Score del GO representativo
        
        # Formatear DEG
        deg_str = ", ".join([f"{count}% {status}" for status, count in res['deg_distribution'].items()])
        print(f"  Proteínas DEG: {deg_str}")
        
        # Formatear Target Group
        target_group_str = ", ".join([f"{count}% {tg}" for tg, count in res['target_group_distribution'].items()])
        print(f"  Distribución de Target_Group: {target_group_str}")
        # La línea 'Total proteínas en módulo' ya está en la cabecera.
        # print(f"  Total proteínas en módulo: {len(res['module_proteins'])}") # Removida para evitar duplicidad

        # agregar a resumen
        module_summary.append({
            'module_id': res['module_id'],
            'module_size': len(res['module_proteins']), # Añadido el tamaño para el CSV
            'representative_go': res['representative_go'],
            'representative_go_p_value': res['representative_go_p_value'],
            'representative_combined_score': res['representative_combined_score'],
            'representative_go_z_score': res['representative_go_z_score'],
            'deg_distribution': str(res['deg_distribution']),
            'target_group_distribution': str(res['target_group_distribution']),
            'proteins': ", ".join(res['module_proteins']) # Opcional, lista de proteínas del módulo
            })
            
    # Guardar como CSV
    output_summary_path = os.path.join(FINAL_CLUSTER_RESULTS_DIR, "hdbscan_module_analysis_summary.csv") # Guardar en la carpeta de resultados de HDBSCAN
    results_df = pd.DataFrame(module_summary)
    results_df.to_csv(output_summary_path, index=False)
    print(f"\nResumen del análisis de módulos guardado en: {output_summary_path}")

    # asignar GO representativo a cada proteína
    print("\n--- Generando lista de Nodos con Términos GO Más Representativos (por Módulo) ---")
    protein_go_df = assign_module_go_to_proteins(analysis_results)
    
    # Imprimir los primeros 20 y luego indicar que hay más
    print(protein_go_df.head(20).to_string(index=False))
    if len(protein_go_df) > 20:
        print(f"... y {len(protein_go_df) - 20} más.")

    # Guardar la tabla en archivo CSV
    output_protein_go_path = os.path.join(FINAL_CLUSTER_RESULTS_DIR, "hdbscan_proteins_with_representative_go.csv") # Guardar en la carpeta de resultados de HDBSCAN
    protein_go_df.to_csv(output_protein_go_path, index=False)
    print(f"\nLista de nodos con GOs representativos guardada en: {output_protein_go_path}")

    print("\n--- Proceso de Validación Biológica Completado ---")


if __name__ == "__main__":
    main()

