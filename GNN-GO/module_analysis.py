# Funciones para la detección de módulos, visualización de embeddings, analisis de enriquecimiento GO y medatados. 

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

def detect_modules(embeddings, protein_ids, n_clusters=None, clustering_method='kmeans', random_state=42):
    
    
    labels = None
    if clustering_method == 'kmeans':
        if n_clusters is None:
            print("Para K-Means, 'n_clusters' debe ser especificado.")
            return None, None
        print(f"Detectando módulos con MiniBatchKMeans (n_clusters={n_clusters})...")
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        labels = clusterer.fit_predict(embeddings)
    elif clustering_method == 'dbscan':
        print("Detectando módulos con DBSCAN. Considera ajustar eps y min_samples.")
        clusterer = DBSCAN(eps=0.5, min_samples=5) 
        labels = clusterer.fit_predict(embeddings)
    else:
        print(f"Método de clustering '{clustering_method}' no soportado.")
        return None, None

    protein_module_map = {protein_ids[i]: labels[i] for i in range(len(protein_ids))}
    
    unique_labels = np.unique(labels) # Obtener etiquetas únicas
    valid_labels = unique_labels[unique_labels != -1] # Excluir el ruido (-1)
    
    if len(valid_labels) > 1:
        valid_indices = labels != -1
        valid_embeddings = embeddings[valid_indices]
        valid_cluster_labels = labels[valid_indices]

        if valid_embeddings.shape[0] > 1: 
            silhouette_avg = silhouette_score(valid_embeddings, valid_cluster_labels)
            davies_bouldin_avg = davies_bouldin_score(valid_embeddings, valid_cluster_labels)
            calinski_harabasz_avg = calinski_harabasz_score(valid_embeddings, valid_cluster_labels)
            print(f"Métricas de Clustering (excluyendo ruido):")
            print(f"  Silhouette Score: {silhouette_avg:.4f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin_avg:.4f}")
            print(f"  Calinski-Harabasz Index: {calinski_harabasz_avg:.4f}")
        else:
            print("No se pueden calcular métricas de clustering: menos de 2 muestras válidas después de excluir ruido.")
    else:
        print("No se pueden calcular métricas de clustering: menos de 2 clusters válidos o solo ruido.")
    
    return labels, protein_module_map

def visualize_embeddings(embeddings, labels, title="Embeddings de Nodos (t-SNE)"):
    
    valid_indices = labels != -1
    embeddings_filtered = embeddings[valid_indices]
    labels_filtered = labels[valid_indices]

    if embeddings_filtered.shape[0] == 0:
        print("No hay puntos válidos para visualizar después de filtrar ruido.")
        return

    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=min(30.0, embeddings_filtered.shape[0]-1),
                max_iter=1000) 
    # esta dando error el n_iter
    embeddings_2d = tsne.fit_transform(embeddings_filtered)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels_filtered)

    for label in unique_labels:
        idx = labels_filtered == label
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            s=12,
            alpha=0.75,
            label=f"Módulo {label}"
        )

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Módulos", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def analyze_modules(protein_module_map, go_terms_df, protein_metadata_df, go_metadata_df, all_proteins_list):
   
    unique_modules = np.unique(list(protein_module_map.values()))
    results = []

    protein_go_map = defaultdict(list)
    for _, row in go_terms_df.iterrows():
        protein_go_map[row['proteina']].append(row['GO_term'])
    
    go_term_details = go_metadata_df.set_index('GO_term')
    protein_deg_map = protein_metadata_df.set_index('proteina')['DEG'].to_dict()
    protein_target_group_map = protein_metadata_df.set_index('proteina')['Target_group'].to_dict()

    all_go_terms_in_network = [term for prot_id in all_proteins_list if prot_id in protein_go_map for term in protein_go_map[prot_id]]
    global_go_counts = pd.Series(all_go_terms_in_network).value_counts().to_dict()
    
    proteins_in_go_universe = len(set(p for p in all_proteins_list if protein_go_map[p])) # Total de proteínas con al menos un GO term asignado

    for module_id in sorted(unique_modules):
        if module_id == -1: 
            continue

        # Proteinas del modulo, se filtran las prpoteinas que pertenecen al modulo actual
        module_proteins = [p for p, m in protein_module_map.items() if m == module_id]
        
        if not module_proteins:
            continue

        # 1. Enriquecimiento GO
        module_go_terms = [term for p in module_proteins for term in protein_go_map[p]]
        module_go_counts = pd.Series(module_go_terms).value_counts().to_dict()
        
        enriched_go_terms_list = []
        p_values = []

        # Test hipergeométrico para cada GO term en el módulo
        for go_term, module_count in module_go_counts.items():
            k = module_count 
            M = global_go_counts.get(go_term, 0)
            n = len(module_proteins)
            N = proteins_in_go_universe
            
            if N == 0 or M == 0 or n == 0:
                p_val = 1.0
            else:
                p_val = stats.hypergeom.sf(k-1, N, M, n)
                # correcion por FDR (Benjamini-Hochberg)
            
            p_values.append(p_val)
            enriched_go_terms_list.append(go_term)
            
        if p_values:
            rejected, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            # Seleccionar el GO term con el menor p-valor corregido, este es el más representativo
            
            best_go_term = None
            min_p_val_corrected = float('inf')
            
            for i, go_term in enumerate(enriched_go_terms_list):
                if p_values_corrected[i] < min_p_val_corrected:
                    min_p_val_corrected = p_values_corrected[i]
                    best_go_term = go_term
            
            go_term_name = go_term_details.loc[best_go_term, 'Term_Name_Clean'] if best_go_term in go_term_details.index else 'N/A'
            representative_go = f"{best_go_term} ({go_term_name})"
            representative_go_p_value = min_p_val_corrected
        else:
            representative_go = "N/A (No GO terms enriched)"
            representative_go_p_value = 1.0 

        # 2. Distribución de Proteínas DEG
        deg_counts = defaultdict(int)
        for p_id in module_proteins:
            deg_status = protein_deg_map.get(p_id, 'unknown')
            deg_counts[deg_status] += 1
        
        deg_distribution = {status: f"{(count / len(module_proteins) * 100):.2f}%" 
                            for status, count in deg_counts.items()}
        
        # 3. Distribución de Target_Group
        individual_t_counts = defaultdict(int)
        for p_id in module_proteins:
            target_group_str = protein_target_group_map.get(p_id, '')
            if target_group_str:
                for tg in target_group_str.split(','):
                    tg_stripped = tg.strip()
                    if tg_stripped:
                        individual_t_counts[tg_stripped] += 1
        
        total_proteins_in_module = len(module_proteins)
        target_group_distribution = {}
        for tg, count in individual_t_counts.items():
            if total_proteins_in_module > 0:
                target_group_distribution[tg] = f"{(count / total_proteins_in_module * 100):.2f}%"
            else:
                target_group_distribution[tg] = "0.00%"


        results.append({
            'module_id': module_id,
            'representative_go': representative_go,
            'representative_go_p_value': representative_go_p_value,
            'deg_distribution': deg_distribution,
            'target_group_distribution': target_group_distribution,
            'module_proteins': module_proteins
        })

    return results

def assign_module_go_to_proteins(results):
    # a partir de los resultados de analyze_modules, asignar el GO representativo a cada proteína

    rows = []
    for module in results:
        rep_go = module['representative_go']
        if '(' in rep_go and ')' in rep_go:
            go_term = rep_go.split('(')[0].strip()
            term_name = rep_go.split('(')[1].replace(')', '').strip()
        else:
            go_term = rep_go
            term_name = ""

        for prot in module['module_proteins']:
            rows.append({
                'proteina': prot,
                'GO_term': go_term,
                'Term_Name_Clean': term_name
            })
    return pd.DataFrame(rows)