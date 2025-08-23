# Funciones para la detección de módulos, visualización de embeddings, analisis de enriquecimiento GO y medatados. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap 
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import seaborn as sns


def analyze_modules(protein_module_map, go_terms_df, protein_metadata_df, go_metadata_df, all_proteins_list):
   # Obtener todos los IDs de módulos únicos, incluyendo -1 (ruid)
    unique_modules = np.unique(list(protein_module_map.values()))
    results = []

    protein_go_map = defaultdict(list)
    for _, row in go_terms_df.iterrows():
        protein_go_map[row['proteina']].append(row['GO_term'])
    
    go_term_details = go_metadata_df.set_index('GO_term')
    protein_deg_map = protein_metadata_df.set_index('proteina')['DEG'].to_dict()
    protein_target_group_map = protein_metadata_df.set_index('proteina')['Target_group'].to_dict()

    # Contar GO terms en todo el universo de proteinas con GO anotado
    all_go_terms_in_network = [term for prot_id in all_proteins_list if prot_id in protein_go_map for term in protein_go_map[prot_id]]
    global_go_counts = pd.Series(all_go_terms_in_network).value_counts().to_dict()
    
    # Total de proteínas con al menos un GO term asignado
    proteins_in_go_universe = len(set(p for p in all_proteins_list if protein_go_map[p])) # Total de proteínas con al menos un GO term asignado

    # Iterar sobre todos los módulos únicos, incluyendo -1
    for module_id in sorted(unique_modules):
        #if module_id == -1: 
        #    continue

        # Eliminamos la condición para que el clsuter de ruido sea procesado.


        # Proteinas del modulo, se filtran las prpoteinas que pertenecen al modulo actual
        module_proteins = [p for p, m in protein_module_map.items() if m == module_id]
        
        if not module_proteins:
            continue

        # 1. Enriquecimiento GO
        representative_go = "N/A (Ruido, no GO enriquecido)" if module_id == -1 else "N/A (No GO terms enriched)"
        representative_go_p_value = 1.0
        representative_combined_score = 0.0
        representative_go_z_score = 0.0

        if module_id != -1: # Realizar el test de enriquecimiento GO solo para clústeres válidos
            module_go_terms = [term for p in module_proteins for term in protein_go_map[p]]
            module_go_counts = pd.Series(module_go_terms).value_counts().to_dict()
            
            enriched_go_terms_list = []
            p_values = []
            z_scores = []

            # Test hipergeométrico para cada GO term en el módulo
            for go_term, module_count in module_go_counts.items():
                k = module_count
                M = global_go_counts.get(go_term, 0)  # Total de proteínas con este GO term en el universo
                n = len(module_proteins) # tamaño del módulo
                N = proteins_in_go_universe # Total de proteínas con al menos un GO term en el universo
                
                # Para evitar divisiones por cero o valores no validos 
                if N == 0 or M == 0 or n == 0:
                    p_val = 1.0 # significa que no hay enriquecimiento, no hay GO term en el universo o en el módulo, se asigna 1.0 porque no hay enriquecimiento
                    z_score = 0.0 # significa que no hay enriquecimiento, si es 0 es porque no hay GO term en el universo o en el módulo, los datos no son validos
                else:
                    # stats.hypergeom.sf(k-1, N, M, n) calcula el p-valor de la probabilidad de obtener al menos k éxitos en una muestra de tamaño n
                    p_val = stats.hypergeom.sf(k-1, N, M, n)
                    mean = n * (M / N)  # Media esperada bajo la hipótesis nula, la hipotesis nula es que el GO term no está enriquecido en el módulo
                    variance = n * (M / N) * (1 - M / N) * (N - n) / (N - 1)  # Varianza esperada bajo la hipótesis nula, se usa para calcular el z-score

                    # evitar division por cero
                    if variance > 0:
                        z_score = (k - mean) / np.sqrt(variance)
                    else:
                        z_score = 0.0

                p_values.append(p_val)
                z_scores.append(z_score)
                enriched_go_terms_list.append(go_term)
                
            if p_values:
                # corregir p.values usando FDR
                rejected, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh') # 0.05 es el nivel de significancia

                # Calcular el combined score para los términos significativos
                combined_scores = []
                for i in range(len(p_values_corrected)):
                    if rejected[i] and p_values_corrected[i] > 0 and z_scores[i] > 0:
                        score = -np.log10(p_values_corrected[i]) * z_scores[i]
                        combined_scores.append(score)
                    else:
                        combined_scores.append(0)
                
                # Seleccionar el GO term con el mayor combined score, este es el GO term representativo del módulo
                best_go_term = None
                max_combined_score = 0.0
                
                for i, go_term in enumerate(enriched_go_terms_list):
                    if combined_scores[i] > max_combined_score:
                        max_combined_score = combined_scores[i]
                        best_go_term = go_term
                
                if best_go_term:
                    go_term_name = go_term_details.loc[best_go_term, 'Term_Name_Clean'] if best_go_term in go_term_details.index else 'N/A'
                    representative_go = f"{best_go_term} ({go_term_name})"
                    # se usa el p-valor corregido del mejor GO term
                    representative_go_p_value = p_values_corrected[enriched_go_terms_list.index(best_go_term)]
                    representative_combined_score = max_combined_score # se guarda el score combinado del mejor GO term
                    representative_go_z_score = z_scores[enriched_go_terms_list.index(best_go_term)]
                else:
                    representative_go = "N/A (No GO terms enriched)"
                    representative_go_p_value = 1.0  # No se encontró GO enriquecido
                    representative_combined_score = 0.0  # No se encontró GO enriquecido
                    representative_go_z_score = 0.0  # No se encontró GO enriquecido

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
            'representative_combined_score': representative_combined_score,
            'representative_go_z_score': representative_go_z_score,
            'deg_distribution': deg_distribution,
            'target_group_distribution': target_group_distribution,
            'module_proteins': module_proteins
        })

    return results

def assign_module_go_to_proteins(results):
    # a partir de los resultados de analyze_modules, asignar el GO representativo a cada proteína

    rows = []
    for module in results:
        # Excluir el módulo de ruido de esta asignación si no queremos asignarle un GO representativo 'N/A'
        if module['module_id'] == -1: 
            continue # No asigna GO representativo a proteínas de ruido aquí

        rep_go = module['representative_go']
        if '(' in rep_go and ')' in rep_go:
            go_term = rep_go.split('(')[0].strip()
            term_name = rep_go.split('(')[1].replace(')', '').strip()
        else:
            go_term = rep_go # si es "N/A (No GO terms enriched)" o similar, se asigna tal cual
            term_name = ""

        for prot in module['module_proteins']:
            rows.append({
                'proteina': prot,
                'GO_term': go_term,
                'Term_Name_Clean': term_name
            })
    return pd.DataFrame(rows)

