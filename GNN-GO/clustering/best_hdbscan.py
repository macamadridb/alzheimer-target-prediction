import numpy as np
import pandas as pd
import hdbscan # Importar HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import os
import seaborn as sns # Para mapas de calor y visualizaciones
import time 

# --- Configuración de Rutas ---
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
EMBEDDINGS_PATH = os.path.join(BASE_OUTPUT_DIR, "Trial_2", "embeddings_normalizados.csv")

# Directorio para guardar los gráficos de resultados (se creará si no existe)
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
RESULTS_GRAPH_DIR = os.path.join(SCRIPT_DIR, "Trial_2", "clustering_optimization_results")
os.makedirs(RESULTS_GRAPH_DIR, exist_ok=True)


# --- Carga de Embeddings Normalizados ---
print(f"Cargando embeddings normalizados desde: {EMBEDDINGS_PATH}")
try:
    df_embeddings = pd.read_csv(EMBEDDINGS_PATH)
    # Asumimos que la primera columna es ID y el resto son embeddings
    protein_ids = df_embeddings.iloc[:, 0]
    X = df_embeddings.iloc[:, 1:].values
    print("Embeddings normalizados cargados exitosamente.")
    print(f"Primeras 5 IDs de proteína: \n{protein_ids.head().to_string()}")
    print(f"Primeras 5 filas de embeddings (fragmento):\n{X[:5, :5]}") # Mostrar un fragmento
    
    # Verificación de normalización
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        print("Advertencia: Los embeddings no parecen estar normalizados a L2=1.0.")
        print("Normalizando L2 los embeddings para el cálculo de distancia coseno.")
        X = X / norms[:, np.newaxis] # Normalizar si no lo están
    else:
        print("Los embeddings están normalizados a L2=1.0. ¡Perfecto para distancia coseno!")

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de embeddings en la ruta: {EMBEDDINGS_PATH}")
    print("Por favor, verifica que la ruta sea correcta y que el archivo exista.")
    print("Se cargarán embeddings de ejemplo para la ejecución para evitar un error.")
    np.random.seed(42)
    X = np.random.rand(100, 768) # 100 puntos, 768 dimensiones
    X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalizar para distancia coseno
except Exception as e:
    print(f"Ocurrió un error al cargar los embeddings: {e}")
    print("Se cargarán embeddings de ejemplo para la ejecución para evitar un error.")
    np.random.seed(42)
    X = np.random.rand(100, 768) # 100 puntos, 768 dimensiones
    X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalizar para distancia coseno

print(f"\nDimensiones de los embeddings (X): {X.shape}")

# --- Búsqueda de Parámetros para HDBSCAN ---
print("\n--- BÚSQUEDA DE PARÁMETROS PARA HDBSCAN ---")

# HDBSCAN tiene principalmente dos parámetros clave para ajustar:
# 1. min_cluster_size: El número mínimo de puntos que se considera que forman un clúster.
# 2. min_samples: (Opcional, pero se puede ajustar) Similar al min_samples de DBSCAN,
#    controla la robustez de un punto como "punto central" o "ruido".
#    Si se establece en None (por defecto), es igual a min_cluster_size.
#    Ajustar min_samples a un valor menor que min_cluster_size puede hacer los clústeres más "sueltos".
# 3. cluster_selection_epsilon: (Opcional) Controla la distancia máxima entre puntos para que se consideren parte del mismo clúster.
# 4. alpha: (Opcional) Controla la suavidad del modelo de densidad. Por defecto es 1.0.

# Rango para min_cluster_size:
# Empezaremos con valores pequeños y subiremos, hasta un cierto porcentaje del dataset si es grande.
max_min_cluster_size = min(200, X.shape[0] // 10) # No ir más allá del 10% del dataset o 200
if X.shape[0] < 2:
    print("ERROR: No hay suficientes muestras para HDBSCAN.")
    exit()
elif X.shape[0] < 5: # min_cluster_size debe ser al menos 2
    min_cluster_size_range = np.arange(2, X.shape[0] + 1)
else:
    min_cluster_size_range = np.unique(np.sort(np.concatenate((
        np.arange(2, 20, 2),    # Pequeños incrementos para valores bajos
        np.arange(20, 101, 10),  # Mayores incrementos para valores medios
        np.arange(100, max_min_cluster_size + 1, 50) # Incrementos más grandes para valores altos
    ))))
print(f"Rango de min_cluster_size: {min_cluster_size_range}")

# Rango para min_samples:
# Ahora exploraremos min_samples de forma independiente
# Puedes definir este rango de acuerdo a lo que tenga sentido para tus datos.
# Es común que min_samples <= min_cluster_size.
# Si lo pones None, se usará el valor de min_cluster_size

min_samples_range = np.unique(np.sort(np.concatenate((
    np.arange(1, 10, 1),   # Valores muy pequeños, incluyendo 1 (menos estricto que default)
    np.arange(10, 50, 5),  # Valores medios
    np.arange(50, 101, 25) # Valores más grandes
))))
# Asegurarse de que min_samples no sea mayor que la cantidad de datos
min_samples_range = min_samples_range[min_samples_range < X.shape[0]]

print(f"Rango de min_samples: {min_samples_range}")

# Rango para cluster_selection_epsilon:
# Valores de 0.0 (por defecto) y algunos otros valores para explorar fusión de clústeres.
# Cuidado: valores muy grandes pueden fusionar todo en un solo clúster.
cluster_selection_epsilon_range = [0.0]  # Solo el valor por defecto
print(f"Rango de cluster_selection_epsilon: {cluster_selection_epsilon_range}")

# Rango para alpha:
# Valores alrededor de 1.0 (por defecto), y otros para ver cómo afecta el MST.
alpha_range = np.unique(np.sort(np.array([0.5, 1.0, 1.5, 2.0])))
print(f"Rango de alpha: {alpha_range}")

hdbscan_results = []

total_combinations = len(min_cluster_size_range) * len(min_samples_range) * len(cluster_selection_epsilon_range) * len(alpha_range)
current_combination = 0

start_time = time.time()

for min_size_val in min_cluster_size_range:
    for min_samples_val in min_samples_range:
        # Filtro opcional: min_samples no debería ser mayor que min_cluster_size
        # Puedes quitar este 'continue' si quieres explorar esas combinaciones,
        # pero HDBSCAN internamente las ajustaría, lo que puede ser redundante.
        if min_samples_val > min_size_val:
            # print(f"  Saltando combinación: min_samples ({min_samples_val}) > min_cluster_size ({min_size_val})")
            continue 

        for cluster_eps_val in [0.0]: # TERCER BUCLE
            for alpha_val in alpha_range: # CUARTO BUCLE
                current_combination += 1
                
                print(f"Probando HDBSCAN con: mcs={min_size_val}, ms={min_samples_val}, "
                      f"cse={cluster_eps_val:.2f}, alpha={alpha_val:.1f} "
                      f"({current_combination}/{total_combinations})...")
                
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size_val,
                                            min_samples=min_samples_val,
                                            cluster_selection_epsilon=cluster_eps_val,
                                            alpha=alpha_val,
                                            metric='euclidean', # Usar la métrica adecuada
                                            prediction_data=True)
                labels = clusterer.fit_predict(X)


                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                total_points = len(labels)
                noise_percentage = (n_noise / total_points) * 100 if total_points > 0 else 0

                silhouette, davies_bouldin, calinski_harabasz = np.nan, np.nan, np.nan

                if n_clusters >= 2:
                    labels_filtered = labels[labels != -1]
                    data_filtered = X[labels != -1]

                    if len(set(labels_filtered)) >= 2 and len(labels_filtered) >= 2:
                        silhouette = silhouette_score(data_filtered, labels_filtered)
                        davies_bouldin = davies_bouldin_score(data_filtered, labels_filtered)
                        calinski_harabasz = calinski_harabasz_score(data_filtered, labels_filtered)
                    else:
                        # print(f"  Advertencia: Insuficientes clusters válidos o puntos (después de filtrar ruido) para métricas en esta combinación. Métricas NaN.")
                        pass # No imprimir si ocurre mucho
                else:
                    # print(f"  Advertencia: Menos de 2 clusters válidos (sin ruido) para esta combinación. Métricas NaN.")
                    pass # No imprimir si ocurre mucho

                hdbscan_results.append({
                    'min_cluster_size': min_size_val,
                    'min_samples': min_samples_val,
                    'cluster_selection_epsilon': cluster_eps_val,
                    'alpha': alpha_val,
                    'Num Clusters (valid)': n_clusters,
                    'Num Ruido': n_noise,
                    'Porcentaje Ruido': noise_percentage,
                    'Silhouette Score': silhouette,
                    'Davies-Bouldin Index': davies_bouldin,
                    'Calinski-Harabasz Index': calinski_harabasz
                })
end_time = time.time()
print(f"\nOptimización completa en {end_time - start_time:.2f} segundos.")



hdbscan_df = pd.DataFrame(hdbscan_results)

print("\n--- Tabla de Resultados Detallada de HDBSCAN por Parámetros (Exhaustiva) ---")
hdbscan_df_sorted = hdbscan_df.sort_values(by=['Silhouette Score', 'Num Clusters (valid)'], ascending=[False, False])
print(hdbscan_df_sorted.to_string(index=False))
hdbscan_df_sorted.to_csv(os.path.join(RESULTS_GRAPH_DIR, 'hdbscan_parameter_search_results_exhaustive.csv'), index=False)
print(f"\nResultados detallados guardados en '{os.path.join(RESULTS_GRAPH_DIR, 'hdbscan_parameter_search_results_exhaustive.csv')}'")

# --- Visualización de Resultados (¡Ahora necesitas visualizar 4D!) ---
# Para visualizar 4D, es más complejo que un heatmap 2D.
# Opciones:
# 1. Heatmaps 2D fijos: Elegir un valor fijo para 2 parámetros y hacer un heatmap de los otros 2.
# 2. Scatter plots 3D: Con coloreado.
# 3. Pair plots (seaborn): Si hay pocas combinaciones.
# 4. Gráficos de línea con subplots para cada parámetro.

print("\nGenerando gráficos de resultados para HDBSCAN (análisis de 4 parámetros)...")

plot_df = hdbscan_df[hdbscan_df['Num Clusters (valid)'] >= 2].copy()

if not plot_df.empty:
    # Ejemplo de Heatmap 2D fijando cluster_selection_epsilon y alpha
    # Podrías generar varios de estos para diferentes valores fijos
    
    # Seleccionar un valor de cluster_selection_epsilon y alpha para el heatmap
    # Elijo el valor por defecto o uno que parezca prometedor del análisis de tabla.
    # Puedes ajustar estos valores manualmente después de ver los resultados de la tabla.
    
    # Intenta encontrar los valores con mejor Silhouette global
    if not hdbscan_df_sorted.empty:
        best_overall_row = hdbscan_df_sorted.iloc[0]
        fixed_cse = best_overall_row['cluster_selection_epsilon']
        fixed_alpha = best_overall_row['alpha']
        print(f"\nGenerando heatmaps fijando cluster_selection_epsilon={fixed_cse:.2f} y alpha={fixed_alpha:.1f}")

        filtered_plot_df = plot_df[
            (plot_df['cluster_selection_epsilon'] == fixed_cse) &
            (plot_df['alpha'] == fixed_alpha)
        ].copy()

        if not filtered_plot_df.empty:
            # Silhouette Score
            pivot_silhouette = filtered_plot_df.pivot_table(index='min_samples', columns='min_cluster_size', values='Silhouette Score')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_silhouette, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
            plt.title(f'HDBSCAN: Silhouette Score (cse={fixed_cse:.2f}, alpha={fixed_alpha:.1f})', fontsize=16)
            plt.xlabel('min_cluster_size', fontsize=12)
            plt.ylabel('min_samples', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_GRAPH_DIR, f"hdbscan_silhouette_heatmap_cse{fixed_cse:.2f}_alpha{fixed_alpha:.1f}.png"), dpi=300)
            plt.close()

            # Porcentaje de Ruido
            pivot_noise_percentage = filtered_plot_df.pivot_table(index='min_samples', columns='min_cluster_size', values='Porcentaje Ruido')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_noise_percentage, annot=True, cmap='Reds', fmt=".1f", linewidths=.5)
            plt.title(f'HDBSCAN: Porcentaje de Ruido (cse={fixed_cse:.2f}, alpha={fixed_alpha:.1f})', fontsize=16)
            plt.xlabel('min_cluster_size', fontsize=12)
            plt.ylabel('min_samples', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_GRAPH_DIR, f"hdbscan_noise_percentage_heatmap_cse{fixed_cse:.2f}_alpha{fixed_alpha:.1f}.png"), dpi=300)
            plt.close()
        else:
            print(f"No hay datos para generar heatmaps para cse={fixed_cse:.2f} y alpha={fixed_alpha:.1f}")

    # Puedes añadir más lógicas para generar otros heatmaps fijando diferentes pares de parámetros.
    # O considerar otros tipos de gráficos si los heatmaps se vuelven demasiado específicos.

else:
    print("\nNo hay suficientes combinaciones de parámetros que resulten en al menos 2 clusters válidos para generar gráficos.")

# Determinación de los Mejores Parámetros para HDBSCAN (AHORA CON TODOS LOS PARÁMETROS)
print("\n--- Determinación de los Mejores Parámetros para HDBSCAN Sugeridos por Cada Métrica (Exhaustiva) ---")

best_results_filtered = hdbscan_df[hdbscan_df['Num Clusters (valid)'] >= 2].copy()

if not best_results_filtered.empty:
    # Ordenar por Silhouette Score (máximo es mejor)
    best_by_silhouette = best_results_filtered.loc[best_results_filtered['Silhouette Score'].idxmax()]
    print(f"\n--- Mejores parámetros basados en Silhouette Score (más alto) ---")
    print(best_by_silhouette)

    # Ordenar por Davies-Bouldin Index (mínimo es mejor)
    best_by_davies_bouldin = best_results_filtered.loc[best_results_filtered['Davies-Bouldin Index'].idxmin()]
    print(f"\n--- Mejores parámetros basados en Davies-Bouldin Index (más bajo) ---")
    print(best_by_davies_bouldin)

    # Ordenar por Calinski-Harabasz Index (máximo es mejor)
    best_by_calinski_harabasz = best_results_filtered.loc[best_results_filtered['Calinski-Harabasz Index'].idxmax()]
    print(f"\n--- Mejores parámetros basados en Calinski-Harabasz Index (más alto) ---")
    print(best_by_calinski_harabasz)

    # Sugerencia final de parámetros óptimos
    optimal_hdbscan_params_suggested = best_by_silhouette[['min_cluster_size', 'min_samples', 'cluster_selection_epsilon', 'alpha']].to_dict()
    print(f"\n--- HDBSCAN | Parámetros sugeridos para usar (basado en Silhouette Score): ---")
    print(f"min_cluster_size: {optimal_hdbscan_params_suggested['min_cluster_size']}, "
          f"min_samples: {optimal_hdbscan_params_suggested['min_samples']}, "
          f"cluster_selection_epsilon: {optimal_hdbscan_params_suggested['cluster_selection_epsilon']:.2f}, "
          f"alpha: {optimal_hdbscan_params_suggested['alpha']:.1f}")
else:
    optimal_hdbscan_params_suggested = None
    print("\nNo se encontraron combinaciones de parámetros con al menos 2 clusters válidos para sugerir un óptimo.")

print("\nAnálisis de optimización de HDBSCAN completado. ")