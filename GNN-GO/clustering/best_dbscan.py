import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine # Importar distancia coseno
import seaborn as sns # Para mapas de calor

# --- Configuración de Rutas ---
# AJUSTA ESTA RUTA A LA UBICACIÓN EXACTA DE TU ARCHIVO embeddings_normalizados.csv
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

# --- Nueva definición de rangos de búsqueda mucho más amplia ---
# Redefinimos un poco el eps_range para tener más puntos en el rango bajo
# y cubrir un espectro más amplio pero con menor densidad en los extremos.
eps_range = np.concatenate((
    np.linspace(0.001, 0.05, 10), # Valores muy pequeños
    np.linspace(0.05, 0.2, 10),  # Rango cercano a lo que vimos en el k-distance plot
    np.linspace(0.2, 0.8, 10),   # Rango medio
    np.linspace(0.8, 1.5, 5)     # Valores más grandes
))
eps_range = np.sort(np.unique(np.round(eps_range, 4))) # Redondear para evitar decimales flotantes y asegurar unicidad
print(f"Nuevo rango de eps: [{eps_range.min():.4f}, {eps_range.max():.4f}] con {len(eps_range)} puntos.")


min_samples_range = np.unique(np.sort(np.concatenate((
    np.arange(2, 20, 2),  # Pequeños incrementos para valores bajos
    np.arange(20, 101, 10), # Mayores incrementos para valores altos
    np.arange(100, 201, 50) # Incluso valores más grandes
))))
print(f"Nuevo rango de min_samples: {min_samples_range}")

results = []
print("\n--- BÚSQUEDA DE PARÁMETROS PARA DBSCAN (RANGOS AMPLIADOS) ---")
print(f"Probando eps en el rango {eps_range.min():.4f}-{eps_range.max():.4f}")
print(f"Probando min_samples en el rango {min_samples_range[0]}-{min_samples_range[-1]}")

total_combinations = len(eps_range) * len(min_samples_range)
current_combination = 0

for eps_val in eps_range:
    for min_s_val in min_samples_range:
        current_combination += 1
        print(f"Probando DBSCAN con eps={eps_val:.4f}, min_samples={min_s_val} "
              f"({current_combination}/{total_combinations})...")
        
        dbscan = DBSCAN(eps=eps_val, min_samples=min_s_val, metric='cosine')
        clusters = dbscan.fit_predict(X)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0) # Número de clusters válidos (excluyendo ruido -1)
        n_noise = list(clusters).count(-1) # Número de puntos de ruido
        total_points = len(clusters)
        noise_percentage = (n_noise / total_points) * 100 if total_points > 0 else 0

        silhouette, davies_bouldin, calinski_harabasz = np.nan, np.nan, np.nan

        # Calcular métricas solo si hay al menos 2 clusters válidos y al menos 2 puntos en ellos
        if n_clusters >= 2:
            labels_filtered = clusters[clusters != -1]
            data_filtered = X[clusters != -1]

            if len(set(labels_filtered)) >= 2 and len(labels_filtered) >= 2:
                silhouette = silhouette_score(data_filtered, labels_filtered)
                davies_bouldin = davies_bouldin_score(data_filtered, labels_filtered)
                calinski_harabasz = calinski_harabasz_score(data_filtered, labels_filtered)
            else:
                # Esto sucede si, por ejemplo, los 2 clusters encontrados tienen solo 1 punto cada uno.
                # O si al filtrar el ruido, queda solo un cluster o menos de 2 puntos válidos.
                print(f"  Advertencia: Insuficientes clusters válidos o puntos (después de filtrar ruido) para métricas en eps={eps_val:.4f}, min_samples={min_s_val}. Métricas NaN.")
        else:
            print(f"  Advertencia: Menos de 2 clusters válidos (sin ruido) para eps={eps_val:.4f}, min_samples={min_s_val}. Métricas NaN.")

        results.append({
            'eps': eps_val,
            'min_samples': min_s_val,
            'Num Clusters (valid)': n_clusters,
            'Num Ruido': n_noise,
            'Porcentaje Ruido': noise_percentage,
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Calinski-Harabasz Index': calinski_harabasz
        })

results_df = pd.DataFrame(results)

print("\n--- Tabla de Resultados Detallada de DBSCAN por Parámetros ---")
# Ordenar para mejor visualización y manejar NaNs al final
results_df_sorted = results_df.sort_values(by=['Silhouette Score', 'Num Clusters (valid)'], ascending=[False, False])
print(results_df_sorted.to_string(index=False)) # Imprimir tabla completa
results_df_sorted.to_csv(os.path.join(RESULTS_GRAPH_DIR, 'dbscan_parameter_search_results_extended.csv'), index=False)
print(f"\nResultados detallados guardados en '{os.path.join(RESULTS_GRAPH_DIR, 'dbscan_parameter_search_results_extended.csv')}'")

# --- Visualización de Resultados (Mapas de Calor) ---
print("\nGenerando gráficos de resultados para DBSCAN...")

# Filtrar resultados donde se puedan calcular métricas (al menos 2 clusters válidos)
plot_df = results_df[results_df['Num Clusters (valid)'] >= 2].copy()

if not plot_df.empty:
    # Convertir 'min_samples' a string o categoría si es necesario para el mapa de calor,
    # aunque seaborn suele manejarlo bien como numérico si los valores son discretos.
    # Si 'min_samples' tiene muchos valores, es mejor agrupar o elegir un subconjunto
    # para la visualización o usar una escala logarítmica si son continuos.
    # Aquí, como son discretos y no demasiados, se puede usar directamente.

    # Pivotear el DataFrame para crear una matriz para el mapa de calor
    # Silhouette Score
    pivot_silhouette = plot_df.pivot_table(index='min_samples', columns='eps', values='Silhouette Score')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_silhouette, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title('DBSCAN: Silhouette Score por (eps, min_samples)', fontsize=16)
    plt.xlabel('Eps', fontsize=12)
    plt.ylabel('Min Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_GRAPH_DIR, "dbscan_silhouette_heatmap.png"), dpi=300)
    plt.close()

    # Davies-Bouldin Index
    pivot_davies = plot_df.pivot_table(index='min_samples', columns='eps', values='Davies-Bouldin Index')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_davies, annot=True, cmap='cividis_r', fmt=".2f", linewidths=.5) # _r para invertir el cmap (valores bajos son mejores)
    plt.title('DBSCAN: Davies-Bouldin Index por (eps, min_samples)', fontsize=16)
    plt.xlabel('Eps', fontsize=12)
    plt.ylabel('Min Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_GRAPH_DIR, "dbscan_davies_bouldin_heatmap.png"), dpi=300)
    plt.close()

    # Calinski-Harabasz Index
    pivot_calinski = plot_df.pivot_table(index='min_samples', columns='eps', values='Calinski-Harabasz Index')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_calinski, annot=True, cmap='magma', fmt=".2f", linewidths=.5)
    plt.title('DBSCAN: Calinski-Harabasz Index por (eps, min_samples)', fontsize=16)
    plt.xlabel('Eps', fontsize=12)
    plt.ylabel('Min Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_GRAPH_DIR, "dbscan_calinski_harabasz_heatmap.png"), dpi=300)
    plt.close()
    
    # Número de Clusters Válidos
    pivot_num_clusters = results_df.pivot_table(index='min_samples', columns='eps', values='Num Clusters (valid)')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_num_clusters, annot=True, cmap='Blues', fmt="g", linewidths=.5)
    plt.title('DBSCAN: Número de Clusters Válidos por (eps, min_samples)', fontsize=16)
    plt.xlabel('Eps', fontsize=12)
    plt.ylabel('Min Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_GRAPH_DIR, "dbscan_num_clusters_heatmap.png"), dpi=300)
    plt.close()

    # Porcentaje de Ruido
    pivot_noise_percentage = results_df.pivot_table(index='min_samples', columns='eps', values='Porcentaje Ruido')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_noise_percentage, annot=True, cmap='Reds', fmt=".1f", linewidths=.5)
    plt.title('DBSCAN: Porcentaje de Ruido por (eps, min_samples)', fontsize=16)
    plt.xlabel('Eps', fontsize=12)
    plt.ylabel('Min Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_GRAPH_DIR, "dbscan_noise_percentage_heatmap.png"), dpi=300)
    plt.close()

else:
    print("\nNo hay suficientes combinaciones de parámetros que resulten en al menos 2 clusters válidos para generar mapas de calor.")
    print("Esto es un fuerte indicio de que DBSCAN podría no ser el algoritmo más adecuado para la estructura de tus datos.")
    print("Considera probar HDBSCAN o K-Means.")


# Determinación de los Mejores Parámetros para DBSCAN
print("\n--- Determinación de los Mejores Parámetros para DBSCAN Sugeridos por Cada Métrica ---")

# Filtraremos solo los resultados donde se pudieron calcular las métricas
best_results_filtered = results_df[results_df['Num Clusters (valid)'] >= 2].copy()

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

    # Una sugerencia de "óptimo" podría ser la combinación que maximiza Silhouette (más robusto)
    optimal_dbscan_params_suggested = best_by_silhouette[['eps', 'min_samples']].to_dict()
    print(f"\n--- DBSCAN | Parámetros sugeridos para usar (basado en Silhouette Score): ---")
    print(f"Eps: {optimal_dbscan_params_suggested['eps']:.4f}, Min Samples: {optimal_dbscan_params_suggested['min_samples']}")
else:
    optimal_dbscan_params_suggested = None
    print("\nNo se encontraron combinaciones de parámetros con al menos 2 clusters válidos para sugerir un óptimo.")

print("\nAnálisis de optimización de DBSCAN completado. ")