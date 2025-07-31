import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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
    df_normalized_embeddings = pd.read_csv(EMBEDDINGS_PATH)
    print("Embeddings normalizados cargados exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de embeddings normalizados en '{EMBEDDINGS_PATH}'.")
    print("Por favor, verifica la ruta y asegúrate de que el paso de normalización haya creado este archivo.")
    exit()

protein_ids = df_normalized_embeddings.iloc[:, 0]
embedding_columns = df_normalized_embeddings.columns[1:]
embeddings_data = df_normalized_embeddings[embedding_columns].values

print(f"Forma de los embeddings cargados: {embeddings_data.shape}")
print(f"Primeras 5 IDs de proteína: \n{protein_ids.head().to_string()}")
print(f"Primeras 5 filas de embeddings (fragmento):\n{embeddings_data[:5, :5]}")

l2_norms_loaded = np.linalg.norm(embeddings_data, axis=1)
print(f"\n--- Verificación de Normas L2 de los Embeddings CARGADOS (¡Deberían ser ~1.0!) ---")
print(f"Mínima norma L2: {np.min(l2_norms_loaded):.10f}")
print(f"Máxima norma L2: {np.max(l2_norms_loaded):.10f}")
print(f"Media de norma L2: {np.mean(l2_norms_loaded):.10f}")
print(f"Desviación estándar de norma L2: {np.std(l2_norms_loaded):.10f}")
if np.isclose(np.mean(l2_norms_loaded), 1.0, atol=1e-5) and np.isclose(np.std(l2_norms_loaded), 0.0, atol=1e-5):
    print("Las normas L2 de los embeddings cargados están correctamente normalizadas a ~1.0.")
else:
    print("ADVERTENCIA: Las normas L2 de los embeddings cargados NO están centradas en 1.0. Revisa tu proceso de normalización.")


# --- Búsqueda del Mejor K para K-Means ---
print("\n--- BÚSQUEDA DEL MEJOR K PARA K-MEANS ---")
# Definir el rango de K a probar
max_k_kmeans = min(50, embeddings_data.shape[0] - 1)
if max_k_kmeans < 2:
    print("ERROR: No hay suficientes muestras para evaluar K-Means con al menos 2 clusters.")
    exit()

k_range = range(2, max_k_kmeans + 1)

kmeans_results = []

for k in k_range:
    print(f"Probando K-Means con K = {k}...")
    # Asegúrate de que n_init='auto' o un número explícito para versiones recientes de sklearn
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings_data)
    inertia = kmeans.inertia_

    # Las métricas requieren al menos 2 clusters únicos y más de 1 muestra en total
    num_unique_clusters = len(np.unique(labels))
    if num_unique_clusters >= 2 and embeddings_data.shape[0] > 1:
        silhouette = silhouette_score(embeddings_data, labels)
        davies_bouldin = davies_bouldin_score(embeddings_data, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings_data, labels)
    else:
        # Esto es raro si KMeans converge correctamente con n_init='auto'
        # Pero es un fallback si KMeans solo encuentra 1 cluster o menos de 2 únicos.
        print(f"  Advertencia: K-Means con K={k} resultó en menos de 2 clusters únicos o pocas muestras. Métricas NaN.")
        silhouette, davies_bouldin, calinski_harabasz = np.nan, np.nan, np.nan

    kmeans_results.append({
        'K': k,
        'Inertia': inertia,
        'Silhouette Score': silhouette,
        'Davies-Bouldin Index': davies_bouldin,
        'Calinski-Harabasz Index': calinski_harabasz
    })

kmeans_df = pd.DataFrame(kmeans_results)

print("\n--- Tabla de Resultados de K-Means por K ---")
print(kmeans_df.to_string(index=False)) # Imprimir tabla a la terminal

# Gráficos K-Means
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
plt.plot(kmeans_df['K'], kmeans_df['Inertia'], marker='o', linestyle='-')
plt.title('K-Means: Método del Codo (Inercia)', fontsize=14)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inercia', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(list(k_range))

plt.subplot(2, 2, 2)
plt.plot(kmeans_df['K'], kmeans_df['Silhouette Score'], marker='o', linestyle='-', color='green')
plt.title('K-Means: Silhouette Score', fontsize=14)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Puntuación Silhouette', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(list(k_range))
plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1)

plt.subplot(2, 2, 3)
plt.plot(kmeans_df['K'], kmeans_df['Davies-Bouldin Index'], marker='o', linestyle='-', color='red')
plt.title('K-Means: Davies-Bouldin Index', fontsize=14)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Índice Davies-Bouldin', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(list(k_range))

plt.subplot(2, 2, 4)
plt.plot(kmeans_df['K'], kmeans_df['Calinski-Harabasz Index'], marker='o', linestyle='-', color='purple')
plt.title('K-Means: Calinski-Harabasz Index', fontsize=14)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Índice Calinski-Harabasz', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(list(k_range))

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.suptitle('Evaluación de Métricas de K-Means por K', y=0.99, fontsize=16)
plt.savefig(os.path.join(RESULTS_GRAPH_DIR, "kmeans_k_evaluation_metrics.png"), dpi=300)
plt.close() # Cierra la figura después de guardarla

# Determinación del Mejor K para K-Means
print("\n--- Determinación del Mejor K para K-Means Sugerido por Cada Métrica ---")
kmeans_filtered = kmeans_df.dropna()
best_k_silhouette = kmeans_filtered.loc[kmeans_filtered['Silhouette Score'].idxmax(), 'K'] if not kmeans_filtered['Silhouette Score'].empty else pd.NA
best_k_davies_bouldin = kmeans_filtered.loc[kmeans_filtered['Davies-Bouldin Index'].idxmin(), 'K'] if not kmeans_filtered['Davies-Bouldin Index'].empty else pd.NA
best_k_calinski_harabasz = kmeans_filtered.loc[kmeans_filtered['Calinski-Harabasz Index'].idxmax(), 'K'] if not kmeans_filtered['Calinski-Harabasz Index'].empty else pd.NA

print(f"K-Means | Mejor K por Silhouette Score (más alto): {best_k_silhouette}")
print(f"K-Means | Mejor K por Davies-Bouldin Index (más bajo): {best_k_davies_bouldin}")
print(f"K-Means | Mejor K por Calinski-Harabasz Index (más alto): {best_k_calinski_harabasz}")
if best_k_silhouette is not pd.NA:
    optimal_k_kmeans_suggested = best_k_silhouette
    print(f"\nK-Means | K sugerido para usar (basado en Silhouette): {optimal_k_kmeans_suggested}")
else:
    optimal_k_kmeans_suggested = None
    print("\nK-Means | No se pudo sugerir un K óptimo automáticamente.")


print("\nAnálisis de optimización de KMEANS completado. ")