import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
import umap.umap_ as umap # Para reducción de dimensionalidad y visualización
import sys

# --- Configuración de Rutas ---
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
EMBEDDINGS_PATH = os.path.join(BASE_OUTPUT_DIR, "Trial_2", "embeddings_normalizados.csv")

# Directorio para guardar los resultados del clustering final
# Se creará una subcarpeta específica para los resultados finales de HDBSCAN
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
FINAL_CLUSTER_RESULTS_DIR = os.path.join(SCRIPT_DIR, "Trial_2", "final_hdbscan_clusters")
os.makedirs(FINAL_CLUSTER_RESULTS_DIR, exist_ok=True)

# --- Carga de Embeddings Normalizados ---
print(f"Cargando embeddings normalizados desde: {EMBEDDINGS_PATH}")
try:
    df_embeddings = pd.read_csv(EMBEDDINGS_PATH)
    protein_ids = df_embeddings.iloc[:, 0]
    X = df_embeddings.iloc[:, 1:].values
    print("Embeddings normalizados cargados exitosamente.")
    
    # Verificación final de normalización L2
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        print("Advertencia: Los embeddings no parecen estar normalizados a L2=1.0. Normalizando L2.")
        X = X / norms[:, np.newaxis]
    else:
        print("Los embeddings están normalizados a L2=1.0. ¡Perfecto para distancia coseno!")

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de embeddings en la ruta: {EMBEDDINGS_PATH}")
    print("Por favor, verifica que la ruta sea correcta y que el archivo exista.")
    print("Se cargarán embeddings de ejemplo para la ejecución para evitar un error.")
    np.random.seed(42)
    X = np.random.rand(5000, 768) # 5000 puntos de ejemplo, 768 dimensiones
    X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalizar L2
    protein_ids = pd.Series([f"Protein_{i}" for i in range(X.shape[0])]) # IDs de ejemplo
except Exception as e:
    print(f"Ocurrió un error al cargar los embeddings: {e}")
    print("Se cargarán embeddings de ejemplo para la ejecución para evitar un error.")
    np.random.seed(42)
    X = np.random.rand(5000, 768) # 5000 puntos de ejemplo, 768 dimensiones
    X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalizar L2
    protein_ids = pd.Series([f"Protein_{i}" for i in range(X.shape[0])]) # IDs de ejemplo

total_points = X.shape[0]

if total_points < 2:
    print("ERROR: No hay suficientes muestras en los embeddings para realizar clustering.")
    sys.exit()

# --- Definir el min_cluster_size óptimo ---
# Basado en tu análisis de los gráficos de optimización de HDBSCAN,
# donde el Silhouette Score es alto y el número de clústeres es manejable.
# Yo sugeriría probar con 50 o 100 como buen punto de partida.
# AJUSTA ESTE VALOR según tu interpretación de los gráficos.
OPTIMAL_MIN_CLUSTER_SIZE = 20 
OPTIMAL_MIN_SAMPLES = 20
OPTIMAL_CLUSTER_SELECTION_EPSILON = 0.0
OPTIMAL_ALPHA = 2.0


print(f"\n--- Ejecutando HDBSCAN con parámetros óptimos ---")
print(f"  min_cluster_size = {OPTIMAL_MIN_CLUSTER_SIZE}")
print(f"  min_samples = {OPTIMAL_MIN_SAMPLES}")
print(f"  cluster_selection_epsilon = {OPTIMAL_CLUSTER_SELECTION_EPSILON:.2f}")
print(f"  alpha = {OPTIMAL_ALPHA:.1f}")
# --- Aplicar HDBSCAN ---
# Usamos metric='euclidean' porque los embeddings están normalizados L2,
# lo que hace que la distancia euclidiana sea proporcional a la distancia coseno.
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=OPTIMAL_MIN_CLUSTER_SIZE,
    min_samples=OPTIMAL_MIN_SAMPLES, # Puede ser igual a min_cluster_size o menor
    cluster_selection_epsilon=OPTIMAL_CLUSTER_SELECTION_EPSILON, # Por defecto 0.0
    alpha=OPTIMAL_ALPHA, # Por defecto 1.0, ajusta si ves
    metric='euclidean', # Usamos euclidiana ya que los embeddings están normalizados L2
    cluster_selection_method='eom', # 'eom' (Excess of Mass) o 'leaf' (más granular)
    prediction_data=True, # Necesario si quieres clasificar nuevos puntos o estimar confianza
    # Puede ser útil ajustar min_samples si min_cluster_size es muy grande, pero
    # por defecto, min_samples es igual a min_cluster_size, lo cual es sensato.
    # min_samples=OPTIMAL_MIN_CLUSTER_SIZE 
)

print("Entrenando y prediciendo clústeres con HDBSCAN...")
labels = clusterer.fit_predict(X)

# --- Análisis de Resultados y Métricas ---
n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0) # Excluir el clúster de ruido (-1)
n_noise = list(labels).count(-1)
noise_percentage = (n_noise / total_points) * 100 if total_points > 0 else 0

print(f"\n--- Resultados de HDBSCAN ({OPTIMAL_MIN_CLUSTER_SIZE} min_cluster_size) ---")
print(f"Número total de puntos: {total_points}")
print(f"Número de clústeres válidos encontrados: {n_clusters}")
print(f"Número de puntos clasificados como ruido: {n_noise} ({noise_percentage:.2f}%)")

silhouette, davies_bouldin, calinski_harabasz = np.nan, np.nan, np.nan

if n_clusters >= 2:
    # Filtrar los puntos de ruido antes de calcular las métricas intrínsecas
    labels_filtered = labels[labels != -1]
    data_filtered = X[labels != -1]

    if len(set(labels_filtered)) >= 2 and len(labels_filtered) >= 2:
        silhouette = silhouette_score(data_filtered, labels_filtered, metric='cosine') # Usar coseno para Silhouette
        davies_bouldin = davies_bouldin_score(data_filtered, labels_filtered)
        calinski_harabasz = calinski_harabasz_score(data_filtered, labels_filtered)
    else:
        print("Advertencia: Insuficientes clústeres válidos o puntos (después de filtrar ruido) para calcular métricas. Se usarán NaN.")
else:
    print("Advertencia: Menos de 2 clústeres válidos (sin ruido) para calcular métricas. Se usarán NaN.")

print(f"Silhouette Score (excluyendo ruido): {silhouette:.4f}")
print(f"Davies-Bouldin Index (excluyendo ruido): {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index (excluyendo ruido): {calinski_harabasz:.4f}")

# --- Guardar Resultados del Clustering ---

# 1. Etiquetas de clúster por ID de proteína
df_results = pd.DataFrame({
    'protein_id': protein_ids,
    'cluster_label': labels
})
output_labels_path = os.path.join(FINAL_CLUSTER_RESULTS_DIR,
                                  f'hdbscan_cluster_labels_mcs{OPTIMAL_MIN_CLUSTER_SIZE}_ms{OPTIMAL_MIN_SAMPLES}_cse{OPTIMAL_CLUSTER_SELECTION_EPSILON:.2f}_alpha{OPTIMAL_ALPHA:.1f}.csv')
df_results.to_csv(output_labels_path, index=False)
print(f"\nEtiquetas de clúster guardadas en: {output_labels_path}")

# 2. Resumen de clústeres
cluster_summary = []
unique_labels = np.unique(labels)
for label in unique_labels:
    if label == -1: # Ignorar el clúster de ruido para este resumen
        continue
    
    cluster_points = X[labels == label]
    cluster_ids = protein_ids[labels == label]
    
    size = len(cluster_points)
    
    # Calcular métricas internas para cada clúster (opcional, puede ser costoso para muchos clústeres)
    # Por ahora, solo tamaño
    
    cluster_summary.append({
        'cluster_id': label,
        'size': size,
        # 'mean_silhouette_local': np.mean(cluster_silhouette_values) # Requeriría calcular silhouette_samples
    })

df_cluster_summary = pd.DataFrame(cluster_summary).sort_values(by='size', ascending=False)
output_summary_path = os.path.join(FINAL_CLUSTER_RESULTS_DIR,
                                   f'hdbscan_cluster_summary_mcs{OPTIMAL_MIN_CLUSTER_SIZE}_ms{OPTIMAL_MIN_SAMPLES}_cse{OPTIMAL_CLUSTER_SELECTION_EPSILON:.2f}_alpha{OPTIMAL_ALPHA:.1f}.csv')
df_cluster_summary.to_csv(output_summary_path, index=False)
print(f"Resumen de clústeres guardado en: {output_summary_path}")


# --- Visualización de Clústeres con UMAP ---
print("\nRealizando reducción de dimensionalidad con UMAP para visualización...")

# UMAP es excelente para embeddings y preserva mejor la estructura global que t-SNE.
# Ajusta n_components (2 o 3), n_neighbors y min_dist según la densidad deseada en la visualización.
reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine', n_jobs=-1)
# Usamos los embeddings filtrados (sin ruido) para UMAP si el ruido es muy alto y confunde la visualización
# O podemos usar todos los embeddings y colorear el ruido aparte.
# Para esta visualización, incluiremos el ruido para ver su distribución.

X_reduced = reducer.fit_transform(X)

plt.figure(figsize=(12, 10))

# Mapear las etiquetas de clúster a un mapa de colores
unique_cluster_labels = np.unique(labels)
#colors = plt.cm.get_cmap('tab20', len(unique_cluster_labels) - (1 if -1 in unique_cluster_labels else 0)) # tab20 para hasta 20 clústeres.
#colors = plt.cm.get_cmap('tab10', len(unique_cluster_labels) - (1 if -1 in unique_cluster_labels else 0))

# Colores para los clústeres (ejemplo de una paleta de 10 colores que contrastan bien)
cluster_colors_list = [
    '#1f77b4',  # Azul fuerte (Clúster 0)
    '#ff7f0e',  # Naranja (Clúster 1)
    '#2ca02c',  # Verde (Clúster 2)
    '#d62728',  # Rojo (Clúster 3)
    '#9467bd',  # Morado (Clúster 4)
    '#8c564b',  # Marrón (Clúster 5)
    '#e377c2',  # Rosa (Clúster 6)
    '#7f7f7f',  # Gris oscuro (para clúster 7, si lo hubiera)
    '#bcbd22',  # Amarillo-verde (para clúster 8)
    '#17becf',   # Cian (para clúster 9)
    '#1f77b4',  # Azul fuerte (Clúster 10)
    '#fab4b4',  # Rosa claro (Clúster 11)
    '#025570',  # Azul oscuro (Clúster 12)
    '#d407e3',  # Magenta (Clúster 13)
    '#c7fabb',  # Verde claro (Clúster 14)
    '#f2c0ac',  # Melocotón (Clúster 15)
]

# Color para el ruido
noise_color = 'lightgray' # Un gris claro para que el ruido sea menos dominante

# Mapear las etiquetas de clúster a los colores definidos
# Se creará un diccionario para asignar colores a los IDs de clúster
color_map = {}
cluster_id_counter = 0
for label in sorted(unique_cluster_labels):
    if label == -1:
        color_map[label] = noise_color
    else:
        # Asigna un color de la lista, ciclando si hay más clústeres que colores definidos
        color_map[label] = cluster_colors_list[cluster_id_counter % len(cluster_colors_list)]
        cluster_id_counter += 1

# Primero, graficar los puntos de ruido
noise_indices = (labels == -1)
if np.any(noise_indices):
    plt.scatter(X_reduced[noise_indices, 0], X_reduced[noise_indices, 1],
                color=color_map[-1], s=5, alpha=0.5, label='Ruido (-1)')

# Luego, graficar los puntos de cada clúster
for label in sorted(unique_cluster_labels):
    if label == -1:
        continue # Ya graficamos el ruido
    
    cluster_indices = (labels == label)
    plt.scatter(X_reduced[cluster_indices, 0], X_reduced[cluster_indices, 1],
                color=color_map[label], s=10, alpha=0.7, label=f'Clúster {label}')


plt.title(f'HDBSCAN Clusters (min_cluster_size={OPTIMAL_MIN_CLUSTER_SIZE}) - UMAP 2D', fontsize=16)
plt.xlabel('UMAP Component 1', fontsize=12)
plt.ylabel('UMAP Component 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar layout para hacer espacio para la leyenda
output_umap_plot_path = os.path.join(FINAL_CLUSTER_RESULTS_DIR,
                                     f'hdbscan_umap_clusters_mcs{OPTIMAL_MIN_CLUSTER_SIZE}_ms{OPTIMAL_MIN_SAMPLES}_cse{OPTIMAL_CLUSTER_SELECTION_EPSILON:.2f}_alpha{OPTIMAL_ALPHA:.1f}.png')
plt.savefig(output_umap_plot_path, dpi=300)
plt.close()
print(f"Visualización UMAP guardada en: {output_umap_plot_path}")

print("\n--- Proceso de Clustering HDBSCAN (Parte 2) Completado ---")
print("Revisa los archivos CSV y el gráfico de UMAP para el análisis final de los clústeres.")