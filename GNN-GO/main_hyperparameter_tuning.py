import os
import platform
import psutil
import torch
import time
import pandas as pd
import numpy as np
import random
import optuna
from optuna.pruners import MedianPruner
# from optuna.samplers import TPESampler # TPESampler es el sampler por defecto 

# Optuna: biblioteca para la optimización de hiperparámetros
# Optuna.pruners.MedianPruner: un podador que detiene las pruebas si su rendimiento es significativamente peor que el de otras pruebas en la misma etapa
# Optuna.samplers.TPESampler: un muestreador que utiliza el algoritmo TPE (Tree-structured Parzen Estimator) por defecto para Optuna, es más eficiente que la busqueda aleatoria o grid search.
from data_preprocessing import *
from model_architecture import *
from training_evaluation import *
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) # Asegura reproducibilidad en PyTorch

# --- Configuración de Rutas de Datos ---
BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "input")
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")
# Crear paths individuales
edge_path = os.path.join(BASE_INPUT_DIR, "Edge.csv")
go_path = os.path.join(BASE_INPUT_DIR, "Go.csv")
protein_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_proteins.csv")
go_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_GO.csv")

# Asegurarse de que los directorios de salida existen
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- FLujo Principal para Optuna ---
# Esta función será llamada por Optuna para cada trial, trial significa una combinación de hiperparámetros.
# Trial es el objeto que utiliza optuna, permite sugerir hiperparámetros y reportar métricas.
def objective(trial, preprocessed_data_split):
    set_seed(42) # Mantener la semilla fija para cada trial para reproducibilidad dentro del trial.

    # Determinar el dispositivo de cómputo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # desempaquetar los datos preprocesados
    data, in_channels_computed = preprocessed_data_split 
    data = data.to(device) # Mover los datos al dispositivo adecuado

    
    # --- Sugerir Hiperparámetros con Optuna ---
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    out_channels = trial.suggest_categorical("out_channels", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    predictor_hidden_channels = trial.suggest_categorical("predictor_hidden_channels", [32, 64, 128])
    dropout = trial.suggest_uniform("dropout", 0.0, 0.5) # Probabilidad de dropout para las capas GAT
    epochs_for_trial = trial.suggest_int("epochs", 50, 200, step=50) # Puedes ajustar el rango de épocas


    # --- Inicialización del Modelo ---
    model = GNNEncoder(in_channels_computed, hidden_channels, out_channels, num_heads=num_heads, dropout=dropout).to(device)
    predictor = LinkPredictor(out_channels, predictor_hidden_channels, 1).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Entrenamiento y Evaluación ---
    print(f"\n--- Trial {trial.number}: Entrenando con HPs: {trial.params} ---")
    best_val_auc = 0.0
    trial_metrics_history = [] # Para almacenar métricas de cada epoch

    for epoch in range(1, epochs_for_trial + 1):
        train_loss, train_auc, train_acc, train_precision, train_recall, train_f1 = train(model, predictor, data, optimizer, criterion)
        # Asegúrate de que tu función test devuelva todas estas métricas
        val_loss, test_loss, val_auc, test_auc, val_acc, test_acc, val_precision, test_precision, val_recall, test_recall, val_f1, test_f1 = test(model, predictor, data)

        # Guardar todas las métricas para este epoch y trial
        trial_metrics_history.append({
            'trial_id': trial.number,
            'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss.item(), 'test_loss': test_loss.item(),
            'train_auc': train_auc, 'val_auc': val_auc, 'test_auc': test_auc,
            'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
            'train_precision': train_precision, 'val_precision': val_precision, 'test_precision': test_precision,
            'train_recall': train_recall, 'val_recall': val_recall, 'test_recall': test_recall,
            'train_f1': train_f1, 'val_f1': val_f1, 'test_f1': test_f1,
            **trial.params # Añade los hiperparámetros del trial
        })
        
        # Reportar la métrica a Optuna para pruning
        trial.report(val_auc, epoch)

        # Manejar pruning
        if trial.should_prune():
            print(f"Trial {trial.number} podado en la época {epoch} debido a un rendimiento bajo.")
            raise optuna.exceptions.TrialPruned()

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs_for_trial:
            print(f'  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}')

    return best_val_auc # Optuna intentará maximizar este valor


# --- Función Principal para Ejecutar el Estudio de Optuna ---
def main_optuna():
    start_total_time = time.time()
    print("Iniciando búsqueda de hiperparámetros con Optuna...")
    print(f"Sistema Operativo: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Procesador (CPU): {platform.processor()}")
    print(f"Núcleos de CPU (físicos/lógicos): {psutil.cpu_count(logical=False)}/{os.cpu_count()}")
    
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Memoria RAM Total: {total_ram_gb:.2f} GB")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f"GPU Disponible: Sí")
        print(f"Nombre de GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        device = torch.device('cuda')
    else:
        print("GPU Disponible: No (Usando CPU)")
    print(f"Dispositivo de Cómputo: {device}")

    # --- Verificación de Archivos de Datos ---
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

    print("\n--- Fase Previa: Carga y Preprocesamiento de Datos (una sola vez) ---")

    # 1. Carga de archivos
    edges_df, go_terms_df, protein_metadata_df, go_metadata_df = load_files(
        edge_path, go_path, protein_metadata_path, go_metadata_path)
    print("Datasets cargados.")

    # 2. Mapeos y características de nodos
    protein_to_idx, _, _ = create_node_mappings(edges_df, go_terms_df, protein_metadata_df)
    # Asumiendo 'all' como filtro GO fijo para la búsqueda de hiperparámetros
    x, num_nodes_covered_by_go, num_go_terms_used, _, _, _, _ = create_node_features(
        protein_to_idx, go_terms_df, protein_metadata_df, go_metadata_df, go_ontology_filter='all'
    )
    in_channels_computed = x.shape[1] # Esto determina la dimensión de entrada del GNN
    print(f"Dimensión de las características de nodo (input para GNN): {in_channels_computed}")

    # 3. Índices y atributos de aristas
    edge_index, edge_attr, num_edges_original, num_edges_bidirectional = create_edge_index_and_attributes(edges_df, protein_to_idx)
    print("Índices y atributos de aristas creados.")

    # Crear objeto Data de PyG
    data_full = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print("Objeto Data de PyTorch Geometric creado.")

    # 4. División de enlaces
    print("Dividiendo enlaces para entrenamiento/validación/prueba (predicción de enlaces)...")
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=True, split_labels=True
    )
    train_data, val_data, test_data = transform(data_full)

    # Reestructurar el objeto data para el entrenamiento
    # Es crucial que data.edge_index y data.edge_attr sean los del TRAIN set para el GNN
    data_for_training = Data(x=train_data.x, edge_index=train_data.edge_index, edge_attr=train_data.edge_attr)
    data_for_training.train_pos_edge_index = train_data.pos_edge_label_index
    data_for_training.train_neg_edge_index = train_data.neg_edge_label_index
    data_for_training.val_pos_edge_index   = val_data.pos_edge_label_index
    data_for_training.val_neg_edge_index   = val_data.neg_edge_label_index
    data_for_training.test_pos_edge_index  = test_data.pos_edge_label_index
    data_for_training.test_neg_edge_index  = test_data.neg_edge_label_index
    
    # Empaquetar los datos preprocesados y la dimensión de entrada
    preprocessed_data_split = (data_for_training, in_channels_computed)
    print("Preprocesamiento de datos completado y listo para Optuna.")

    # Configurar el estudio de Optuna
    # `direction="maximize"` porque queremos maximizar el AUC de validación
    # `sampler` por defecto es TPESampler, que es bueno.
    # `pruner` para detener trials no prometedores. MedianPruner es un buen punto de partida.
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42) # Asegura reproducibilidad del sampler
    )

    # Ejecutar la optimización
    n_trials = 10 # Número de combinaciones de hiperparámetros a probar
    print(f"\nIniciando {n_trials} trials de búsqueda de hiperparámetros...")
    # Usamos una función lambda para pasar argumentos adicionales a `objective`
    study.optimize(lambda trial: objective(trial, preprocessed_data_split), n_trials=n_trials, show_progress_bar=True)

    print("\n--- Búsqueda de Hiperparámetros Completada ---")
    print(f"Mejor trial:")
    print(f"  Valor (AUC de validación): {study.best_value:.4f}")
    print(f"  Mejores Hiperparámetros: {study.best_params}")

    # Opcional: Guardar los resultados del estudio
    study_results_path = os.path.join(BASE_OUTPUT_DIR, "optuna_study_results.csv")
    df_results = study.trials_dataframe()
    df_results.to_csv(study_results_path, index=False)
    print(f"\nResultados completos del estudio guardados en: {study_results_path}")

    end_total_time = time.time()
    total_execution_time = end_total_time - start_total_time
    print(f"\nTiempo total de ejecución del pipeline de Optuna: {total_execution_time:.2f} segundos ({total_execution_time/60:.2f} minutos)")

if __name__ == "__main__":
    main_optuna()