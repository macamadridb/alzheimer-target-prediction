# main_hyperparameter_tuning.py
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
# TPE Sampler para Optuna es por defecto, por eso no es necesario importarlo explícitamente
import optuna.visualization as ov # Importar para visualizaciones

# Importar funciones de los otros archivos (asegúrate de que estén accesibles)
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
    # torch.use_deterministic_algorithms(True) # Esto puede causar problemas con algunas operaciones de GNN

# --- Configuración de Rutas de Datos ---
BASE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "input")
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "GNN-GO", "output")

# Crear paths individuales
edge_path = os.path.join(BASE_INPUT_DIR, "Edge.csv")
go_path = os.path.join(BASE_INPUT_DIR, "GO.csv")
protein_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_proteins.csv")
go_metadata_path = os.path.join(BASE_INPUT_DIR, "metadata_GO.csv")

# Los directorios existen
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- Función Objetivo para Optuna ---
def objective(trial, preprocessed_data_split):
    set_seed(42) # Mantener la semilla fija para cada trial para reproducibilidad dentro del trial.

    # Determinar el dispositivo de cómputo (se hace aquí ya que cada trial puede usar diferentes HPs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Desempaquetar los datos preprocesados
    data, in_channels_computed = preprocessed_data_split
    data = data.to(device) # Mover los datos al dispositivo dentro del trial

    # --- Sugerir Hiperparámetros con Optuna ---
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    out_channels = trial.suggest_categorical("out_channels", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    predictor_hidden_channels = trial.suggest_categorical("predictor_hidden_channels", [32, 64, 128])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5) # Renombrado a dropout_rate
    activation_function_name = trial.suggest_categorical("activation_function", ["relu", "tanh"])

    # El número de épocas también es un hiperparámetro. Ajusta el rango según sea necesario.
    epochs_for_trial = trial.suggest_int("epochs", 50, 200, step=50)

    # --- Inicialización del Modelo ---
    model = GNNEncoder(in_channels_computed, hidden_channels, out_channels, 
                       num_heads=num_heads, dropout_rate=dropout_rate, 
                       activation_fn_name=activation_function_name).to(device)
    predictor = LinkPredictor(out_channels, predictor_hidden_channels, 1).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Entrenamiento y Evaluación ---
    print(f"\n--- Trial {trial.number}: Entrenando con HPs: {trial.params} ---")
    best_val_auc = 0.0
    # Diccionario para guardar las métricas del epoch con el mejor val_auc
    best_val_auc_epoch_metrics = {} 

    for epoch in range(1, epochs_for_trial + 1):
        train_loss, train_auc, train_acc, train_precision, train_recall, train_f1 = train(model, predictor, data, optimizer, criterion)
        val_loss, test_loss, val_auc, test_auc, val_acc, test_acc, val_precision, test_precision, val_recall, test_recall, val_f1, test_f1 = test(model, predictor, data)
        
        # Reportar la métrica a Optuna para pruning
        trial.report(val_auc, epoch)

        # Actualizar el mejor AUC de validación y guardar todas las métricas de ese epoch
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_auc_epoch_metrics = {
                'epoch_at_best_val_auc': epoch,
                'train_loss': train_loss, 'val_loss': val_loss.item(), 'test_loss': test_loss.item(),
                'train_auc': train_auc, 'val_auc': val_auc, 'test_auc': test_auc,
                'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
                'train_precision': train_precision, 'val_precision': val_precision, 'test_precision': test_precision,
                'train_recall': train_recall, 'val_recall': val_recall, 'test_recall': test_recall,
                'train_f1': train_f1, 'val_f1': val_f1, 'test_f1': test_f1,
            }

        if trial.should_prune():
            print(f"Trial {trial.number} podado en la época {epoch} debido a un rendimiento bajo.")
            # Si el trial es podado, aún queremos guardar las mejores métricas obtenidas hasta el momento
            if best_val_auc_epoch_metrics: # Asegura que se haya registrado al menos un epoch
                trial.set_user_attr("best_val_auc_epoch_metrics", best_val_auc_epoch_metrics)
            raise optuna.exceptions.TrialPruned()

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs_for_trial:
            print(f'  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}')

    # Al final del trial exitoso, guardar las métricas del mejor epoch
    trial.set_user_attr("best_val_auc_epoch_metrics", best_val_auc_epoch_metrics)
    
    return best_val_auc

# --- Función Principal para Ejecutar el Estudio de Optuna ---
def main_optuna():
    start_total_time = time.time()
    print("Iniciando búsqueda de hiperparámetros con Optuna...")
    print("\n--- Información del Sistema ---")
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
    else:
        print("GPU Disponible: No (Usando CPU)")
    print(f"Dispositivo de Cómputo: {device}")

    print("\n--- Verificación de Archivos de Datos ---")
    if not os.path.isdir(BASE_INPUT_DIR):
        print(f"🚨 Error: El directorio de entrada '{BASE_INPUT_DIR}' no existe.")
        exit()

    EXPECTED_FILES = [
        "Edge.csv", "GO.csv", "metadata_proteins.csv", "metadata_GO.csv"
    ]
    for filename in EXPECTED_FILES:
        filepath = os.path.join(BASE_INPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"🚨 Error: El archivo '{filepath}' no se encontró.")
            print(f"Asegúrate de que todos los archivos CSV estén en '{BASE_INPUT_DIR}'.")
            exit()
    print("✔️ Todos los archivos de datos encontrados en el directorio de entrada.")

    # --- PREPROCESAMIENTO DE DATOS (FUERA DEL BUCLE DE OPTUNA) ---
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

    # --- Guardar descripción del espacio de búsqueda ---
    search_space_description = f"""
    Espacio de Búsqueda de Hiperparámetros (Optuna):
    -----------------------------------------------
    - hidden_channels: {['64', '128', '256']} (Categórico)
    - out_channels: {['32', '64', '128']} (Categórico)
    - num_heads: {['2', '4', '8']} (Categórico)
    - learning_rate: [1e-4, 1e-2] (Log-uniforme)
    - predictor_hidden_channels: {['32', '64', '128']} (Categórico)
    - dropout_rate: [0.0, 0.5] (Uniforme)
    - activation_function: {['relu', 'tanh']} (Categórico)
    - epochs: [50, 200] con paso de 50 (Entero)
    """
    search_space_path = os.path.join(BASE_OUTPUT_DIR, "optuna_search_space.txt")
    with open(search_space_path, "w") as f:
        f.write(search_space_description)
    print(f"\nDescripción del espacio de búsqueda guardada en: {search_space_path}")


    # Configurar el estudio de Optuna
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42) # Asegura reproducibilidad del sampler
    )

    # Ejecutar la optimización, pasando los datos preprocesados
    n_trials = 50 # Ajusta este valor según tu capacidad computacional y tiempo
    print(f"\nIniciando {n_trials} trials de búsqueda de hiperparámetros...")
    # Usamos una función lambda para pasar argumentos adicionales a `objective`
    study.optimize(lambda trial: objective(trial, preprocessed_data_split), n_trials=n_trials, show_progress_bar=True)

    print("\n--- Búsqueda de Hiperparámetros Completada ---")
    print(f"Mejor trial:")
    print(f"  Valor (AUC de validación): {study.best_value:.4f}")
    print(f"  Mejores Hiperparámetros: {study.best_params}")

    study_results_path = os.path.join(BASE_OUTPUT_DIR, "optuna_study_results.csv")
    df_results = study.trials_dataframe()
    df_results.to_csv(study_results_path, index=False)
    print(f"\nResultados completos del estudio (incluyendo HPs y estado) guardados en: {study_results_path}")

    # --- Generar visualizaciones de Optuna ---
    print("\n--- Generando visualizaciones de Optuna (requiere Plotly) ---")
    try:
        fig_history = ov.plot_optimization_history(study)
        fig_history.write_html(os.path.join(BASE_OUTPUT_DIR, "optuna_optimization_history.html"))
        print(f"Historia de optimización guardada en: {os.path.join(BASE_OUTPUT_DIR, 'optuna_optimization_history.html')}")
        
        fig_importance = ov.plot_param_importances(study)
        fig_importance.write_html(os.path.join(BASE_OUTPUT_DIR, "optuna_param_importances.html"))
        print(f"Importancia de parámetros guardada en: {os.path.join(BASE_OUTPUT_DIR, 'optuna_param_importances.html')}")
        
        fig_parallel = ov.plot_parallel_coordinate(study)
        fig_parallel.write_html(os.path.join(BASE_OUTPUT_DIR, "optuna_parallel_coordinate.html"))
        print(f"Gráfico de coordenadas paralelas guardado en: {os.path.join(BASE_OUTPUT_DIR, 'optuna_parallel_coordinate.html')}")

    except Exception as e:
        print(f"No se pudieron generar las visualizaciones de Optuna. Asegúrate de que 'plotly' esté instalado (`pip install plotly`). Error: {e}")

    # --- Guardar las 10 mejores arquitecturas basadas en métricas de TEST ---
    print("\n--- Guardando las 10 mejores arquitecturas basadas en Test AUC ---")
    completed_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Recuperar los hiperparámetros y las métricas guardadas
            hparams = trial.params
            metrics = trial.user_attrs.get("best_val_auc_epoch_metrics", {})
            
            # Combinar HPs y métricas
            trial_data = {**hparams, **metrics}
            trial_data['trial_id'] = trial.number
            trial_data['value_optimized_by_optuna'] = trial.value # El best_val_auc
            completed_trials.append(trial_data)

    if completed_trials:
        df_all_completed_trials = pd.DataFrame(completed_trials)
        # Ordenar por 'test_auc' de forma descendente y tomar los top 10
        # Puedes cambiar 'test_auc' por 'test_f1' si prefieres esa métrica
        top_10_architectures = df_all_completed_trials.sort_values(by='test_auc', ascending=False).head(10)
        
        top_10_path = os.path.join(BASE_OUTPUT_DIR, "top_10_architectures_test_metrics.csv")
        top_10_architectures.to_csv(top_10_path, index=False)
        print(f"Las 10 mejores arquitecturas (basadas en Test AUC) guardadas en: {top_10_path}")
        print("\nColumnas del archivo de las 10 mejores arquitecturas:")
        print(top_10_architectures.columns.tolist())
    else:
        print("No hay trials completados para generar el archivo de las 10 mejores arquitecturas.")


    end_total_time = time.time()
    total_execution_time = end_total_time - start_total_time
    print(f"\nTiempo total de ejecución del pipeline de Optuna: {total_execution_time:.2f} segundos ({total_execution_time/60:.2f} minutos)")

if __name__ == "__main__":
    main_optuna()