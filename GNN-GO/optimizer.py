import torch
import torch.nn as nn
import torch.nn.functional as F
# Importar tus modelos desde 'model_architecture.py'
from model_architecture import GNNEncoder, LinkPredictor
# Importar tus funciones de entrenamiento y evaluación desde 'training_evaluation.py'
from training_evaluation import train, test

import optuna
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit # Para dividir el dataset y generar enlaces negativos


# --- La Función Objetivo para Optuna ---
def objective(trial):
    """
    Función que Optuna llama para cada "trial" (combinación de hiperparámetros).
    Entrena el modelo con los hiperparámetros sugeridos y devuelve la métrica a minimizar.
    """
    # --- Espacio de búsqueda de hiperparámetros ---
    # Hiperparámetros de GNNEncoder
    hidden_channels_encoder = trial.suggest_categorical('hidden_channels_encoder', [64, 128, 256])
    out_channels_encoder = trial.suggest_categorical('out_channels_encoder', [64, 128]) # Embeddings de salida de la GNN
    num_heads = trial.suggest_int('num_heads', 1, 4) # Número de cabezas de atención para GATConv
    
    # Hiperparámetros de LinkPredictor
    hidden_channels_predictor = trial.suggest_categorical('hidden_channels_predictor', [64, 128, 256])
    
    # Hiperparámetros de entrenamiento
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    epochs = trial.suggest_categorical('epochs', [50, 100, 150]) # Número de épocas fijas para el trial

    # --- Configuración de Dispositivo ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Carga y Preparación de Datos (EJEMPLO - REEMPLAZA CON TUS DATOS REALES) ---
    # Debes cargar aquí tu objeto 'Data' principal de PyTorch Geometric.
    # Por ejemplo, si tienes un grafo 'data_original' que contiene x, edge_index, edge_attr.
    
    # Placeholder de datos:
    num_nodes = 100
    in_channels = 16 # Dimensión de las características de los nodos
    num_edges = 500
    x = torch.randn(num_nodes, in_channels)
    row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    # Convertir a no dirigido para garantizar simetría si es el caso de tu GNN
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_attr = torch.randn(edge_index.size(1), 1)
    
    data_original = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_original = data_original.to(device)

    # Dividir el grafo en conjuntos de entrenamiento, validación y prueba
    # y generar automáticamente enlaces negativos para cada conjunto.
    # Esto creará atributos como 'train_pos_edge_index', 'val_pos_edge_index', etc.
    transform = RandomLinkSplit(
        num_val=0.1,  # 10% de enlaces para validación
        num_test=0.1, # 10% de enlaces para prueba
        is_undirected=True, # Importante si tus bordes deben ser no dirigidos
        # 'add_negative_train_samples=False' porque la función 'train' de training_evaluation.py
        # ya espera 'train_neg_edge_index' del objeto 'data'
        add_negative_train_samples=False 
    )
    # Aplica la transformación para obtener los objetos Data con los índices de enlaces divididos
    train_data, val_data, test_data = transform(data_original)

    # --- Inicialización del Modelo y Optimizador ---
    model = GNNEncoder(in_channels=in_channels,
                       hidden_channels=hidden_channels_encoder,
                       out_channels=out_channels_encoder,
                       num_heads=num_heads).to(device)
    
    predictor = LinkPredictor(in_channels=out_channels_encoder, # Debe coincidir con out_channels_encoder
                              hidden_channels=hidden_channels_predictor,
                              out_channels=1).to(device) # out_channels=1 para logits de clasificación binaria

    # Tu función 'train' de training_evaluation.py espera un solo optimizador.
    # Combinamos los parámetros de ambos modelos en un solo optimizador.
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), 
                                 lr=learning_rate, 
                                 weight_decay=weight_decay)
    
    # Usamos BCEWithLogitsLoss para clasificación binaria de enlaces
    criterion = nn.BCEWithLogitsLoss() 

    # --- Ciclo de Entrenamiento ---
    best_val_auc = -1.0 # Queremos maximizar AUC, así que inicializamos con un valor bajo
    for epoch in range(epochs):
        # La función 'train' de training_evaluation.py ya espera un objeto 'data'
        # que contiene train_pos_edge_index y train_neg_edge_index
        train_loss, train_auc, _, _, _, _ = train(model, predictor, train_data, optimizer, criterion)
        
        # La función 'test' de training_evaluation.py ya calcula val_auc
        # Se le pasa el val_data (que contiene val_pos_edge_index y val_neg_edge_index)
        val_loss, test_loss, val_auc, test_auc, _, _, _, _, _, _, _, _ = test(model, predictor, val_data) # Ojo: val_data para test!
        
        # Reportar la métrica (val_auc) a Optuna para permitir la poda temprana
        trial.report(val_auc, epoch) # Optuna intenta maximizar este valor por defecto si direction="maximize"

        # Si el rendimiento no es prometedor, Optuna puede podar (detener) este trial temprano
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Opcional: Aquí podrías guardar el estado del modelo si necesitas el "mejor" modelo de cada trial

    return best_val_auc # Optuna intentará maximizar este valor
    # Si quisieras minimizar, retornarías (1 - best_val_auc) y direction="minimize"

# --- Ejecución de la Optimización con Optuna ---
if __name__ == "__main__":
    # --- Configuración del Estudio Optuna ---
    # 'direction="maximize"' porque queremos maximizar el AUC de validación
    # Puedes usar 'storage' para persistir los resultados en una base de datos SQLite.
    study = optuna.create_study(direction="maximize", study_name="gnn_link_prediction_tuning", storage="sqlite:///gnn_hp_tuning.db")
    
    # Si no quieres usar una base de datos, puedes usar un estudio en memoria (los resultados se perderán):
    # study = optuna.create_study(direction="maximize")
    
    # --- Ejecutar la Optimización ---
    n_trials = 50 # Número de combinaciones de hiperparámetros a probar
    print(f"Iniciando la optimización con {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # --- Resultados Finales ---
    print("\n--- Optimización Finalizada ---")
    print(f"Mejor trial encontrado:")
    print(f"  Valor de la métrica (AUC de validación): {study.best_value:.4f}")
    print(f"  Mejores Hiperparámetros: {study.best_params}")

    # Puedes ver un resumen de todos los trials
    # print("\n--- Resumen de todos los trials ---")
    # for i, trial in enumerate(study.trials):
    #     print(f"Trial {i}: Valor = {trial.value:.4f}, Hiperparámetros = {trial.params}")

    # Optuna también ofrece funcionalidades para visualizar los resultados
    # (requiere 'plotly' y 'matplotlib'). Por ejemplo:
    # import optuna.visualization as ov
    # fig_hist = ov.plot_optimization_history(study)
    # fig_hist.show()
    # fig_param_imp = ov.plot_param_importances(study)
    # fig_param_imp.show()