# Funciones de entrenamiento y evaluación

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Conjunto de entrenamiento
def train(model, predictor, data, optimizer, criterion):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index, data.edge_attr)

    # predictor para enlaces positivos 
    pos_edge_index = data.train_pos_edge_index
    pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])

    # predictor para enlaces negativos
    neg_edge_index = data.train_neg_edge_index
    neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])

    # concatenar predicciones y etiquetas reales
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    target = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)

    # calcular la pérdida y actualizar los pesos
    train_loss = criterion(pred.squeeze(), target)
    train_loss.backward()
    optimizer.step()

    preds_bin = (pred.squeeze() >= 0.5).cpu().numpy()  # Umbral por defecto de 0.5
    targets = target.cpu().numpy()

    train_auc = roc_auc_score(targets, pred.detach().cpu().numpy())
    train_acc = accuracy_score(targets, preds_bin)
    train_precision = precision_score(targets, preds_bin, zero_division=0)
    train_recall = recall_score(targets, preds_bin, zero_division=0)
    train_f1 = f1_score(targets, preds_bin, zero_division=0)


    # Métricas de entrenamiento
    
    return train_loss.item(), train_auc, train_acc, train_precision, train_recall, train_f1

@torch.no_grad() # Desactiva el calculo de gradientes
def test(model, predictor, data):
    model.eval()
    predictor.eval()

    z = model(data.x, data.edge_index, data.edge_attr)

    # VALIDACIÓN
    pos_val_edge_index = data.val_pos_edge_index
    neg_val_edge_index = data.val_neg_edge_index
    
    pos_val_pred = predictor(z[pos_val_edge_index[0]], z[pos_val_edge_index[1]])
    neg_val_pred = predictor(z[neg_val_edge_index[0]], z[neg_val_edge_index[1]])

    # Conjunto de prueba
    pos_test_edge_index = data.test_pos_edge_index
    neg_test_edge_index = data.test_neg_edge_index

    pos_test_pred = predictor(z[pos_test_edge_index[0]], z[pos_test_edge_index[1]])
    neg_test_pred = predictor(z[neg_test_edge_index[0]], z[neg_test_edge_index[1]])

    # Construccion de vectores de predicciones y etiquetas
    val_preds = torch.cat([pos_val_pred, neg_val_pred], dim=0).squeeze().cpu().numpy()
    val_targets = torch.cat([torch.ones(pos_val_pred.size(0)), torch.zeros(neg_val_pred.size(0))], dim=0).cpu().numpy()
    
    test_preds = torch.cat([pos_test_pred, neg_test_pred], dim=0).squeeze().cpu().numpy()
    test_targets = torch.cat([torch.ones(pos_test_pred.size(0)), torch.zeros(neg_test_pred.size(0))], dim=0).cpu().numpy()

    # Loss
    val_loss = F.binary_cross_entropy_with_logits(torch.tensor(val_preds), torch.tensor(val_targets))
    test_loss = F.binary_cross_entropy_with_logits(torch.tensor(test_preds), torch.tensor(test_targets))

    # Calcular ROC AUC
    val_auc = roc_auc_score(val_targets, val_preds) # para el conjunto de validación
    test_auc = roc_auc_score(test_targets, test_preds) # para el conjunto de prueba

    # Umbral para convertir las predicciones en etiquetas binarias
    val_preds_bin = (val_preds >= 0.5).astype(int) # 0.5 es el umbral por defecto
    test_preds_bin = (test_preds >= 0.5).astype(int) # 0.5 es el umbral por defecto


    # Accuracy
    val_acc = accuracy_score(val_targets, val_preds_bin)
    test_acc = accuracy_score(test_targets, test_preds_bin)

    # Precision
    val_precision = precision_score(val_targets, val_preds_bin, zero_division=0)
    test_precision = precision_score(test_targets, test_preds_bin, zero_division=0)

    # Recall
    val_recall = recall_score(val_targets, val_preds_bin, zero_division=0)
    test_recall = recall_score(test_targets, test_preds_bin, zero_division=0)

    # F1 Score
    val_f1 = f1_score(val_targets, val_preds_bin, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds_bin, zero_division=0)

    return val_loss, test_loss, val_auc, test_auc, val_acc, test_acc, val_precision, test_precision, val_recall, test_recall, val_f1, test_f1