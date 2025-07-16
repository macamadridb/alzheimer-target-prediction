# Funciones de entrenamiento y evaluación

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def train(model, predictor, data, optimizer, criterion):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index, data.edge_attr)

    pos_edge_index = data.train_pos_edge_index
    pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])

    neg_edge_index = data.train_neg_edge_index
    neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])

    pred = torch.cat([pos_pred, neg_pred], dim=0)
    target = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)

    loss = criterion(pred.squeeze(), target)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, predictor, data):
    model.eval()
    predictor.eval()

    z = model(data.x, data.edge_index, data.edge_attr)

    pos_val_edge_index = data.val_pos_edge_index
    neg_val_edge_index = data.val_neg_edge_index
    
    pos_val_pred = predictor(z[pos_val_edge_index[0]], z[pos_val_edge_index[1]])
    neg_val_pred = predictor(z[neg_val_edge_index[0]], z[neg_val_edge_index[1]])

    pos_test_edge_index = data.test_pos_edge_index
    neg_test_edge_index = data.test_neg_edge_index

    pos_test_pred = predictor(z[pos_test_edge_index[0]], z[pos_test_edge_index[1]])
    neg_test_pred = predictor(z[neg_test_edge_index[0]], z[neg_test_edge_index[1]])

    val_preds = torch.cat([pos_val_pred, neg_val_pred], dim=0).squeeze().cpu().numpy()
    val_targets = torch.cat([torch.ones(pos_val_pred.size(0)), torch.zeros(neg_val_pred.size(0))], dim=0).cpu().numpy()
    
    test_preds = torch.cat([pos_test_pred, neg_test_pred], dim=0).squeeze().cpu().numpy()
    test_targets = torch.cat([torch.ones(pos_test_pred.size(0)), torch.zeros(neg_test_pred.size(0))], dim=0).cpu().numpy()

    val_auc = roc_auc_score(val_targets, val_preds)
    test_auc = roc_auc_score(test_targets, test_preds)

    return val_auc, test_auc