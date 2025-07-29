# model_architecture.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch 

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, add_self_loops=True, dropout_rate=0.260034010588905, activation_fn_name="relu"):
        super(GNNEncoder, self).__init__()
        
        # GATConv aplica dropout tanto en los coeficientes de atención como en las transformaciones lineales
        # Pasamos dropout_rate a ambas capas GATConv
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                             add_self_loops=add_self_loops, edge_dim=1, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, 
                             add_self_loops=add_self_loops, edge_dim=1, concat=False, dropout=dropout_rate)
        
        # Dropout adicional entre capas GNN (opcional pero común)
        self.dropout_layer = nn.Dropout(dropout_rate) 

        # Seleccionar la función de activación basada en el nombre
        if activation_fn_name == "relu":
            self.activation_fn = F.relu
        elif activation_fn_name == "tanh":
            self.activation_fn = F.tanh
        else:
            raise ValueError(f"Función de activación '{activation_fn_name}' no soportada.")

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.activation_fn(x)  # Aplicar la función de activación seleccionada
        x = self.dropout_layer(x)  # Aplicar dropout
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels) # x_i y x_j concatenados
        self.lin2 = nn.Linear(hidden_channels, out_channels) 

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1) # Concatenar embeddings de los dos nodos
        x = self.lin1(x)
        x = F.relu(x) # Activación ReLU
        x = self.lin2(x)
        return x