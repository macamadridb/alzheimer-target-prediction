# Definición de mi modelo GNN y predictor de enlaces

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch 

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, add_self_loops=True):
        super(GNNEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                             add_self_loops=add_self_loops, edge_dim=1, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, 
                             add_self_loops=add_self_loops, edge_dim=1, dropout=0.5)
        
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)  
        # funcion de activacion
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
       # x = F.elu(x) # Aplicar activación ELU opcional
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels) 
        self.lin2 = nn.Linear(hidden_channels, out_channels) 

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
    
    # Se usa para regresión, ocupar función de perdida nn.MSELoss