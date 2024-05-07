import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, ResGatedGraphConv, BatchNorm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_channel):
        super().__init__()
        conv = ResGatedGraphConv
        self.input_layer = conv(num_features, hidden_channels)
        self.hidden_layers = nn.ModuleList([
            conv(hidden_channels, hidden_channels),
            conv(hidden_channels, hidden_channels),
            conv(hidden_channels, hidden_channels),
            conv(hidden_channels, hidden_channels),
            
        ])
        self.norm = BatchNorm(hidden_channels)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = Linear(hidden_channels,output_channel)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(device)
        x = F.leaky_relu(self.input_layer(x, edge_index))
        
        for hl in self.hidden_layers:
            x = F.leaky_relu(hl(x, edge_index))
        self.norm
        self.dropout
        
        for hl in self.hidden_layers:
            x = F.leaky_relu(hl(x, edge_index))
        self.norm
        self.dropout

        for hl in self.hidden_layers:
            x = F.leaky_relu(hl(x, edge_index))
            
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x
