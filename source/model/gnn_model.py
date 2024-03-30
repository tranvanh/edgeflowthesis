import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, ResGatedGraphConv

conv = ResGatedGraphConv

class GNN(torch.nn.Module):
    def __init__(self, num_features, output_channel):
        super().__init__()
        self.input_layer = conv(num_features, 32)
        self.hidden_layers = nn.ModuleList([
            conv(32, 64),
            conv(64, 128),
            conv(128, 128),
            conv(128, 128),
            conv(128, 256),
            conv(256, 256),
            conv(256, 256),
            conv(256, 512),
            conv(512, 512),
            conv(512, 512),
        ])
        self.output_layer = Linear(512,output_channel)

    def forward(self, data):
        x, edge_index = data.verts_packed(), data.edges_packed()
        edge_index_transpose = torch.transpose(edge_index, 0, 1)
        
        x = F.relu(self.input_layer(x, edge_index_transpose))
        for hl in self.hidden_layers:
            x = F.relu(hl(x, edge_index_transpose))

        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        # x = sphere_mesh.offset_verts(x)
        return x

