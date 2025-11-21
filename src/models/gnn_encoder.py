"""Graph encoder for market state."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class MarketGraphEncoder(nn.Module):
    """
    A simple GraphSAGE encoder producing a pooled embedding per graph.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(layers)
        self.output_dim = hidden_dim

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        pooled = global_mean_pool(x, batch)
        return pooled.squeeze(0)

