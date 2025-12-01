import torch
import numpy as np
from torch_scatter import scatter_sum
from .utils import read_st_data

class STData:
    def __init__(self, args, device):
        self.name = args.dataset

        data, adata = read_st_data(args=args)

        self.num_nodes = data.x.shape[0]
        self.feature = data.x.to(device)
        self.num_features = data.x.shape[1]
        self.num_edges = int(data.edge_index.shape[1] / 2)
        self.edge_index = data.edge_index
        self.adata = adata

        # carry spatial edge weights if present
        self.weight = (
            data.edge_weight
            if hasattr(data, "edge_weight") and (data.edge_weight is not None)
            else torch.ones(self.edge_index.shape[1])
        )
        self.degrees = scatter_sum(self.weight, self.edge_index[0], dim_size=data.num_nodes).to(device)

        # Labels only if present (eval_mode == 1)
        if hasattr(data, "y") and (data.y is not None):
            self.labels = data.y.tolist()
            self.num_classes = int(len(np.unique(self.labels)))
            self.has_labels = True
        else:
            self.labels = None
            self.num_classes = 0
            self.has_labels = False

        self.adj = torch.sparse_coo_tensor(
            indices=self.edge_index,
            values=self.weight,
            size=(self.num_nodes, self.num_nodes)
        )

        self.weight = self.weight.to(device)
        self.edge_index = self.edge_index.to(device)

    def print_statistic(self):
        print(f"Dataset Name: {self.name}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Number of features: {self.num_features}")
        print(f"Number of classes: {self.num_classes} (labels loaded: {getattr(self, 'has_labels', False)})")
        print(f"Feature: {self.feature.shape}")
        print(f"edge_index: {self.edge_index.shape}")