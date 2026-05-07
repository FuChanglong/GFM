import os.path as osp

import numpy as np
import torch
from torch_geometric.utils import to_undirected


def get_raw_text_pubmed(use_text=False, seed=0):
    if osp.exists(f"./processed_data/pubmed.pt"):
        # 加载数据
        data = torch.load(f"./processed_data/pubmed.pt", map_location='cpu', weights_only=False)
        data.num_nodes = data.y.shape[0]
        edge_index = to_undirected(data.edge_index)

        data.edge_index = edge_index
        data.num_nodes = data.y.shape[0]

        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

        data.train_mask = torch.tensor(
            [x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor(
            [x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor(
            [x in data.test_id for x in range(data.num_nodes)])
        return data, []
    else:
        raise NotImplementedError('No existing pubmed dataset!')
