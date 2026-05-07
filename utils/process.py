import json
import os
import random

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm


def process_subgraph_edges(subgraph, sim_threshold=0.6, drop_ratio=0.7):
    edge_index = subgraph.edge_index
    node_feats = subgraph.x
    device = node_feats.device
    N = node_feats.shape[0]
    edge_index = edge_index.flatten().reshape(2, -1).to(device).long()
    num_total_edges = edge_index.size(1)
    if num_total_edges == 0:
        return subgraph.clone()

    src_idxs = edge_index[0, :]
    dst_idxs = edge_index[1, :]
    src_feats = node_feats[src_idxs, :]
    dst_feats = node_feats[dst_idxs, :]

    src_feats_norm = F.normalize(src_feats, p=2, dim=1)
    dst_feats_norm = F.normalize(dst_feats, p=2, dim=1)
    similarity = (src_feats_norm * dst_feats_norm).sum(dim=1)
    similarity = (similarity + 1) / 2

    high_sim_mask = similarity >= sim_threshold
    low_sim_mask = ~high_sim_mask
    high_sim_edges = edge_index[:, high_sim_mask]
    low_sim_edges = edge_index[:, low_sim_mask]

    high_sim_edges = high_sim_edges.flatten().reshape(2, -1).long()
    low_sim_edges = low_sim_edges.flatten().reshape(2, -1).long()

    num_low_sim = low_sim_edges.size(1)
    num_keep = int(num_low_sim * (1 - drop_ratio))
    num_keep = max(0, min(num_keep, num_low_sim))

    if num_keep > 0:
        keep_idx = torch.randperm(num_low_sim, device=device)[:num_keep]
        kept_low_sim_edges = low_sim_edges[:, keep_idx]
    else:
        kept_low_sim_edges = torch.empty((2, 0), dtype=torch.long, device=device)

    tensors_to_cat = [high_sim_edges, kept_low_sim_edges]
    tensors_to_cat = [t.flatten().reshape(2, -1).long().to(device) for t in tensors_to_cat]
    new_edge_index = torch.cat(tensors_to_cat, dim=1)

    new_subgraph = subgraph.clone()
    new_subgraph.edge_index = new_edge_index.flatten().reshape(2, -1).long()

    return new_subgraph


def parse_source_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    mapping_id = {'ogbn-arxiv': 0, 'arxiv_2023': 1, 'pubmed': 2, 'ogbn-products': 3, 'reddit': 4}
    with open(f'./summary/summary-{name}.json', 'r') as fcc_file:  # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data
    collected_graph_data = []
    print("process", name)
    for id, jd in enumerate(tqdm(json_data)):
        # assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j: i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']],
                     summary=summary, graph_id=mapping_id[name])
        graph = transform(graph)  # add PE
        collected_graph_data.append(graph)
    return collected_graph_data


def parse_target_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    with open(f'./target_data/{name}.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        if edges.shape[1] == 0:
            edges = torch.tensor([[id], [id]])

        node_idx = torch.unique(edges)
        node_idx_map = {j: i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']])
        graph = transform(graph)
        collected_graph_data.append(graph)

    if True:
        save_dir = './saved_target_data'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{name}_graph_data.pt')
        torch.save(collected_graph_data, save_path)
    return collected_graph_data


def parse_target_data_lp(name, data, seed):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    with open(f'./target_data/{name}_lp_seed{seed}.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        if edges.shape[1] == 0:
            edges = torch.tensor([[id], [id]])

        node_idx = torch.unique(edges)
        node_idx_map = {j: i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']])
        graph = transform(graph)
        collected_graph_data.append(graph)
    return collected_graph_data


def load_parsed_source_data(name, load_path=None):
    if load_path is None:
        load_path = f'./processed_data/{name}-tc.pt'
    collected_graph_data = torch.load(load_path, weights_only=False)
    return collected_graph_data


def load_parsed_target_data(name, load_path=None):
    if load_path is None:
        load_path = f'./saved_target_data/{name}_graph_data.pt'
    collected_graph_data = torch.load(load_path, weights_only=False)
    return collected_graph_data


def split_dataloader(data, graphs, batch_size, seed=0, name='cora'):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                              shuffle=False)  # use DataListLoader for DP rather than DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    return train_loader, val_loader, test_loader


def split_dataloader_2(data, graphs, batch_size, seed=0, name='cora'):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]

    train_loader = DataListLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)  # use DataListLoader for DP rather than DataLoader
    val_loader = DataListLoader(val_dataset, batch_size=batch_size)
    test_loader = DataListLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
