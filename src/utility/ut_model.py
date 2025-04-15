from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool, global_max_pool, SAGPooling
from torch.nn import Linear, ReLU, LayerNorm, BatchNorm1d
import torch 

"""A dic to associate pooling method to the function"""
name_to_pooling = {
    "global_mean_pool": global_mean_pool,
    "global_add_pool": global_add_pool,
    "global_sort_pool": global_sort_pool,
    "global_max_pool": global_max_pool,
    "SAGPooling": SAGPooling
}

"""A dic to associate projector to the function"""
name_to_predictor = {
    "Linear": Linear
}

"""A dic to associate activation function to function"""
name_to_activation = {
    "ReLU": ReLU
}

"""A dic to associate normalization function to function"""
name_to_norm = {
    "layernorm": LayerNorm,
    "batchnorm": BatchNorm1d
}

def drop_edges(data, drop_prob=0.2):
    num_edges = data.edge_index.size(1)
    mask = torch.rand(num_edges) > drop_prob
    edge_index = data.edge_index[:, mask]
    edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

def mask_node_features(data, mask_prob=0.1):
    mask = torch.rand(data.x.size(0)) < mask_prob
    data.x[mask] = 0
    return data

def add_noisy_node_features(data, noise_level=0.05):
    """Add a slight noise into the node feature"""
    noise = torch.randn_like(data.x) * noise_level
    data.x = data.x + noise
    return data