import os
import argparse
import torch
from pathlib import Path

from typing import Optional

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import to_scipy_sparse_matrix, homophily
from torch_geometric.datasets import KarateClub, Planetoid, \
    Actor, WikipediaNetwork, WebKB


BASE_OUT_DIR = os.path.join(Path.home(), 'work/laplace-gnn-results')


# def weighted_homophily(adj: torch.tensor, y: torch.tensor):
#     """
#     Computes the homophily of a graph given the edge index, edge weight and
#     node labels.
#     """
#     sparse_adj = adj.to_sparse()
#     homophily = 0
#     for i, j, d in zip(sparse_adj.indices()[0], sparse_adj.indices()[1],
#                        sparse_adj.values()):
#         homophily += d * (y[i] == y[j]).float()
#     return homophily / sparse_adj.values().sum()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str,
        choices=['cora', 'citeseer', 'pubmed',
                 'chameleon', 'squirrel',
                 'actor', 'texas', 'wisconsin', 'cornell',])
    parser.add_argument(
        '--model_type', type=str,
        choices=['stegcn', 'clipgcn', 'gcn', 'gat']
    )
    parser.add_argument(
        '--base_out_dir', type=str, default=BASE_OUT_DIR)
    parser.add_argument(
        '--subset_of_weights', type=str, default='all',
        choices=['all', 'last'])
    parser.add_argument(
        '--hessian_structure', type=str, default='kron',
        choices=['full', 'diag', 'kron'])
    parser.add_argument(
        '--hidden_channels', type=int, default=None,
        help="Number of hidden channels in the GCN model")
    parser.add_argument(
        '--ste_thresh', type=float, default=None)
    parser.add_argument(
        '--knng_k', type=int, default=3,
        help="Number of nearest neighbors for knn graph")
    parser.add_argument(
        '--lr', type=float, default=None,
        help="Learning rate for model weights")
    parser.add_argument(
        '--lr_adj', type=float, default=None,
        help="Learning rate for adjacency matrix")
    parser.add_argument(
        '--weight_decay', type=float, default=None,
        help="Weight decay for model weights")
    parser.add_argument(
        '--n_epochs', type=int, default=200,
        help="Number of epochs for training")
    parser.add_argument(
        '--n_hypersteps', type=int, default=10,
        help="Number of hypersteps for adjacency matrix")
    parser.add_argument(
        '--n_epochs_burnin', type=int, default=100,
        help="Number of epochs before starting hypersteps")
    parser.add_argument(
        '--marglik_frequency', type=int, default=20,
        help="Frequency of computing marginal likelihood")
    parser.add_argument(
        '--init_graph', type=str, default='original',
        choices=['original', 'knng', 'none'],
        help="Initial graph structure")
    parser.add_argument(
        '--dropout_p', type=float, default=None,
        help="Dropout probability")
    parser.add_argument(
        '--n_repeats', type=int, default=1,
        help="Number of repeats for training")
    parser.add_argument(
        '--stop_criterion', type=str, default='valloss',
        choices=['valloss', 'marglik'],
        help="Stopping criterion for training")
    return parser

def load_data(dataset):
    root = os.path.join(Path.home(), 'data')
    if dataset in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(root=root, name=dataset.capitalize())[0]
    elif dataset == 'actor':
        data = Actor(root=root)[0]
    elif dataset == 'chameleon':
        data = WikipediaNetwork(root=root, name='chameleon')[0]
    elif dataset in ['texas', 'wisconsin', 'cornell']:
        data = WebKB(root=root, name=dataset.capitalize())[0]
    elif dataset == 'karate':
        data = KarateClub()[0]
        torch.manual_seed(0)
        data.train_mask = torch.zeros((data.y.size(0), 10), dtype=torch.bool)
        data.test_mask = torch.zeros((data.y.size(0), 10), dtype=torch.bool)
        for i in range(10):
            rand_idx = torch.randperm(data.x.size(0))
            data.train_mask[rand_idx[:20], i] = True
            data.test_mask[rand_idx[20:], i] = True
        # rand_idx = torch.randperm(data.x.size(0))
        # data.train_mask = torch.zeros_like(data.y, dtype=torch.bool)
        # data.test_mask = torch.zeros_like(data.y, dtype=torch.bool)    
        # data.train_mask[rand_idx[:20]] = True
        # data.test_mask[rand_idx[20:]] = True
    if data.train_mask.ndim == 1:
        data.train_mask = data.train_mask.unsqueeze(1)
        data.val_mask = data.val_mask.unsqueeze(1)
        data.test_mask = data.test_mask.unsqueeze(1)
    return data


def edge_index_to_adj(edge_index, num_nodes=None):
    return torch.tensor(
        to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).toarray(),
        dtype=torch.int64)


def adj_to_edge_index(adj):
    _adj = adj.clone()
    _adj.fill_diagonal_(0)
    return _adj.nonzero().t().contiguous()


def knn_graph(x, k=8, as_adj_mask=False, cosine=False):
    from torch_geometric.nn import knn_graph
    edge_index = knn_graph(x, k, batch=None, loop=False, cosine=cosine)
    if as_adj_mask:
        num_nodes = x.size(0)
        adj = edge_index_to_adj(edge_index, num_nodes).T
        # import ipdb; ipdb.set_trace()
        adj_mask = (adj + torch.eye(num_nodes)).bool()
        return adj_mask
    return edge_index


# def receptive_field(adj_matrix, num_hops):
#     # adj = adj_matrix.to_sparse()
#     neigh = adj_matrix.clone()
#     for _ in range(num_hops - 1):
#         # import ipdb; ipdb.set_trace()
#         neigh += neigh @ adj_matrix
#         # neigh = neigh + adj_matrix
#     return neigh.bool()


# def batch_to_full_adj(batch_adj, batch_nodes, total_num_nodes):
#     full_adj = torch.zeros(total_num_nodes, total_num_nodes)
#     full_adj[batch_nodes, :][:, batch_nodes] = batch_adj
#     return full_adj
def normalize_adj(adj):
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = rowsum.pow(-0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return (adj @ d_mat_inv_sqrt).T @ d_mat_inv_sqrt


def preprocess_adj(adj):
    device = adj.device
    adj_id = torch.eye(adj.shape[0]).to(device)
    adj_normalized = normalize_adj(
        adj + adj_id)
    return adj_normalized
