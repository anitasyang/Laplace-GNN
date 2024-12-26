import os
import os.path as osp
import argparse
import numpy as np
import torch
from pathlib import Path

from sklearn.model_selection import ShuffleSplit

from typing import Optional

from torch import nn
from torch_geometric.nn import knn_graph
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
    def to_bool(value):
        return value.lower() in ['true', '1', 'yes', 'y']
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str,
        choices=['cora', 'citeseer', 'pubmed',
                 'chameleon', 'squirrel',
                 'actor', 'texas', 'wisconsin', 'cornell',
                 'karate'])
    parser.add_argument(
        '--model_type', type=str,
        choices=['stegcn', 'clipgcn', 'gcn', 'lorastegcn',
                 'gat', 'stegraphsage', 'graphsage'])
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
        # choices=['original', 'knng', 'none'],
        help="Initial graph structure")
    parser.add_argument(
        '--dropout_p', type=float, default=None,
        help="Dropout probability")
    parser.add_argument(
        '--n_repeats', type=int, default=1,
        help="Number of repeats for training")
    parser.add_argument(
        '--stop_criterion', type=str, default=None,
        choices=['valloss', 'marglik'],
        help="Stopping criterion for training")
    parser.add_argument(
        '--lora_r', type=int, default=None,
        help="Number of dim for LoRA")
    parser.add_argument(
        '--lora_alpha', type=int, default=16,
        help="Scaling factor for LoRA")
    parser.add_argument(
        '--gpu_num', type=int, default=0,
        help="GPU number")
    parser.add_argument(
        '--n_data_rand_splits', type=int, default=10,
        help="Random splits for train-val-test")
    parser.add_argument(
        '--n_hyper_stop', type=int, default=None,
        help='Epoch to stop hyperparameter optimization')
    parser.add_argument(
        '--norm', type=str, default=None,
        choices=['none', 'batch', 'layer'],
        help='Normalization for the model')
    parser.add_argument(
        '--res', type=to_bool, default=None,
        help='Use residual connection')
    parser.add_argument(
        '--weight_decay_adj', type=float, default=None,
        help="Weight decay for adjacency matrix")
    parser.add_argument(
        '--heads', type=int, default=1,
        help="Number of attention heads")
    parser.add_argument(
        '--symmetric', type=to_bool, default=False,
        help="Use symmetrize adjacency matrix")
    return parser

def load_data(dataset, n_rand_splits=1):
    root = os.path.join(Path.home(), 'data')
    if dataset in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(root=root, name=dataset.capitalize())[0]
    elif dataset == 'actor':
        data = Actor(root=root)[0]
    elif dataset in ['chameleon', 'squirrel']:
        data = WikipediaNetwork(root=root, name=dataset)[0]
    elif dataset in ['texas', 'wisconsin', 'cornell']:
        data = WebKB(root=root, name=dataset.capitalize())[0]
    elif dataset == 'karate':
        data = KarateClub()[0]
    
    # if n_rand_splits == 1 and hasattr(data, 'val_mask'):
    #     if data.train_mask.ndim > 1:
    #         data.train_mask = data.train_mask[:, 0]
    #         data.val_mask = data.val_mask[:, 0]
    #         data.test_mask = data.test_mask[:, 0]
    #     data.train_indices = torch.nonzero(
    #         data.train_mask, as_tuple=True)[0].unsqueeze(1)
    #     data.val_indices = torch.nonzero(
    #         data.val_mask, as_tuple=True)[0].unsqueeze(1)
    #     data.test_indices = torch.nonzero(
    #         data.test_mask, as_tuple=True)[0].unsqueeze(1)
    # else:
    # 60-20-20 split
    train_percentage = 0.6
    val_percentage = 0.2
    train_indices, val_indices, test_indices = [], [], []
    train_masks, val_masks, test_masks = [], [], []
    rs = ShuffleSplit(n_splits=n_rand_splits,
                        train_size=train_percentage + val_percentage,
                        random_state=0)
    for i , (train_and_val_index, test_index) in enumerate(rs.split(data.x)):
        train_index, val_index = next(ShuffleSplit(
            n_splits=1, train_size=train_percentage, random_state=0).split(
            data.x[train_and_val_index]))

        train_index = train_and_val_index[train_index]
        val_index = train_and_val_index[val_index]
        train_mask = torch.zeros_like(data.y, dtype=torch.bool)
        
        val_mask = torch.zeros_like(data.y, dtype=torch.bool)
        test_mask = torch.zeros_like(data.y, dtype=torch.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        
        train_indices.append(train_index)
        val_indices.append(val_index)
        test_indices.append(test_index)
        
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    data.train_indices = torch.tensor(np.array(train_indices)).t()
    data.val_indices = torch.tensor(np.array(val_indices)).t()
    data.test_indices = torch.tensor(np.array(test_indices)).t()
    
    data.train_mask = torch.stack(train_masks, dim=1)
    data.val_mask = torch.stack(val_masks, dim=1)
    data.test_mask = torch.stack(test_masks, dim=1)
    return data


def edge_index_to_adj(edge_index, num_nodes=None, edge_weight=None):
    return torch.tensor(
        to_scipy_sparse_matrix(
            edge_index, edge_attr=edge_weight,
            num_nodes=num_nodes).toarray(),
        dtype=torch.int64)


def adj_to_edge_index(adj):
    _adj = adj.clone()
    _adj.fill_diagonal_(0)
    return _adj.nonzero().t().contiguous()



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

def get_knn_graph(X, k=3, return_edge_index=False):
    """
    Get k-nearest neighbor graph.
    """
    edge_index = knn_graph(
        X, k=k, batch=None, loop=False, cosine=False)
    adj = edge_index_to_adj(
        edge_index, num_nodes=X.size(0)).float()
    # symmetrize
    adj = (adj + adj.t()).bool().float()
    adj.fill_diagonal_(1)
    if return_edge_index:
        edge_index = adj_to_edge_index(adj)
        return adj, edge_index
    return adj

def normalize_adj(adj):
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = rowsum.pow(-0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return (adj @ d_mat_inv_sqrt).T @ d_mat_inv_sqrt


def preprocess_adj(adj):
    device = adj.device
    adj_id = torch.eye(adj.shape[0]).to(device)
    adj_normalized = normalize_adj(
        adj + adj_id)
    return adj_normalized


def get_learned_graphs_dir(
        out_dir, init_graph, model_type, hessian_structure,
        subset_of_weights):
    return osp.join(
            out_dir,
            '_'.join([
                init_graph,
                model_type,
                hessian_structure,
                subset_of_weights, 'strucs']))


def fully_connected_labels(labels):
    """
    Create fully connected components for each label.
    """
    num_nodes = labels.size(0)
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            adj[i, j] = labels[i] == labels[j]
    return adj.bool().float()