import os
import os.path as osp
import GPUtil
import argparse
import numpy as np
import torch
from pathlib import Path

from sklearn.model_selection import ShuffleSplit

from typing import Optional

from torch import nn
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_scipy_sparse_matrix, homophily
from torch_geometric.data import Data
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
                 'karate', 'banana', 'circle'])
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
    parser.add_argument(
        '--train_masked_update', type=to_bool, default=False,
        help="Exclude train-train edge updates")
    parser.add_argument(
        '--num_sampled_nodes_per_hop', type=int, default=10,
        help="Number of sampled nodes per hop for GraphSAGE")
    parser.add_argument(
        '--optimizer', type=str, default='adam',
        choices=['adam', 'sam'],
        help="Optimizer for model training")
    parser.add_argument(
        '--grad_norm', type=to_bool, default=False,
        help="Use gradient clipping")
    parser.add_argument(
        '--sign_grad', type=to_bool, default=False,
        help="Use sign gradient")
    parser.add_argument(
        '--momentum_adj', type=float, default=0.,
        help="SGD momentum for adjacency matrix")
    parser.add_argument(
        '--early_stop', type=to_bool, default=False,
        help="Use early stopping")
    parser.add_argument(
        '--overwrite_config', type=to_bool, default=False,
        help="Overwrite config file")
    parser.add_argument(
        '--plot_loss', type=to_bool, default=False,
        help="Plot loss")
    parser.add_argument(
        '--num_layers', type=int, default=2,
        help="Number of layers for GCN model")
    return parser

def gen_edge_index(labels, num_edges, homophily, seed=0):
    num_nodes = labels.shape[0]
    edge_index = torch.zeros(num_edges, 2, dtype=torch.long)

    np.random.seed(seed)
    for i in range(num_edges):
        while True:
            src = np.random.randint(num_nodes)
            dst = np.random.randint(num_nodes)
            rand = np.random.rand()
            if src != dst:
                if (labels[src] == labels[dst] and rand < homophily) or \
                    (labels[src] != labels[dst] and rand >= homophily):
                    edge_index[i] = torch.tensor([src, dst])
                    break
    intra_edges = (labels[edge_index[:, 0]] == labels[edge_index[:, 1]]).sum()
    inter_edges = num_edges - intra_edges
    print(f"Generated {intra_edges} intra-edges and {inter_edges} inter-edges")
    return edge_index.t().contiguous()

def gen_circle_edge_index(x, y, num_edges, homophily, seed=0):
    np.random.seed(seed)
    edge_index = []
    nodes = np.random.randint(0, x.shape[0], num_edges)
    for u in nodes:
        x_u = x[u]
        distances = np.sum((x - x_u) ** 2, axis=1)
        sorted_indices = np.argsort(distances)
        for v in sorted_indices[:10]:
            if (homophily and y[u] == y[v]) or (not homophily and y[u] != y[v]):
                edge_index.append([u, v])
                break
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


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
    elif dataset == 'banana':
        import pandas as pd
        df = pd.read_csv("data/banana.csv")
        df['Class'].replace({-1: 0}, inplace=True)
        x = torch.tensor(df[['At1', 'At2']].values, dtype=torch.float)
        y = torch.tensor(df['Class'].values, dtype=torch.long)
        # edge_index = torch.cat(
        #     [torch.arange(x.size(0)).unsqueeze(0),
        #      torch.arange(x.size(0)).unsqueeze(0)], dim=0)
        # df_x = pd.read_csv("data/snelson/train_inputs.csv", header=None)
        # df_y = pd.read_csv("data/snelson/train_outputs.csv", header=None)
        # x = torch.tensor(df_x.values, dtype=torch.float)
        # y = torch.tensor(df_y.values, dtype=torch.float)
        data = Data(x=x, y=y, edge_index=edge_index)
        # import ipdb, ipdb; ipdb.set_trace()
        # random graph
        # data.edge_index = torch.randint(0, data.x.size(0), (2, 500))
    elif dataset == 'circle':
        # from sklearn.datasets import make_circles
        # X, y = make_circles(n_samples=100, noise=0.05, factor=0.7,
        #                     random_state=42)
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        # edge_index = torch.cat(
        #     [torch.arange(X.shape[0]).unsqueeze(0),
        #      torch.arange(X.shape[0]).unsqueeze(0)], dim=0)
        edge_index = gen_edge_index(y, 70, 0.2)
        # edge_index = gen_circle_edge_index(X, y, 200, True)
        # np.random.seed(42)
        # # edge_index = np.random.randint(0, X.shape[0], (2, 5000))
        # # edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # edge_index = np.random.randint(0, X.shape[0], (2, 10))
        # nodes_0 = np.where(y == 0)[0]
        # nodes_0 = nodes_0[np.random.randint(0, len(nodes_0), 300)]
        # nodes_1 = np.where(y == 1)[0]
        # nodes_1 = nodes_1[np.random.randint(0, len(nodes_1), 300)]
        
        # # Heterophilic graph
        # # syn_edges = np.stack([nodes_0, nodes_1], axis=0)
        # # # Homophilic graph
        # syn_edges = np.concatenate([
        #     np.stack([nodes_0[:len(nodes_0) // 2], nodes_0[len(nodes_0) // 2:]], axis=0),
        #     np.stack([nodes_1[:len(nodes_1) // 2], nodes_1[len(nodes_1) // 2:]], axis=0)], axis=1)
        # import ipdb; ipdb.set_trace()

        # edge_index = np.concatenate([
        #     edge_index, syn_edges], axis=1)
        # edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=torch.tensor(X, dtype=torch.float),
                    y=torch.tensor(y, dtype=torch.long),
                    edge_index=edge_index)
        with open('/home/anita/work/laplace-gnn-results/circle/data_hetero.pkl', 'wb') as f:
            torch.save(data, f)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
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

def unused_gpu():
    for gpu in GPUtil.getGPUs():
        if gpu.load == 0:
            return gpu.id
    return None