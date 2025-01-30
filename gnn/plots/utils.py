import glob
import numpy as np
import torch
from collections import defaultdict

from gnn.utils import get_learned_graphs_dir, edge_index_to_adj
from gnn.models.utils import power_adj

BASE_OUT_DIR = "/home/anita/work/laplace-gnn-results"


def get_learned_graphs(
        out_dir,
        init_graph,
        model_type,
        hessian_structure='kron',
        subset_of_weights='all',
        epoch_num=None,):
    dir = get_learned_graphs_dir(
        out_dir, init_graph, model_type,
        hessian_structure, subset_of_weights)
    if epoch_num is not None:
        fns = [f"{dir}/epoch_{epoch_num}.pt"]
    else:
        fns = sorted(glob.glob(f"{dir}/*.pt"))
    for fn in fns:
        if 'epoch' not in fn:
            continue
        rst = torch.load(fn, weights_only=True)
        yield fn, rst


def label_informativeness(edge_index, labels, adj=None):
    num_classes = labels.max().item() + 1

    if adj is None:
        adj = edge_index_to_adj(edge_index, labels.size(0)).float()
    adj = (adj + adj.t()).bool().float()
    adj.fill_diagonal_(0)
    
    total_num_edges = adj.sum()

    edges = adj.nonzero().t().tolist()
    
    num_c1c2 = defaultdict(int)
    for i, j in zip(*edges):
        label_i, label_j = labels[i].item(), labels[j].item()
        num_c1c2[tuple(sorted((label_i, label_j)))] += 1
        # num_c1c2[(label_j, label_i)] += 1
    p_c1c2 = {k: v / total_num_edges for k, v in num_c1c2.items()}

    # degree weighted distribution of class labels
    deg = adj.sum(dim=1)
    p_c = dict()
    for i in range(num_classes):
        idx = (labels == i).nonzero().view(-1)
        p_c[i] = deg[idx].sum().item() / total_num_edges
    # import ipdb; ipdb.set_trace()
    return 2 - (sum([x * np.log(x) for x in p_c1c2.values()]) /
        sum([x * np.log(x) for x in p_c.values()])).item()


def test_receptive_field(adj, train_nodes, test_nodes, n_layers):
    """
    For each test node, count of times it appears in the receptive
    field of train nodes.
    """
    _adj = (adj + adj.t()).bool().float()
    _adj = power_adj(adj, n_layers)
    _adj = (_adj > 0).int()
    _adj.fill_diagonal_(0)
    return _adj[train_nodes, :][:, test_nodes].sum(dim=0)


def edge_diff(old_adj, new_adj, labels):
    """
    Compare edges in old and new adjacency matrices.
    """
    old_edges = old_adj.nonzero().t().tolist()
    new_edges = new_adj.nonzero().t().tolist()
    
    del_edges = np.setdiff1d(old_edges, new_edges).tolist()
    add_edges = np.setdiff1d(new_edges, old_edges).tolist()
    
    n_intra_del_edges = len(filter(lambda x: labels[x[0]] == labels[x[1]], del_edges))
    n_inter_del_edges = len(del_edges) - n_intra_del_edges
    n_intra_add_edges = len(filter(lambda x: labels[x[0]] == labels[x[1]], add_edges))
    n_inter_add_edges = len(add_edges) - n_intra_add_edges

    print(f"Num del edges: {len(del_edges)} | "
          f"intra: {n_intra_del_edges}, inter: {n_inter_del_edges}")
    print(f"Num add edges: {len(add_edges)} | "
          f"intra: {n_intra_add_edges}, inter: {n_inter_add_edges}")

