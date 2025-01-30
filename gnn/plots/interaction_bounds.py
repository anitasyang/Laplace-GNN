import os.path as osp
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from gnn.utils import load_data, edge_index_to_adj, \
    normalize_adj, get_knn_graph
from gnn.plots.utils import test_receptive_field, get_learned_graphs
from gnn.models.utils import power_adj
from torch_geometric.nn import knn_graph
from torch_geometric.utils import homophily

"""
Plot the power of the learned graphs
Calculate the bounds for inter/intra-class interactions.
(Measure oversquashing)
"""


def global_homophily(adj, labels):
    # _adj = (adj + adj.T).bool().float()
    _adj = adj.clone()
    _adj.fill_diagonal_(0)
    indices = _adj.nonzero()
    same_labels = (labels[indices[:, 0]] == labels[indices[:, 1]])
    if not len(same_labels):
        return 0
    return same_labels.sum().item() / len(same_labels)

def local_homophily(adj, nodes, labels):  # alpha
    # _adj = (adj + adj.T).bool().float()
    _adj = adj.clone()
    _adj.fill_diagonal_(0)
    local_h = dict()
    for u in nodes.tolist():
        neigh = _adj[u, :]
        neigh_idx = neigh.nonzero().view(-1)

        if len(neigh_idx) == 0:
            local_h[u] = 0
            continue
        n_same_class = (labels[neigh_idx] == labels[u]).sum().item()
        n_neigh = len(neigh_idx)
        local_h[u] = n_same_class / n_neigh
    return local_h


def avg_local_homophilies(adj, train_nodes, test_nodes, labels):
    # import ipdb; ipdb.set_trace()
    global_h = global_homophily(adj, labels)
    train_local_h = local_homophily(adj, train_nodes, labels)
    test_local_h = local_homophily(adj, test_nodes, labels)
    # h_shifts = sum([(v - knng_global_h) for v in knng_test_local_h.values()])
    avg_train_local_h = sum(train_local_h.values()) / len(train_nodes)
    avg_test_local_h = sum(test_local_h.values()) / len(test_nodes)
    # import ipdb; ipdb.set_trace()
    return global_h, avg_train_local_h, avg_test_local_h


def avg_receptive_field_degree(adj, nodes, n_layers):
    """
    Count average degree of nodes
    """
    adj, nodes = adj.cpu(), nodes.cpu()
    adj = (adj + adj.T).bool().float()
    adj.fill_diagonal_(1)
    adj = power_adj(adj, n_layers)
    adj.fill_diagonal_(0)
    return adj[nodes, :].count_nonzero().item() / len(nodes)


def interaction_bound(edge_index=None, adj=None, test_nodes=None):
    if edge_index is None and adj is None:
        raise ValueError("Either edge_index or adj must be provided")
    if adj is None:
        adj = edge_index_to_adj(edge_index, data.num_nodes)
    adj = (adj + adj.T).bool().float()
    # import ipdb; ipdb.set_trace()
    norm_adj = normalize_adj(adj.float())
    norm_adj = power_adj(norm_adj, n_layers)
    # norm_adj = normalize_adj(adj.float()) ** n_layers
    if test_nodes is not None:
        test_norm_adj = norm_adj[test_nodes, :]
        norm_adj = torch.zeros_like(norm_adj)
        norm_adj[test_nodes, :] = test_norm_adj
        norm_adj[:, test_nodes] = test_norm_adj.T
    total = norm_adj.sum()
    same_class = 0
    adj_layers = power_adj(adj, n_layers)
    # import ipdb; ipdb.set_trace()
    for i, nodes in label_to_nodes.items():
        sum_nodes = norm_adj[nodes, :][:, nodes].sum()
        count_nodes = adj[nodes, :][:, nodes].sum().item()
        count_nodes_layers = adj_layers[nodes, :][:, nodes].sum().item()
        # print(f"\t Class {i}: {sum_nodes.item():.3f}, {count_nodes}, {count_nodes_layers}")
        same_class += sum_nodes
    return same_class.item(), (total - same_class).item()

if __name__ == "__main__":

    OUT_PLOT_DIR = "images"

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='Name of dataset')
    parser.add_argument('--init_graph', type=str, default='original',
                        help='Type of initial graph')
    parser.add_argument('--model_type', type=str, default='stegcn',
                        help='Type of model')
    parser.add_argument('--epoch_num', type=int,
                        help='Epoch number of learned graph')
    args = parser.parse_args()

    n_layers = 2
    split_idx = 0

    # Load the learned graph
    data = load_data(args.dataset)

    train_nodes = data.train_indices[:, split_idx]
    val_nodes = data.val_indices[:, split_idx]
    test_nodes = data.test_indices[:, split_idx]

    rst_graphs = get_learned_graphs(
        out_dir=f"/home/anita/work/laplace-gnn-results/{args.dataset}",
        init_graph=args.init_graph,
        model_type=args.model_type,
        epoch_num=args.epoch_num)

    num_classes = data.y.max().item() + 1
    new_idx = dict()
    boundary_idx = []
    label_to_nodes = dict()
    for i in range(num_classes):
        idx = (data.y == i).nonzero().view(-1)
        label_to_nodes[i] = idx
        boundary_idx.append((len(new_idx), len(new_idx) + len(idx)))
        new_idx.update(
            dict(zip(idx.tolist(),
                        [j + len(new_idx) for j in range(len(idx))])))


    def shuffle_nodes(edge_index):
        edge_index = np.array(edge_index)
        return torch.Tensor(np.vectorize(new_idx.get)(edge_index))


    def plot(edge_index, title, out_fn, power=1):
        edge_index = shuffle_nodes(edge_index)
        adj = edge_index_to_adj(edge_index, data.num_nodes)
        adj = (adj + adj.T).bool().float()
        adj.fill_diagonal_(1)
        # adj = adj.float() ** power
        # print((adj != adj.T).sum().item())
        # adj = normalize_adj(adj.float()) ** power
        adj = power_adj(adj, power)
        # import ipdb; ipdb.set_trace()
        # print("Diag sum:", adj.diag().sum().item())
        fig, ax = plt.subplots()
        cax = ax.matshow(adj, cmap='viridis')
        x_min, x_max = ax.get_xlim()  # Limits in the x-direction
        max_idx = max([x for _, x in boundary_idx])

        idx_incr = (x_max - x_min) / max_idx
        for start_idx, stop_idx in boundary_idx:
            start = start_idx * idx_incr - 0.5
            stop = stop_idx * idx_incr - 0.5
            if stop_idx != max_idx:
                ax.plot([stop, stop], [start, stop],
                        color='red', linestyle='--', linewidth=1)
                ax.plot([start, stop], [stop, stop],
                        color='red', linestyle='--', linewidth=1)
            ax.plot([start, start], [start, stop],
                    color='red', linestyle='--', linewidth=1)
            ax.plot([start, stop], [start, start],
                    color='red', linestyle='--', linewidth=1)
        plt.title(title)
        plt.savefig(osp.join(OUT_PLOT_DIR, out_fn))
        return adj


    def plot_avg_local_h(epochs, train_local_hs, test_local_hs):
        fig, ax1 = plt.subplots()
        pre_epochs = [x for x in range(0, min(epochs), 20)]
        epochs += pre_epochs

        if args.init_graph == 'knng':
            train_local_hs += [knng_avg_train_local_h] * len(pre_epochs)
            test_local_hs += [knng_avg_test_local_h] * len(pre_epochs)
        elif args.init_graph == 'original':
            train_local_hs += [og_avg_train_local_h] * len(pre_epochs)
            test_local_hs += [og_avg_test_local_h] * len(pre_epochs)
        
        sorted_idx = np.argsort(epochs)
        epochs, train_local_hs, test_local_hs = np.array(epochs), \
            np.array(train_local_hs), np.array(test_local_hs)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Avg Local Homophily', color='blue')
        ax1.plot(epochs[sorted_idx], train_local_hs[sorted_idx],
                    color='cornflowerblue',
                    label='Train')
        ax1.plot(epochs[sorted_idx], test_local_hs[sorted_idx],
                    color='mediumblue',
                    label='Test')
        ax1.tick_params(axis='y', labelcolor='blue')
        # ax1.legend(loc='upper left')

        losses = torch.load(osp.join(
            f"/home/anita/work/laplace-gnn-results/{args.dataset}/{args.init_graph}_{args.model_type}_kron_all_strucs",
            "losses.pt"), weights_only=True)
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.set_ylabel('Loss', color='red')
        ax2.scatter(losses['epochs'], losses['train_loss'], color='palevioletred',
                    label='Train', s=8)
        ax2.scatter(losses['epochs'], losses['val_loss'], color='crimson',
                    label='Validation', s=8)
        ax2.tick_params(axis='y', labelcolor='red')
        # ax2.legend(loc='upper right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        out_fn = osp.join(OUT_PLOT_DIR,
                          f"{args.dataset}_{args.init_graph}_{args.model_type}_marglik_local_h.png")
        plt.savefig(out_fn)
        print(f"Saved avg local homophily to {out_fn}")

    def plot_interaction_bounds(epochs, global_intra, global_inter,
                                test_intra, test_inter):
        # import ipdb; ipdb.set_trace()
        epochs, global_intra, global_inter = np.array(epochs), \
            np.array(global_intra), np.array(global_inter)
        test_intra, test_inter = np.array(test_intra), np.array(test_inter)
        sorted_idx = np.argsort(epochs)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('$||\hat{\mathbf{A}}_{\mathrm{intra/inter}}^{n_\mathrm{layers}}||_1$')
        ax1.plot(epochs[sorted_idx], global_intra[sorted_idx], color='blue', label='Global Intra')
        ax1.plot(epochs[sorted_idx], global_inter[sorted_idx], color='red', label='Global Inter')
        ax1.plot(epochs[sorted_idx], test_intra[sorted_idx], color='blue', label='Test Intra',
                    linestyle='--')
        ax1.plot(epochs[sorted_idx], test_inter[sorted_idx], color='red', label='Test Inter',
                    linestyle='--')
        ax1.legend(loc='upper left')
        out_fn = osp.join(OUT_PLOT_DIR,
                            f"{args.dataset}_{args.init_graph}_{args.model_type}_interaction_bounds.png")
        plt.savefig(out_fn)
        print(f"Saved interaction bounds to {out_fn}")


    def count_type_edges(edge_index, labels):
        intra_edges = (labels[edge_index[0]] == labels[edge_index[1]]).sum().item()
        inter_edges = len(edge_index[0]) - intra_edges
        return intra_edges, inter_edges


    def plot_edge_dist(adj1, adj2, fn):
        fig1, ax = plt.subplots()
        deg1 = adj1.sum(dim=1)
        deg2 = adj2.sum(dim=1)
        # import ipdb; ipdb.set_trace()
        ax.hist(np.arange(deg1.size(0)), np.arange(deg1.size(0)) + 0.5,
                    weights=deg1.cpu().numpy(), alpha=0.5, label='KNN')
        ax.hist(np.arange(deg2.size(0)) +0.5, np.arange(deg2.size(0)) + 1,
                    weights=deg2.cpu().numpy(), alpha=0.5, label='Learned')
        ax.legend()
        plt.xlabel("Node")
        plt.ylabel("Degree")
        plt.title("Degree distribution")
        plt.savefig(fn)



    # Original graph
    plot(data.edge_index, "Original graph", f"{args.dataset}_og.png", n_layers)

    intra_bound, inter_bound = interaction_bound(
        edge_index=data.edge_index)
    print("Original graph (intra and inter interaction bounds): ",
        f"{intra_bound:.3f}, {inter_bound:.3f}")
    test_intra_bound, test_inter_bound = interaction_bound(
        edge_index=data.edge_index, test_nodes=test_nodes)
    intra_edges, inter_edges = count_type_edges(data.edge_index, data.y)
    print("\t Original graph test bounds: ",
            f"{test_intra_bound:.3f}, {test_inter_bound:.3f}")
    og_adj = edge_index_to_adj(data.edge_index, data.num_nodes)
    # og_global_h = homophily(data.edge_index, data.y)
    og_global_h = global_homophily(og_adj, data.y)
    print(f"\t Homophily: {og_global_h:.3f}")
    print(f"\t Num edges: {data.edge_index.size(1)} ({intra_edges}, {inter_edges})")
    og_train_local_h = local_homophily(og_adj, train_nodes, data.y)
    og_test_local_h = local_homophily(og_adj, test_nodes, data.y)
    og_test_h_shifts = sum([(v - og_global_h) for v in og_test_local_h.values()])
    og_avg_train_local_h = sum(og_train_local_h.values()) / len(train_nodes)
    og_avg_test_local_h = sum(og_test_local_h.values()) / len(test_nodes)
    og_n_test_receptive_field = test_receptive_field(og_adj, train_nodes, test_nodes, n_layers)
    print(f"\t Homophily test shifts: {og_test_h_shifts:.3f}")
    print(f"\t Avg local train/test homophily: {og_avg_train_local_h:.3f}, {og_avg_test_local_h:.5f}")
    print(f"\t Number of times test nodes appear in the receptive field:",
        og_n_test_receptive_field)

    # KNN graph
    knng_adj, knng_edge_index = get_knn_graph(data.x, k=3, return_edge_index=True)
    plot(knng_edge_index, "KNN graph", f"{args.dataset}_knng.png", n_layers)
    intra_bound, inter_bound = interaction_bound(adj=knng_adj)
    intra_edges, inter_edges = count_type_edges(knng_edge_index, data.y)
    print("KNN graph (intra and inter interaction bounds): ",
        f"{intra_bound:.3f}, {inter_bound:.3f}")
    test_intra_bound, test_inter_bound = interaction_bound(
        adj=knng_adj, test_nodes=test_nodes)
    print("\t KNN graph test bounds: ",
        f"{test_intra_bound:.3f}, {test_inter_bound:.3f}")
    knng_global_h = global_homophily(knng_adj, data.y)
    print(f"\t Homophily: {knng_global_h:.3f}")
    print(f"\t Num edges: {knng_edge_index.size(1)} ({intra_edges}, {inter_edges})")
    knng_train_local_h = local_homophily(knng_adj, train_nodes, data.y)
    knng_test_local_h = local_homophily(knng_adj, test_nodes, data.y)
    h_shifts = sum([(v - knng_global_h) for v in knng_test_local_h.values()])
    knng_avg_train_local_h = sum(knng_train_local_h.values()) / len(train_nodes)
    knng_avg_test_local_h = sum(knng_test_local_h.values()) / len(test_nodes)
    knng_test_receptive_field = test_receptive_field(knng_adj, train_nodes, test_nodes, n_layers)
    print(f"\t Homophily test shifts: {h_shifts:.3f}")
    print(f"\t Avg local train/test homophily: {knng_avg_train_local_h:.3f}, {knng_avg_test_local_h:.5f}")
    print(f"\t Number of times test nodes appear in the receptive field:",
        knng_test_receptive_field)

    epochs = []
    train_local_hs, test_local_hs = [], []
    margliks = []
    # interaction bounds
    global_intra_interaction, global_inter_interaction = [], []
    test_intra_interaction, test_inter_interaction = [], []
    for fn, rst in rst_graphs:
        if 'epoch' not in fn:
            continue
        epoch = fn.split('/')[-1].split('.')[0]
        print(fn)
        edge_index = rst['edge_index'].cpu()
        adj = edge_index_to_adj(edge_index, data.num_nodes)
        # plot(edge_index,
        #     f"Learned from {args.init_graph} graph {epoch}:",
        #     f"{args.dataset}_{epoch}.png", n_layers)
        intra_bound, inter_bound = interaction_bound(edge_index=edge_index)
        print(f"Learned graph {epoch} (intra and inter interaction bounds): ",
            f"{intra_bound:.3f}, {inter_bound:.3f}")
        test_intra_bound, test_inter_bound = interaction_bound(
            edge_index=edge_index, test_nodes=test_nodes)
        
        global_intra_interaction.append(intra_bound)
        global_inter_interaction.append(inter_bound)
        test_intra_interaction.append(test_intra_bound)
        test_inter_interaction.append(test_inter_bound)

        intra_edges, inter_edges = count_type_edges(edge_index, data.y)
        print("\t Bounds for test nodes: ",
            f"{test_intra_bound:.3f}, {test_inter_bound:.3f}")
        global_h = global_homophily(adj, data.y)
        print(f"\t Homophily: {global_h:.3f}")
        print(f"\t Num edges: {edge_index.size(1)} ({intra_edges}, {inter_edges})")
        # plot_edge_dist(knng_adj, adj, f"{args.dataset}_{epoch}_deg.png")
        train_local_h = local_homophily(adj, train_nodes, data.y)
        test_local_h = local_homophily(adj, test_nodes, data.y)
        avg_train_local_h = sum(train_local_h.values()) / len(train_nodes)
        avg_test_local_h = sum(test_local_h.values()) / len(test_nodes)
        test_h_shifts = sum([(v - global_h) for v in test_local_h.values()])
        print(f"\t Homophily test shifts: {test_h_shifts:.3f}")
        print(f"\t Avg local train/test homophily: {avg_train_local_h:.3f}, {avg_test_local_h:.5f}")
        n_test_receptive_field = test_receptive_field(adj, train_nodes, test_nodes, n_layers)
        print(f"\t Number of times test nodes appear in the receptive field:",
            n_test_receptive_field)
        # import ipdb; ipdb.set_trace()

        epochs.append(int(re.search(r'\d+', epoch).group(0)))
        margliks.append(rst['marglik'])
        train_local_hs.append(avg_train_local_h)
        test_local_hs.append(avg_test_local_h)

    plot_interaction_bounds(
        epochs, global_intra_interaction, global_inter_interaction,
        test_intra_interaction, test_inter_interaction)
    plot_avg_local_h(epochs, train_local_hs, test_local_hs)

    # knng_incorrect = [18, 39, 51, 54, 66, 93, 98, 101, 120, 143, 152, 164, 165, 184, 223]
    # incorrect = [18, 29, 39, 51, 54, 66, 93, 98, 120, 143, 152, 164, 165, 184, 223]
    # print("KNNG Incorrect")
    # for u in test_nodes.tolist():
    #     knng_shift = knng_test_local_h[u] - knng_global_h
    #     learned_shift = test_local_h[u] - global_h
    #     print(f"\t Node {u}: knng {knng_test_local_h[u]:.3f}, {knng_shift:.3f}",
    #         "*" if u in knng_incorrect else "",
    #         f"\t learned {test_local_h[u]:.3f}, {learned_shift:.3f}",
    #         "*" if u in incorrect else "")

    # print("Learned Incorrect")
    # for u in test_nodes:
    #     print(f"\t Node {u} shift: {(test_local_h[u] - global_h):.3f}",
    #           "*" if u in incorrect else "")
    # import ipdb; ipdb.set_trace()


    