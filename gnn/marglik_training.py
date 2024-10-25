from copy import deepcopy
import os
import os.path as osp
from pathlib import Path
import glob
import re
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU, Tanh
from torch import nn
from torch.func import grad
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import homophily

from matplotlib import pyplot as plt

from laplace import Laplace

from model import SimpleGCN
from utils import edge_index_to_adj, adj_to_edge_index, \
    load_data, knn_graph, argument_parser


def marglik_optimization(
        model,
        y,
        train_mask,
        lr=0.01,
        lr_adj=0.1,
        weight_decay=0.5,
        n_epochs=100,
        n_hypersteps=20,
        n_epochs_burnin=40,
        marglik_frequency=20,
        subset_of_weights="all",
        hessian_structure="diag",
        device='cpu',
        learned_graphs_dir=None,):
    """
    Parameters
    ----------
    model : torch.nn.Module
        torch neural network model with 'adj' as a parameter
    learned_graphs_dir : str
        directory to save intermediate adjacency matrices
    """

    if not osp.exists(learned_graphs_dir):
        os.makedirs(learned_graphs_dir)
    if learned_graphs_dir is not None:
        torch.save(model.adj,
                    osp.join(learned_graphs_dir, "epoch_0.pt"))

    if 'adj' not in [k for k, _ in model.named_parameters()]:
        raise ValueError("Expected 'adj' in model parameters")
    
    # use weight decay
    optimizer = torch.optim.Adam(
        [v for k, v in model.named_parameters() if k != 'adj'],
        lr=lr, weight_decay=weight_decay)
    adj_optimizer = torch.optim.SGD([model.adj], lr=lr_adj)
    
    best_marglik = np.inf
    best_model_dict = None
    losses = list()
    margliks = list()
    
    criterion = torch.nn.CrossEntropyLoss()

    train_indices = torch.nonzero(train_mask).flatten()
    train_labels = y[train_mask]
    train_dataset = TensorDataset(train_indices, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

    margliks = []
    model.to(device)
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        model.train()
        for data in train_dataloader:
            train_indices, train_labels = data
            train_indices, train_labels = train_indices.to(device), train_labels.to(device)

            f = model(train_indices)
            optimizer.zero_grad()
            loss = criterion(f, train_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_perf += (f.argmax(dim=1) == train_labels).sum().item()
        
        losses.append(epoch_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={epoch_loss:.3f}, Perf={epoch_perf / len(train_indices):.3f}")

        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        lap = Laplace(
            model,
            "classification",
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,)
        lap.fit(train_dataloader)
        
        for _ in range(n_hypersteps):
            adj_optimizer.zero_grad()
            lap.fit(train_dataloader)
            marglik = -lap.log_marginal_likelihood()
            marglik.backward()
            adj_optimizer.step()
            margliks.append(marglik.item())
        
        _adj = (model.adj > model.threshold).int()
        
        _edge_index = adj_to_edge_index(_adj)
        h = homophily(_edge_index, y.to(device))
        
        num_edges = _edge_index.size(1)
        # Save intermediate edge indices
        if learned_graphs_dir is not None:
            torch.save(_edge_index,
                       osp.join(learned_graphs_dir, f"epoch_{epoch}.pt"))
        print(f"Num edges: {num_edges}, Homophily: {h:.3f}")
        print(f"Epoch {epoch}: Marglik={-margliks[-1]:.2f}, ")
        
        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_marglik = margliks[-1]
            print('Saving new best model')
            
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
    
    lap = Laplace(
        model,
        "classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,)
    lap.fit(train_dataloader)
    return lap, model, margliks, losses


if __name__ == "__main__":
    args = argument_parser()
    
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print(f"Using GPU ID: 0")
    else:
        device = torch.device('cpu')

    data = load_data(args.dataset)
    train_mask = data.train_mask[:, 0]
    test_mask = data.test_mask[:, 0]


    print(f"Train nodes: {train_mask.sum()}, Test nodes: {data.test_mask.sum()}")

    # initialize adjacency matrix
    if args.init_graph == 'original':
        adj = torch.eye(data.x.size(0)) + edge_index_to_adj(
            data.edge_index, num_nodes=data.x.size(0))
    elif args.init_graph == 'knng':
        from torch_geometric.nn import knn_graph
        edge_index = knn_graph(data.x, k=args.knng_k,
                               batch=None, loop=False, cosine=False)
        adj = edge_index_to_adj(edge_index, num_nodes=data.x.size(0)).float()
        adj = (adj + adj.t()).bool().float()
        adj += torch.eye(data.x.size(0))
    elif args.init_graph == 'none':
        adj = torch.eye(data.x.size(0))
    else:
        raise ValueError(
            f"Unknown initial graph structure: {args.init_graph}.",
            "Choose from 'original', 'knng', 'none'")

    h = homophily(data.edge_index, data.y)
    print(f"Original num edges: {data.edge_index.size(1)}, Homophily: {h:.3f}")
    print(f"Initial num edges: {adj.sum().int()}")
    
    # custom gcn model
    model_kwargs = {
        'in_channels': data.x.size(1),
        'hidden_channels': args.hidden_channels,
        'out_channels': data.y.max().item() + 1,
        'X': data.x.to(device),
        'threshold': args.ste_thresh,
    }

    model = SimpleGCN(init_adj=adj, **model_kwargs)

    out_dir = osp.join(args.base_out_dir, args.dataset)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    learned_graphs_dir = osp.join(
        out_dir,
        '_'.join([args.hessian_structure,
                  args.subset_of_weights, 'strucs']))
        
    lap, model, margliks, losses = marglik_optimization(
                                        model,
                                        y=data.y,
                                        train_mask=train_mask,
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,
                                        lr_adj=args.lr_adj,
                                        n_epochs=args.n_epochs,
                                        n_hypersteps=args.n_hypersteps,
                                        n_epochs_burnin=args.n_epochs_burnin,
                                        marglik_frequency=args.marglik_frequency,
                                        subset_of_weights=args.subset_of_weights,
                                        hessian_structure=args.hessian_structure,
                                        device=device,
                                        learned_graphs_dir=learned_graphs_dir,
                                    )    
    
    marglik = lap.log_marginal_likelihood()
    edge_index = adj_to_edge_index(model.adj)
    h = homophily(edge_index, data.y.to(device))
    print(f"Final num edges: {edge_index.size(1)}, Homophily: {h:.3f}")
    print(f"Final Marglik: {marglik:.2f}")
    
    # test accuracy
    model.eval()
    test_indices = torch.nonzero(test_mask).flatten()
    test_labels = data.y[test_indices]
    test_indices, test_labels = test_indices.to(device), test_labels.to(device)
    f = model(test_indices)
    acc = (f.argmax(dim=1) == test_labels).sum().item() / test_indices.size(0)
    print(f"Test Accuracy: {acc:.3f}")

    # save model
    torch.save(model.state_dict(), osp.join(out_dir, f'{args.dataset}_model.pt'))
