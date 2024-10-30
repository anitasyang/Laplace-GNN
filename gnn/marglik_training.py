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
        train_mask,
        y,
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
        learned_graphs_dir=None,
        no_adj_update=False):
    """
    Parameters
    ----------
    model : torch.nn.Module
        torch neural network model with 'adj' as a parameter
    learned_graphs_dir : str
        directory to save intermediate adjacency matrices
    """

    if no_adj_update:
        print("Not updating adjacency matrix")

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
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

    N = train_labels.size(0)

    margliks = []
    model.to(device)
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        model.train()
        for data in train_loader:
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
            print(f"Epoch {epoch}: Loss={epoch_loss:.3f}, Perf={epoch_perf / N:.3f}")

        if no_adj_update or (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        lap = Laplace(
            model,
            "classification",
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,)
        lap.fit(train_loader)
        
        for _ in range(n_hypersteps):
            adj_optimizer.zero_grad()
            lap.fit(train_loader)
            marglik = -lap.log_marginal_likelihood()
            marglik.backward()
            adj_optimizer.step()
            margliks.append(marglik.item())
            # import ipdb; ipdb.set_trace()
        _adj = (model.adj > model.threshold).int()
        
        _edge_index = adj_to_edge_index(_adj)
        h = homophily(_edge_index, y.to(device))
        
        num_edges = _edge_index.size(1)
        # Save intermediate edge indices
        if learned_graphs_dir is not None:
            torch.save(_edge_index,
                       osp.join(learned_graphs_dir, f"epoch_{epoch}.pt"))
        print(f"Epoch {epoch}: Marglik={-margliks[-1]:.2f}, Num edges={num_edges}, Homophily={h:.3f}")
        
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
    lap.fit(train_loader)
    return lap, margliks, losses


if __name__ == "__main__":
    parser = argument_parser()
    parser.add_argument('--no_adj_update', action='store_true',
                        help='Do not update adjacency matrix')
    args = parser.parse_args()
    
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU ID: 0")
    else:
        device = torch.device('cpu')

    data = load_data(args.dataset)
    train_mask = data.train_mask[:, 0]
    val_mask = data.val_mask[:, 0]
    test_mask = data.test_mask[:, 0]

    val_indices = torch.nonzero(val_mask).flatten().to(device)
    val_labels = data.y[val_mask].to(device)

    test_indices = torch.nonzero(test_mask).flatten().to(device)
    test_labels = data.y[test_mask].to(device)

    print(f"Train nodes: {train_mask.sum()}, Test nodes: {data.test_mask.sum()}")

    out_dir = osp.join(args.base_out_dir, args.dataset)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
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
    
    best_model_dict = None
    best_marglik = -np.inf
    best_model_meta = dict()

    margliks = list()
    model_metadata = list()

    criterion = torch.nn.CrossEntropyLoss()

    learned_graphs_dir = osp.join(
        out_dir,
        '_'.join([args.hessian_structure,
                args.subset_of_weights, 'strucs']))


    # grid search over STE thresholds if not specified
    ste_thresholds = np.arange(0.1, 1.0, 0.1) if args.ste_thresh is None \
        else [args.ste_thresh]
    lr_adjs = np.arange(0.1, 0.5, 0.1) if args.lr_adj is None \
        else [args.lr_adj]
    
    if args.no_adj_update:
        ste_thresholds = [0.5]
        lr_adjs = [0.]
    
    for lr_adj in lr_adjs:
        for thres in ste_thresholds:
            print('-' * 20, f"Adj lr={lr_adj:.2f}, STE threshold={thres:.2f}", '-' * 20)
            # custom gcn model
            model = SimpleGCN(
                in_channels=data.x.size(1),
                hidden_channels=args.hidden_channels,
                out_channels=data.y.max().item() + 1,
                threshold=thres,
                dropout_p=args.dropout_p,
                init_adj=adj,
                X=data.x.to(device),
                update_adj=not args.no_adj_update,)

            lap, _margliks, _losses = marglik_optimization(
                                                model,
                                                train_mask,
                                                y=data.y,
                                                lr_adj=lr_adj,
                                                lr=args.lr,
                                                weight_decay=args.weight_decay,
                                                n_epochs=args.n_epochs,
                                                n_hypersteps=args.n_hypersteps,
                                                n_epochs_burnin=args.n_epochs_burnin,
                                                marglik_frequency=args.marglik_frequency,
                                                subset_of_weights=args.subset_of_weights,
                                                hessian_structure=args.hessian_structure,
                                                device=device,
                                                learned_graphs_dir=learned_graphs_dir,
                                                no_adj_update=args.no_adj_update,
                                            )    

            marglik = lap.log_marginal_likelihood().item()
            edge_index = adj_to_edge_index((model.adj > model.threshold).int())
            h = homophily(edge_index, data.y.to(device))
            print(f"Final num edges: {edge_index.size(1)}, Homophily: {h:.3f}")
            print(f"Final Marglik: {marglik:.2f}")
            
            # mean predictive
            lap.model.eval()
            mean_val_f = lap.model(val_indices)
            mean_val_loss = criterion(mean_val_f, val_labels).item()
            mean_val_acc = (mean_val_f.argmax(dim=1) == val_labels).sum().item() \
                / val_labels.size(0)
            
            mean_test_f = model(test_indices)
            mean_test_loss = criterion(mean_test_f, test_labels).item()
            mean_test_acc = (mean_test_f.argmax(dim=1) == test_labels).sum().item() \
                / test_labels.size(0)
            
            # posterior predictive
            mc_val_f = lap(
                val_indices,
                pred_type='nn',
                link_approx='mc',
                n_samples=32,
                diagonal_output=False,)
            mc_val_loss = criterion(mc_val_f, val_labels).item()
            mc_val_acc = (mc_val_f.argmax(dim=1) == val_labels).sum().item() \
                / val_labels.size(0)

            mc_test_f = lap(
                test_indices,
                pred_type='nn',
                link_approx='mc',
                n_samples=32,
                diagonal_output=False,)
            mc_test_loss = criterion(mc_test_f, test_labels).item()
            mc_test_acc = (mc_test_f.argmax(dim=1) == test_labels).sum().item() \
                / test_labels.size(0)

            meta = {
                'STE threshold': thres,
                'lr_adj': lr_adj,
                'marglik': marglik,
                'mean val loss': mean_val_loss,
                'mean val acc': mean_val_acc,
                'mc val loss': mc_val_loss,
                'mc val acc': mc_val_acc,
                'mean test loss': mean_test_loss,
                'mean test acc': mean_test_acc,
                'mc test loss': mc_test_loss,
                'mc test acc': mc_test_acc,
                'homophily': h,
                'num edges': edge_index.size(1),
            }

            print(f'Marglik={marglik}, Mean Val Acc={mean_val_acc:.3f}, Mean Test Acc={mean_test_acc:.3f}')

            # save model
            if marglik > best_marglik:
                best_marglik = marglik
                best_model_dict = deepcopy(lap.model.state_dict())
                print('Saving best model to:', out_dir)
                torch.save(lap.model.state_dict(), osp.join(out_dir, f'{args.dataset}_model.pt'))
                best_model_meta = meta
            
            margliks.append(marglik)
            model_metadata.append(meta)
    
    if best_model_meta:
        print(f"Best model by Marglik: " + 
            '\n\t'.join(f'{k}={v:.3f}' for k, v in best_model_meta.items()))
    
    # best model by val loss
    idx = np.argmin([x['mean val loss'] for x in model_metadata])
    print(f"Best model by Val Loss: " +
          '\n\t'.join(f'{k}={v:.3f}' for k, v in model_metadata[idx].items()))
    # import ipdb; ipdb.set_trace()
    
    # save metadata
    rst_file = osp.join(out_dir, f'{args.dataset}_rst.pkl')
    with open(rst_file, 'wb') as f:
        torch.save(model_metadata, f)
    print(f"Saved results to {rst_file}")
    