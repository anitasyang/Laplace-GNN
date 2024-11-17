from copy import deepcopy
import os
import os.path as osp
from pathlib import Path
import glob
import re
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import homophily
from collections import defaultdict

from matplotlib import pyplot as plt

from laplace import Laplace

from model import STEGCN, ClipGCN, GCN, LoRASTEGCN
from utils import edge_index_to_adj, adj_to_edge_index, \
    load_data, knn_graph, argument_parser


def marglik_optimization(
        model,
        train_indices,
        train_labels,
        val_indices=None,
        val_labels=None,
        y=None,
        stop_criterion='marglik',
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

    if stop_criterion == 'valloss' and val_indices is None:
        raise ValueError("Validation mask is required for val loss stopping criterion")

    if not osp.exists(learned_graphs_dir):
        os.makedirs(learned_graphs_dir)
    if learned_graphs_dir is not None:
        torch.save(model.adj,
                    osp.join(learned_graphs_dir, "epoch_0.pt"))

    if 'adj' not in [k for k, _ in model.named_parameters()]:
        raise ValueError("Expected 'adj' in model parameters")
        
    no_adj_update = args.model_type in ['gcn', 'gat']

    # use weight decay
    optimizer = torch.optim.Adam(
        [v for k, v in model.named_parameters() if 'adj' not in k],
        lr=lr, weight_decay=weight_decay)

    if args.model_type == 'lorastegcn':
        adj_optimizer = torch.optim.SGD(
            [model.adj_lora_A, model.adj_lora_B], lr=lr_adj)
        print("Using LoRA adj optimizer")
    else:
        adj_optimizer = torch.optim.SGD([model.adj], lr=lr_adj)
    
    losses = list()
    best_model_dict = None
    
    best_marglik = np.inf
    margliks = list()

    best_val_loss = np.inf
    val_losses = list()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(train_indices, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    

    N = train_labels.size(0)

    margliks = []
    model.to(device)
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        model.train()

        # regular training
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

        # if no_adj_update or (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
        #     continue
        if stop_criterion == 'marglik' or not no_adj_update:
            lap = Laplace(
                model,
                "classification",
                subset_of_weights=subset_of_weights,
                hessian_structure=hessian_structure,)
            lap.fit(train_loader)
            marglik = -lap.log_marginal_likelihood()
            margliks.append(marglik.item())

        if not no_adj_update and (epoch % marglik_frequency) == 0 and epoch >= n_epochs_burnin:
            for _ in range(n_hypersteps):
                adj_optimizer.zero_grad()
                marglik.backward()
                adj_optimizer.step()
                lap.fit(train_loader)
                marglik = -lap.log_marginal_likelihood()
                margliks.append(marglik.item())
            
            _adj = model.forward_adj()  # adj used in forward pass
            
            _edge_index = adj_to_edge_index(_adj)
            h = homophily(_edge_index, y.to(device))
            
            num_edges = _edge_index.size(1)
            # Save intermediate edge indices
            if learned_graphs_dir is not None:
                torch.save(_edge_index,
                        osp.join(learned_graphs_dir, f"epoch_{epoch}.pt"))
            print(f"Epoch {epoch}: Marglik={-margliks[-1]:.2f}, Num edges={num_edges}, Homophily={h:.3f}")
        
        if val_indices is not None and val_labels is not None:
            val_f = model(val_indices)
            val_loss = criterion(val_f, val_labels).item()
            val_losses.append(val_loss)
            # val_acc = (val_f.argmax(dim=1) == val_labels).sum().item() / val_labels.size(0)

        # early stopping on marginal likelihood
        if stop_criterion == 'marglik' and margliks[-1] < best_marglik and epoch > n_epochs_burnin:
            best_model_dict = deepcopy(model.state_dict())
            best_model_epoch = epoch
            best_marglik = margliks[-1]
            # print(f'Epoch {epoch}: Saving new best model based on marglik. Marglik={-best_marglik:.2f}')
        elif stop_criterion == 'valloss' and val_losses[-1] < best_val_loss:
            best_model_dict = deepcopy(model.state_dict())
            best_model_epoch = epoch
            best_val_loss = val_losses[-1]
            # print(f'Epoch {epoch}: Saving new best model based on val loss. Val Loss={best_val_loss:.2f}')
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss={epoch_loss:.3f}, ' +
                  f'Perf={epoch_perf / N:.3f}, ' +
                  f'Val Loss={val_loss:.3f}')

    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
    
    lap = Laplace(
        model,
        "classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,)
    lap.fit(train_loader)
    return lap, margliks, val_losses, losses, best_model_epoch


def mean_eval(lap: Laplace, indices, labels, criterion):
    lap.model.eval()
    f = lap.model(indices)
    loss = criterion(f, labels).item()
    acc = (f.argmax(dim=1) == labels).sum().item() * 100 \
        / labels.size(0)
    return loss, acc


def mc_eval(lap: Laplace, indices, labels, criterion,
            pred_type='nn', n_samples=100, diagonal_output=False):
    lap.model.eval()
    f = lap(
        indices,
        pred_type=pred_type,
        link_approx='mc',
        n_samples=n_samples,
        diagonal_output=diagonal_output,)
    loss = criterion(f, labels).item()
    acc = (f.argmax(dim=1) == labels).sum().item() * 100 \
        / labels.size(0)
    return loss, acc

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    
    # device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_num}')
        print(f"Using GPU ID: {args.gpu_num}")
    else:
        device = torch.device('cpu')

    data = load_data(
        args.dataset, args.n_data_rand_splits)
    data.to(device)

    out_dir = osp.join(args.base_out_dir, args.dataset)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    # initialize adjacency matrix
    if args.init_graph == 'original':
        adj = edge_index_to_adj(
            data.edge_index, num_nodes=data.x.size(0)).float()
        # import ipdb; ipdb.set_trace()
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

    if args.stop_criterion == None:
        args.stop_criterion = 'marglik' if args.model_type in \
            ['stegcn', 'lorastegcn'] else 'valloss'

    if args.model_type in ['gcn', 'gat'] and args.stop_criterion == 'marglik':
        raise ValueError(
            "Marglik should not be used as the stop criteria for GCN and GAT models")

    if 'ste' in args.model_type and args.stop_criterion == 'valloss':
        raise ValueError(
            "Validation loss should not be used as the stop criteria for STE models")
    
    best_model_dict = None
    best_marglik = -np.inf
    best_model_meta = dict()

    margliks = list()
    model_metadata = list()

    criterion = torch.nn.CrossEntropyLoss()

    learned_graphs_dir = osp.join(
        out_dir,
        '_'.join([
            args.model_type,
            args.hessian_structure,
            args.subset_of_weights, 'strucs']))


    lr_adjs = [0.3, 0.4, 0.5, 0.6, 0.7]if args.lr_adj is None \
        else [args.lr_adj]

    
    if args.model_type in ['gcn', 'gat']:
        lr_adjs = [0.]  # no adj learning for GCN and GAT
    
    if args.model_type in ['stegcn', 'lorastegcn']:
        # grid search over STE thresholds if not specified
        ste_thresholds = np.arange(0.1, 1.0, 0.1) if args.ste_thresh is None \
            else [args.ste_thresh]
    else:
        ste_thresholds = [0.]  # no ste threshold for other models
    
    lrs = [args.lr] if args.lr is not None \
        else [0.05, 0.1]
    weight_decays = [args.weight_decay] if args.weight_decay is not None \
        else [5e-4, 5e-5, 5e-6]
    hidden_channels = [args.hidden_channels] if args.hidden_channels is not None \
        else [16, 32, 64]
    dropouts = [args.dropout_p] if args.dropout_p is not None \
        else [0.2, 0.3, 0.4, 0.5]
    lora_rs = [16, 32, 64] if (args.lora_r is None and 'lora' in args.model_type) \
        else [args.lora_r]

    
    for lora_r in tqdm(lora_rs, desc='LoRA r'):
        for lr in tqdm(lrs, desc='Learning rate'):
            for weight_decay in tqdm(weight_decays, desc='Weight decay'):
                for hidden_channel in tqdm(hidden_channels, desc='Hidden channels'):
                    for dropout in tqdm(dropouts, desc='Dropout'):
                        for lr_adj in tqdm(lr_adjs, desc='Learning rate adj'):
                            for thres in tqdm(ste_thresholds, desc='STE threshold'):
                                print('-' * 10,
                                    f"lr={lr:.2f}, ",
                                    f"weight_decay={weight_decay}, ",
                                    f"hidden_channels={hidden_channel}, ",
                                    f"dropout={dropout:.2f}, ",
                                    f"lr_adj={lr_adj:.2f}, ",
                                    f"ste_thres={thres:.2f}",
                                    f"lora_r={str(lora_r)}",
                                    '-' * 10)
                                
                                stats = defaultdict(list)

                                n_splits = data.train_indices.size(1)
                                for split_idx in range(n_splits):
                                    train_indices = data.train_indices[:, split_idx]
                                    train_labels = data.y[train_indices]
                                    val_indices = data.val_indices[:, split_idx]
                                    val_labels = data.y[val_indices]
                                    test_indices = data.test_indices[:, split_idx]
                                    test_labels = data.y[test_indices]

                                    for repeat in range(args.n_repeats):
                                        print('-' * 20,
                                              f"Split: {split_idx + 1} / {n_splits} (Repeat {repeat + 1})",
                                              '-' * 20)
                                        if args.model_type == 'stegcn':
                                            model = STEGCN(
                                                in_channels=data.x.size(1),
                                                hidden_channels=hidden_channel,
                                                out_channels=data.y.max().item() + 1,
                                                threshold=thres,
                                                dropout_p=dropout,
                                                init_adj=adj,
                                                X=data.x,)
                                        elif args.model_type == 'clipgcn':
                                            model = ClipGCN(
                                                in_channels=data.x.size(1),
                                                hidden_channels=hidden_channel,
                                                out_channels=data.y.max().item() + 1,
                                                dropout_p=dropout,
                                                init_adj=adj,
                                                X=data.x,)
                                        elif args.model_type == 'gcn':
                                            model = GCN(
                                                in_channels=data.x.size(1),
                                                hidden_channels=hidden_channel,
                                                out_channels=data.y.max().item() + 1,
                                                init_adj=adj,
                                                X=data.x,
                                                dropout_p=dropout,)
                                        elif args.model_type == 'lorastegcn':
                                            model = LoRASTEGCN(
                                                in_channels=data.x.size(1),
                                                hidden_channels=hidden_channel,
                                                out_channels=data.y.max().item() + 1,
                                                init_adj=adj,
                                                r=lora_r,
                                                lora_alpha=args.lora_alpha,
                                                X=data.x,
                                                dropout_p=dropout,)
                                        # elif args.model_type == 'gat':
                                        #     from models import GAT
                                        #     model = GAT(
                                        #         in_channels=data.x.size(1),
                                        #         hidden_channels=hidden_channel,
                                        #         out_channels=data.y.max().item() + 1,
                                        #         num_layers=2,
                                        #         init_adj=adj,
                                        #         X=data.x,
                                        #         heads=args.heads,
                                        #         dropout_p=dropout,)
                                        else:
                                            raise ValueError(f"Unknown model type: {args.model_type}")

                                        lap, _, _, _, best_model_epoch = marglik_optimization(
                                                                            model,
                                                                            train_indices=train_indices,
                                                                            train_labels=train_labels,
                                                                            val_indices=val_indices,
                                                                            val_labels=val_labels,
                                                                            y=data.y,
                                                                            stop_criterion=args.stop_criterion,
                                                                            lr_adj=lr_adj,
                                                                            lr=lr,
                                                                            weight_decay=weight_decay,
                                                                            n_epochs=args.n_epochs,
                                                                            n_hypersteps=args.n_hypersteps,
                                                                            n_epochs_burnin=args.n_epochs_burnin,
                                                                            marglik_frequency=args.marglik_frequency,
                                                                            subset_of_weights=args.subset_of_weights,
                                                                            hessian_structure=args.hessian_structure,
                                                                            device=device,
                                                                            learned_graphs_dir=learned_graphs_dir,
                                                                        )    

                                        marglik = lap.log_marginal_likelihood().item()
                                        out_adj = lap.model.forward_adj()
                                        edge_index = adj_to_edge_index(out_adj)
                                        h = homophily(edge_index, data.y.to(device))
                                        print(f"Final num edges: {edge_index.size(1)}, Homophily: {h:.3f}")
                                        print(f"Final Marglik: {marglik:.2f}")
                                        
                                        # mean predictive
                                        mean_val_loss, mean_val_acc = mean_eval(lap, val_indices, val_labels, criterion)
                                        mean_test_loss, mean_test_acc = mean_eval(lap, test_indices, test_labels, criterion)
                                        
                                        # nn posterior predictive
                                        nn_mc_val_loss, nn_mc_val_acc = mc_eval(
                                            lap, val_indices, val_labels, criterion, pred_type='nn')
                                        nn_mc_test_loss, nn_mc_test_acc = mc_eval(
                                            lap, test_indices, test_labels, criterion, pred_type='nn')
                                        
                                        # # glm posterior predictive
                                        # glm_mc_val_loss, glm_mc_val_acc = mc_eval(
                                        #     lap, val_indices, val_labels, criterion, pred_type='glm')
                                        # glm_mc_test_loss, glm_mc_test_acc = mc_eval(
                                        #     lap, test_indices, test_labels, criterion, pred_type='glm')
                                        # stats['glm mc val loss'].append(glm_mc_val_loss)
                                        # stats['glm mc val acc'].append(glm_mc_val_acc)
                                        # stats['glm mc test los '].append(glm_mc_test_acc)

                                        # track test acc at mean
                                        stats['marglik'].append(marglik)
                                        stats['mean val loss'].append(mean_val_loss)
                                        stats['mean val acc'].append(mean_val_acc)
                                        stats['nn mc val loss'].append(nn_mc_val_loss)
                                        stats['nn mc val acc'].append(nn_mc_val_acc)
                                        stats['mean test loss'].append(mean_test_loss)
                                        stats['mean test acc'].append(mean_test_acc)
                                        stats['nn mc test loss'].append(nn_mc_test_loss)
                                        stats['nn mc test acc'].append(nn_mc_test_acc)
                                        stats['homophily'].append(h)
                                        stats['num edges'].append(edge_index.size(1))
                                        stats['best model epoch'].append(best_model_epoch)
                                        

                                        print(f'Marglik={marglik}, Mean Val Acc={mean_val_acc:.3f}, Mean Test Acc={mean_test_acc:.3f}')

                                        # save model
                                        if marglik > best_marglik:
                                            best_marglik = marglik
                                            best_model_dict = deepcopy(lap.model.state_dict())
                                            print('Saving best model to:', out_dir)
                                            torch.save(lap.model.state_dict(), osp.join(out_dir, f'{args.dataset}_model.pt'))
                                            # best_model_meta = meta
                                        
                                        margliks.append(marglik)
                                meta = {k: (np.mean(v), np.std(v)) for k, v in stats.items()}
                                meta['lr'] = lr
                                meta['weight decay'] = weight_decay
                                meta['hidden channels'] = hidden_channel
                                meta['ste threshold'] = thres
                                meta['lr_adj'] = lr_adj
                                meta['dropout'] = dropout
                                meta['lora_r'] = lora_r

                                model_metadata.append(meta)
    
    
    def print_meta(meta):
        for k, v in meta.items():
            if isinstance(v, tuple):
                print(f"\t{k}: {v[0]:.3f} ({v[1]:.3f})")
            else:
                print(f"\t{k}: {v}")
    
    
    # best model by marglik
    idx = np.argmax([x['marglik'][0] for x in model_metadata])
    print("Best model by Marglik: ")
    print_meta(model_metadata[idx])

    # best model by val loss
    idx = np.argmin([x['mean val loss'][0] for x in model_metadata])
    print("Best model by Val Loss: ")
    print_meta(model_metadata[idx])
    
    # save metadata
    rst_file = osp.join(out_dir, f'{args.dataset}_{args.model_type}_rst.pt')
    with open(rst_file, 'wb') as f:
        torch.save(model_metadata, f)
    print(f"Saved results to {rst_file}")
    print(f"Intermediate graphs saved to {learned_graphs_dir}")
    import ipdb; ipdb.set_trace()
