from copy import deepcopy
import warnings
import os
import os.path as osp
import time
from tqdm import tqdm
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import homophily
from collections import defaultdict

from matplotlib import pyplot as plt

from laplace import Laplace
# from sam.sam import SAM

from gnn.models import GCN, STEGCN, LoRASTEGCN, GAT, \
    STEGraphSAGE, GraphSAGE
from gnn.utils import edge_index_to_adj, adj_to_edge_index, \
    load_data, get_knn_graph, argument_parser, get_learned_graphs_dir, \
    fully_connected_labels, unused_gpu
from gnn.plots.interaction_bounds import avg_local_homophilies, \
                avg_receptive_field_degree

PATIENCE = 20

# Models
model_classes = {
    'stegcn': STEGCN,
    'stegraphsage': STEGraphSAGE,
    'graphsage': GraphSAGE,
    'clipgcn': NotImplemented,  # Replace with ClipGCN when implemented
    'gcn': GCN,
    'lorastegcn': LoRASTEGCN,
    'gat': GAT,
}


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
        n_hyper_stop=None,
        marglik_frequency=20,
        subset_of_weights="all",
        hessian_structure="kron",
        device='cpu',
        learned_graphs_dir=None,
        args_dict={}):
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

    if learned_graphs_dir is not None and not osp.exists(learned_graphs_dir):
        os.makedirs(learned_graphs_dir)

    if 'adj' not in [k for k, _ in model.named_parameters()]:
        raise ValueError("Expected 'adj' in model parameters")
        
    no_adj_update = args_dict['model_type'] in ['gcn', 'gat', 'graphsage']
    n_hyper_stop = n_hyper_stop if n_hyper_stop is not None else n_epochs

    if args_dict['optimizer'] == 'sam':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            [v for k, v in model.named_parameters() if 'adj' not in k],
            base_optimizer, lr=lr, momentum=0.9)
    else:
        # use weight decay
        optimizer = torch.optim.Adam(
            [v for k, v in model.named_parameters() if 'adj' not in k],
            lr=lr, weight_decay=weight_decay)
        # import ipdb; ipdb.set_trace()
    if not no_adj_update:
        if args_dict['model_type'] == 'lorastegcn':
            adj_optimizer = torch.optim.SGD(
                [model.adj_lora_A, model.adj_lora_B], lr=lr_adj,
                weight_decay=args_dict['weight_decay_adj'])
            print("Using LoRA adj optimizer")
        else:
            adj_optimizer = torch.optim.SGD([model.adj], lr=lr_adj,
                                            weight_decay=args_dict['weight_decay_adj'],
                                            momentum=args_dict['momentum_adj'])
            # adj_optimizer = torch.optim.Adam(
            #     [model.adj], lr=lr_adj, weight_decay=weight_decay_adj)
        # adj_scheduler = torch.optim.lr_scheduler.StepLR(
        #     adj_optimizer, step_size=10, gamma=0.01)
        # adj_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     adj_optimizer, 10)

    losses = list()
    # best_model_dict = None
    best_marglik_model_dict = None
    best_valloss_model_dict = None
    
    best_neg_marglik = np.inf
    neg_margliks = list()

    best_valloss = np.inf
    val_losses = list()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(train_indices, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=10000,
                              shuffle=False)

    # for analysis
    eval_indices = torch.arange(y.size(0)).to(device)
    eval_indices = eval_indices[~torch.isin(eval_indices, train_indices)]

    N = train_labels.size(0)

    model.to(device)
    
    _adj = model.full_adj()
    global_h, avg_train_local_h, avg_eval_local_h = avg_local_homophilies(
                _adj, train_indices, eval_indices, y)
    print("Homophily global, local train, local eval:"
            f"{global_h:.3f}, {avg_train_local_h:.3f}, {avg_eval_local_h:.3f}")
    num_edges_train = _adj[train_indices, :].sum().item()
    num_edges_eval = _adj[eval_indices, :].sum().item()
    num_edges_train_train = _adj[train_indices, :][:, train_indices].sum().item()
    num_edges_train_eval = _adj[train_indices, :][:, eval_indices].sum().item()
    num_edges_eval_eval = _adj[eval_indices, :][:, eval_indices].sum().item()
    print(f"Num edges: {_adj.sum().item()} "
          f"(train {num_edges_train}, eval {num_edges_eval}) "
          f"(train-train {num_edges_train_train}, train-eval {num_edges_train_eval}, eval-eval {num_edges_eval_eval})")
    
    torch.autograd.set_detect_anomaly(True)

    marglik_patience, val_patience = 0, 0

    torch.manual_seed(0)
    # If using CUDA
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0) 
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
            
            if args_dict['optimizer'] == 'sam':
                def closure():
                    f = model(train_indices)
                    loss = criterion(f, train_labels)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.step()
            # adj_scheduler.step()
            
            epoch_loss += loss.item()
            epoch_perf += (f.argmax(dim=1) == train_labels).sum().item()
        
        losses.append(epoch_loss)

        # if no_adj_update or (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
        #     continue
        # if stop_criterion == 'marglik':
        # hessian_structure='kron'
        # import ipdb; ipdb.set_trace()
        if epoch < n_hyper_stop and not no_adj_update and \
                (epoch % marglik_frequency) == 0 and epoch >= n_epochs_burnin:
        # if not no_adj_update:  # TODO: change this later
            lap = Laplace(
                model,
                "classification",
                subset_of_weights=subset_of_weights,
                hessian_structure=hessian_structure,)
            lap.fit(train_loader)
            neg_marglik = -lap.log_marginal_likelihood()
            # neg_margliks.append(neg_marglik.item())

            for _ in range(n_hypersteps):
                # model.zero_grad()
                # neg_marglik.backward(retain_graph=True)
                # neg_marglik_grad = model.adj.grad.clone()
                # model.zero_grad()
                # neg_loglik.backward(retain_graph=True)
                # neg_loglik_grad = model.adj.grad.clone()
                # import ipdb; ipdb.set_trace()
                # model.zero_grad()
                adj_optimizer.zero_grad()
                neg_marglik.backward()
                if args_dict['grad_norm']:
                    # import ipdb; ipdb.set_trace()
                    torch.nn.utils.clip_grad_norm_(model.adj, max_norm=1.0)
                adj_optimizer.step()
                lap.fit(train_loader)
                # print(model.adj.sum().item())
                neg_marglik = -lap.log_marginal_likelihood()
                # neg_margliks.append(neg_marglik.item())

            # Analysis
            _adj = model.full_adj()
            
            _edge_index = adj_to_edge_index(_adj)
            h = homophily(_edge_index, y.to(device))
            num_edges = _adj.sum().item()  # includes self-loop
            num_edges_train = _adj[train_indices, :].sum().item()
            num_edges_eval = _adj[eval_indices, :].sum().item()
            num_edges_train_train = _adj[train_indices, :][:, train_indices].sum().item()
            num_edges_train_eval = _adj[train_indices, :][:, eval_indices].sum().item()
            num_edges_eval_eval = _adj[eval_indices, :][:, eval_indices].sum().item()


            global_h, avg_train_local_h, avg_eval_local_h = avg_local_homophilies(
                _adj, train_indices, eval_indices, y)
            print("Homophily global, local train, local eval:"
                  f"{global_h:.3f}, {avg_train_local_h:.3f}, {avg_eval_local_h:.3f}")

            # Save intermediate edge indices
            if learned_graphs_dir is not None:
                torch.save(
                    {'edge_index': _edge_index.detach().cpu(),
                     'marglik': -neg_marglik.item(),
                     'num_edges': num_edges,
                     'homophily': h,
                     'epoch': epoch},
                    osp.join(learned_graphs_dir, f"epoch_{epoch}.pt"))
                torch.save(_adj.detach(), osp.join(learned_graphs_dir, 'latest_adj.pt'))
                # import ipdb; ipdb.set_trace()
            print(f"Epoch {epoch}: Marglik={-neg_margliks[-1]:.2f}, "
                  f"Num edges={_adj.sum().item()} (train {num_edges_train}, eval {num_edges_eval}), "
                  f"(train-train {num_edges_train_train}, train-eval {num_edges_train_eval}, eval-eval {num_edges_eval_eval}), "
                  f"Homophily={h:.3f}")
        
        # calculate marginal likelihood
        lap = Laplace(
            model,
            "classification",
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,)
        lap.fit(train_loader)
        # neg_marglik = -lap.log_marginal_likelihood()
        neg_marglik = -lap.log_marginal_likelihood()
        neg_margliks.append(neg_marglik.item())

        # calculate validation loss
        # if val_indices is not None and val_labels is not None:
        val_f = model(val_indices)
        val_loss = criterion(val_f, val_labels).item()
        val_losses.append(val_loss)
        val_acc = (val_f.argmax(dim=1) == val_labels).sum().item() / val_labels.size(0)

        # early stopping on marginal likelihood
        if ('ste' not in args_dict['model_type']) or (
                'ste' in args_dict['model_type'] and epoch > n_epochs_burnin):
            if not args_dict['early_stop'] or (args_dict['early_stop'] and marglik_patience < PATIENCE):
                if neg_marglik < best_neg_marglik:  # improved
                    best_neg_marglik = neg_marglik.item()
                    best_marglik_model_dict = deepcopy(model.state_dict())
                    best_marglik_epoch = epoch
                    marglik_patience = 0
                    # print(neg_marglik.item())
                else:
                    marglik_patience += 1
            
            if not args_dict['early_stop'] or (args_dict['early_stop'] and val_patience < PATIENCE):
                if val_loss < best_valloss:
                        best_valloss = val_loss
                        best_valloss_model_dict = deepcopy(model.state_dict())
                        best_valloss_epoch = epoch
                        val_patience = 0
                else:
                    val_patience += 1

            if args_dict['early_stop'] and marglik_patience == PATIENCE:
                print("Early stopping on marginal likelihood. No more graph update.")
                no_adj_update = True
                marglik_patience += 1
        
        # if stop_criterion == 'marglik' and neg_margliks[-1] < best_neg_marglik and epoch > n_epochs_burnin:
        #     best_model_dict = deepcopy(model.state_dict())
        #     best_model_epoch = epoch
        #     best_neg_marglik = margliks[-1]
        #     # print(f'Epoch {epoch}: Saving new best model based on marglik. Marglik={-best_neg_marglik:.2f}')
        # elif stop_criterion == 'valloss' and val_losses[-1] < best_val_loss:
        #     best_model_dict = deepcopy(model.state_dict())
        #     best_model_epoch = epoch
        #     best_val_loss = val_losses[-1]
        #     # print(f'Epoch {epoch}: Saving new best model based on val loss. Val Loss={best_val_loss:.2f}')

        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss={epoch_loss:.3f}, ' +
                  f'Perf={epoch_perf / N:.3f}, ' +
                  f'Marglik={-neg_marglik:.3}, ' +
                  f'Val Loss={val_loss:.3f}, ' +
                  f'Val Acc={val_acc:.3f}')
            print()
    
    return {
        'marglik': {'model_dict': best_marglik_model_dict,
                    'epoch': best_marglik_epoch},
        'valloss': {'model_dict': best_valloss_model_dict,
                    'epoch': best_valloss_epoch},
    }, losses, val_losses, neg_margliks
    

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
    args_dict = vars(args)
    
    if not args_dict['overwrite_config']:
        config_path = './configs/{}/{}_config.yaml'.format(
            args_dict['init_graph'], args_dict['model_type'].lower())
        print("Loading config from:", config_path)
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        args_dict.update(config['Default'])
        args_dict.update(config[args_dict['dataset'].capitalize()])

    args_dict = {k: None if str(v).lower() == 'none' else v
                 for k, v in args_dict.items()}

    print("Arguments:")
    for k, v in args_dict.items():
        print(f"\t{k}: {v}")
    
    
    # device
    print()
    if torch.cuda.is_available():
        gpu_num = unused_gpu()
        if gpu_num is not None:
            print(f"Using unused GPU {gpu_num}")
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device(f'cuda:{args_dict["gpu_num"]}')
            print(f'Using GPU ID: {args_dict["gpu_num"]}')
    else:
        device = torch.device('cpu')


    data = load_data(
        args_dict['dataset'], args_dict['n_data_rand_splits'])
    data.to(device)

    out_dir = osp.join(args_dict['base_out_dir'], args_dict['dataset'])
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # initialize adjacency matrix
    if args_dict['init_graph'] == 'original':
        adj = edge_index_to_adj(
            data.edge_index, num_nodes=data.x.size(0)).float()
        adj[adj > 1] = 1
        # import ipdb; ipdb.set_trace()
    elif args_dict['init_graph'] == 'knng':
        adj = get_knn_graph(data.x, args_dict['knng_k'])
    elif args_dict['init_graph'] is None:
        adj = torch.eye(data.x.size(0))
    elif osp.exists(args_dict['init_graph']):
        rst = torch.load(args_dict['init_graph'], weights_only=True)
        adj = edge_index_to_adj(
            rst['edge_index'].detach(), data.num_nodes).float()
        # adj = (adj + adj.t()).bool().float()
        # adj.fill_diagonal_(1)
    else:
        raise ValueError(
            f"Unknown initial graph structure: {args_dict['init_graph']}.",
            "Choose from 'original', 'knng', 'none'")

    h = homophily(data.edge_index, data.y)

    print(f"Original num edges: {data.edge_index.size(1)}, Homophily: {h:.3f}")
    print(f"Initial num edges: {adj.sum().int()}")

    if args_dict['stop_criterion'] == None:
        args_dict['stop_criterion'] = 'marglik' if args_dict['model_type'] in \
            ['stegcn', 'lorastegcn', 'stegraphsage'] else 'valloss'

    if args_dict['model_type'] in ['gcn', 'gat'] and args_dict['stop_criterion'] == 'marglik':
        warnings.warn(
            "Marglik should not be used as the stop criteria for GCN and GAT models")

    if 'ste' in args_dict['model_type'] and args_dict['stop_criterion'] == 'valloss':
        warnings.warn(
            "Validation loss should not be used as the stop criteria for STE models")
    
    # best_model_dict = None
    best_marglik = -np.inf
    best_valloss = np.inf
    
    best_model_dicts = dict()
    best_model_meta = dict()

    margliks = list()
    model_metadata = list()

    criterion = torch.nn.CrossEntropyLoss()

    learned_graphs_dir = get_learned_graphs_dir(
        out_dir, args_dict['init_graph'], args_dict['model_type'],
        args_dict['hessian_structure'], args_dict['subset_of_weights']
    )


    lr_adjs = [0.3, 0.4, 0.5, 0.6, 0.7]if args_dict['lr_adj'] is None \
        else [args_dict['lr_adj']]

    
    if args_dict['model_type'] in ['gcn', 'gat']:
        lr_adjs = [0.]  # no adj learning for GCN and GAT
    
    if args_dict['model_type'] in ['stegcn', 'lorastegcn', 'stegraphsage']:
        # grid search over STE thresholds if not specified
        ste_thresholds = np.arange(0.1, 1.0, 0.1) if args_dict['ste_thresh'] is None \
            else [args_dict['ste_thresh']]
    else:
        ste_thresholds = [0.]  # no ste threshold for other models
    
    lrs = [args_dict['lr']] if args_dict['lr'] is not None \
        else [0.01, 0.05, 0.1]
    weight_decays = [args_dict['weight_decay']] if args_dict['weight_decay'] is not None \
        else [5e-4, 5e-5, 5e-6]
    hidden_channels = [args_dict['hidden_channels']] if args_dict['hidden_channels'] is not None \
        else [16, 32, 64]
    dropouts = [args_dict['dropout_p']] if args_dict['dropout_p'] is not None \
        else [0.2, 0.3, 0.4, 0.5]
    lora_rs = [16, 32, 64] if (args_dict['lora_r'] is None and 'lora' in args_dict['model_type']) \
        else [args_dict['lora_r']]
    norms = ['batch', 'layer', 'none'] if args_dict['norm'] is None \
        else [args_dict['norm']]
    res_conn = [True, False] if args_dict['res'] is None \
        else [args_dict['res']]
    weight_decays_adj = [5e-3, 5e-4, 5e-5, 5e-6, 5e-7] if (args_dict['weight_decay_adj'] is None and
                                               'ste' in args_dict['model_type']) \
        else [args_dict['weight_decay_adj']]
    # import ipdb; ipdb.set_trace()
    def hyperparam_search():
        hyperparam_space = {
            'res': res_conn,
            'norm': norms,
            'lora_r': lora_rs,
            'lr': lrs,
            'weight_decay': weight_decays,
            'hidden_channel': hidden_channels,
            'dropout': dropouts,
            'lr_adj': lr_adjs,
            'thres': ste_thresholds,
            'weight_decay_adj': weight_decays_adj,
        }
        for k, v in hyperparam_space.items():
            if len(v) > 1:
                print(f"Hyperparam search space for {k}: {v}")
        combinations = product(*hyperparam_space.values())
        total = np.prod([len(v) for v in hyperparam_space.values()])

        for combination in tqdm(combinations, desc='Hyperparam search', total=total):
            yield dict(zip(hyperparam_space.keys(), combination))
    
    for hyperparams in hyperparam_search():
        res = hyperparams['res']
        norm = hyperparams['norm']
        lora_r = hyperparams['lora_r']
        lr = hyperparams['lr']
        weight_decay = hyperparams['weight_decay']
        hidden_channel = hyperparams['hidden_channel']
        dropout = hyperparams['dropout']
        lr_adj = hyperparams['lr_adj']
        thres = hyperparams['thres']
        weight_decay_adj = hyperparams['weight_decay_adj']

        print('-' * 10,
            f"lr={lr:.2f}, ",
            f"weight_decay={weight_decay}, ",
            f"hidden_channels={hidden_channel}, ",
            f"dropout={dropout:.2f}, ",
            f"lr_adj={lr_adj:.2f}, ",
            f"ste_thres={thres:.2f}, ",
            f"lora_r={str(lora_r)}, ",
            f"norm={norm}, ",
            f"res={res}",
            f"weight_decay_adj={weight_decay_adj}",
            '-' * 10)
        
        # Common arguments for all models
        common_args = {
            'in_channels': data.x.size(1),
            'hidden_channels': hidden_channel,
            'out_channels': data.y.max().item() + 1,
            'num_layers': args_dict['num_layers'],
            'dropout_p': dropout,
            'init_adj': adj,
            'norm': args_dict['norm'],
            'res': args_dict['res'],
            'X': data.x,
            'symmetric': args_dict['symmetric'],
        }
    
        n_splits = data.train_indices.size(1)
        
        stats = {'marglik': dict(),
                 'valloss': dict(),}

        def add_stats(stop_criteria, key, split_idx, value):
            if key not in stats[stop_criteria]:
                stats[stop_criteria][key] = [[] for _ in range(n_splits)]
            stats[stop_criteria][key][split_idx].append(value)

        for split_idx in range(n_splits):
            train_indices = data.train_indices[:, split_idx]
            train_labels = data.y[train_indices]
            val_indices = data.val_indices[:, split_idx]
            val_labels = data.y[val_indices]
            test_indices = data.test_indices[:, split_idx]
            test_labels = data.y[test_indices]

            # homophily
            global_h, avg_train_local_h, avg_test_local_h = avg_local_homophilies(
                adj, train_indices, test_indices, data.y)
            print("Homophily global, local train, local test:"
                    f"{global_h:.3f}, {avg_train_local_h:.3f}, {avg_test_local_h:.3f}")
            train_nodes_receptive_field = avg_receptive_field_degree(adj, train_indices, 2)
            test_nodes_receptive_field = avg_receptive_field_degree(adj, test_indices, 2)
            print(f"Train / test nodes avg receptive field degree:"
                  f"{train_nodes_receptive_field:.3f} "
                  f"{test_nodes_receptive_field:.3f}")
            
            # Model-specific arguments
            model_specific_args = {
                'stegcn': {
                    'threshold': thres,
                    'train_masked_update': args_dict['train_masked_update'],
                    'train_nodes': train_indices,
                    'sign_grad': args_dict['sign_grad'],
                },
                'stegraphsage': {
                    'threshold': thres,
                    'train_masked_update': args_dict['train_masked_update'],
                    'train_nodes': train_indices,
                    'num_sampled_nodes_per_hop': args_dict['num_sampled_nodes_per_hop'],
                    'sign_grad': args_dict['sign_grad'],
                },
                'graphsage': {
                    'num_sampled_nodes_per_hop': args_dict['num_sampled_nodes_per_hop'],
                },
                'clipgcn': {},
                'gcn': {},
                'lorastegcn': {
                    'r': lora_r,
                    'lora_alpha': args_dict['lora_alpha'],
                },
                'gat': {
                    'heads': args_dict['heads'],
                },
            }

            for repeat in range(args_dict['n_repeats']):
                print('-' * 20,
                      f"Split: {split_idx + 1} / {n_splits} (Repeat {repeat + 1})",
                      '-' * 20)
                # initialize model
                # import ipdb; ipdb.set_trace()
                model = model_classes[args_dict['model_type']](**common_args, **model_specific_args[args_dict['model_type']])
                
                model_dicts, losses, val_losses, neg_margliks = marglik_optimization(
                                                    model,
                                                    train_indices=train_indices,
                                                    train_labels=train_labels,
                                                    val_indices=val_indices,
                                                    val_labels=val_labels,
                                                    y=data.y,
                                                    stop_criterion=args_dict['stop_criterion'],
                                                    lr_adj=lr_adj,
                                                    lr=lr,
                                                    weight_decay=weight_decay,
                                                    n_epochs=args_dict['n_epochs'],
                                                    n_hypersteps=args_dict['n_hypersteps'],
                                                    n_epochs_burnin=args_dict['n_epochs_burnin'],
                                                    n_hyper_stop=args_dict['n_hyper_stop'],
                                                    marglik_frequency=args_dict['marglik_frequency'],
                                                    subset_of_weights=args_dict['subset_of_weights'],
                                                    hessian_structure=args_dict['hessian_structure'],
                                                    device=device,
                                                    learned_graphs_dir=learned_graphs_dir,
                                                    args_dict=args_dict)
                # import ipdb; ipdb.set_trace()

                train_dataset = TensorDataset(train_indices, train_labels)
                train_loader = DataLoader(train_dataset, batch_size=10000,
                                        shuffle=False)
                for stop_criteria, trained_model in model_dicts.items():
                    model = model_classes[args_dict['model_type']](
                        **common_args, **model_specific_args[args_dict['model_type']])
                    model.load_state_dict(trained_model['model_dict'])
                    model.to(device)
                    # import ipdb; ipdb.set_trace()
                    
                    lap = Laplace(
                        model,
                        "classification",
                        subset_of_weights=args_dict['subset_of_weights'],
                        hessian_structure=args_dict['hessian_structure'],)
                    lap.fit(train_loader)
                
                    marglik = lap.log_marginal_likelihood().item()

                    out_adj = lap.model.full_adj()
                    edge_index = adj_to_edge_index(out_adj)
                    h = homophily(edge_index, data.y.to(device))
                    n_edges = out_adj.sum().item()
                    print(f"Stop criterion: {stop_criteria}")
                    print(f"Final num edges: {n_edges}, Homophily: {h:.3f}")
                    print(f"Final Marglik: {marglik:.2f}")
                
                    # mean predictive
                    mean_val_loss, mean_val_acc = mean_eval(lap, val_indices, val_labels, criterion)
                    mean_test_loss, mean_test_acc = mean_eval(lap, test_indices, test_labels, criterion)
                    
                    # track test acc at mean
                    # stats[stop_criteria]['marglik'].append(marglik)
                    # stats[stop_criteria]['mean val loss'].append(mean_val_loss)
                    # stats[stop_criteria]['mean val acc'].append(mean_val_acc)

                    # stats[stop_criteria]['mean test loss'].append(mean_test_loss)
                    # stats[stop_criteria]['mean test acc'].append(mean_test_acc)

                    # stats[stop_criteria]['homophily'].append(h)
                    # stats[stop_criteria]['num edges'].append(edge_index.size(1))
                    # stats[stop_criteria]['best model epoch'].append(trained_model['epoch'])
                    # import ipdb; ipdb.set_trace()
                    add_stats(stop_criteria, 'marglik', split_idx, marglik)
                    add_stats(stop_criteria, 'mean val loss', split_idx, mean_val_loss)
                    add_stats(stop_criteria, 'mean val acc', split_idx, mean_val_acc)
                    
                    add_stats(stop_criteria, 'mean test loss', split_idx, mean_test_loss)
                    add_stats(stop_criteria, 'mean test acc', split_idx, mean_test_acc)
                    
                    add_stats(stop_criteria, 'homophily', split_idx, h)
                    add_stats(stop_criteria, 'num edges', split_idx, edge_index.size(1))
                    add_stats(stop_criteria, 'best model epoch', split_idx, trained_model['epoch'])

                    print(f'Marglik={marglik}, '
                          f'Mean Val Acc={mean_val_acc:.3f}, '
                          f'Mean Test Acc={mean_test_acc:.3f}, '
                          f'Best Model Epoch={trained_model["epoch"]}')

                    # save model
                    if stop_criteria == 'marglik' and marglik > best_marglik:
                        best_model_dicts['marglik'] = trained_model['model_dict']
                        best_marglik = marglik
                        best_model_meta['marglik'] = hyperparams
                    elif stop_criteria == 'valloss' and mean_val_loss < best_valloss:
                        best_model_dicts['valloss'] = trained_model['model_dict']
                        best_valloss = mean_val_loss
                        best_model_meta['valloss'] = hyperparams
                    
                    # if marglik > best_marglik:
                    #     best_marglik = marglik
                    #     best_marglik_model_dict = deepcopy(lap.model.state_dict())
                    
                    # if mean_val_loss < best_valloss:
                    #     best_valloss = mean_val_loss
                    #     best_valloss_model_dict = deepcopy(lap.model.state_dict())

                    if (args_dict['stop_criterion'] == 'marglik' and marglik > best_marglik) or \
                        (args_dict['stop_criterion'] == 'valloss' and mean_val_loss < best_valloss):
                        # best_marglik = marglik
                        # best_valloss = mean_val_loss
                        model_fn = osp.join(out_dir, f'{args_dict["init_graph"]}_{args_dict["model_type"]}_model.pt')

                        best_model_dict = deepcopy(lap.model.state_dict())
                        print('Saving best model to:', out_dir)
                        torch.save(lap.model.state_dict(),
                                model_fn)
                        torch.save({'edge_index': edge_index, 'marglik': marglik,
                                    'test_acc': mean_test_acc,
                                    'homophily': h, 'num_edges': edge_index.size(1)},
                                    osp.join(out_dir, f'{args_dict["dataset"]}_edge_index.pt'))

                    del lap
                    torch.cuda.empty_cache()
                    margliks.append(marglik)
        # meta = {k: (np.mean(v), np.std(v)) for k, v in stats.items()}
        meta = dict()
        for stop_criteria, _stats in stats.items():
            meta[stop_criteria] = {k: (np.mean(v), np.std(v))
                                   for k, v in _stats.items()}
        meta['params'] = hyperparams
        meta['params']['n_data_rand_splits'] = args_dict['n_data_rand_splits']
        meta['params']['n_repeats'] = args_dict['n_repeats']
        meta['params']['n_epochs_burnin'] = args_dict['n_epochs_burnin']
        meta['params']['n_epochs'] = args_dict['n_epochs']
        meta['params']['n_hypersteps'] = args_dict['n_hypersteps']
        meta['params']['n_hyper_stop'] = args_dict['n_hyper_stop']
        meta['params']['marglik_frequency'] = args_dict['marglik_frequency']
        meta['params']['symmetric'] = args_dict['symmetric']
        meta['params']['train_masked_update'] = args_dict['train_masked_update']
        meta['params']['num_sampled_nodes_per_hop'] = args_dict['num_sampled_nodes_per_hop']
        meta['params']['init_graphs'] = args_dict['init_graph']
        meta['params']['model_type'] = args_dict['model_type']
        meta['params']['optimizer'] = args_dict['optimizer']
        meta['params']['grad norm'] = args_dict['grad_norm']
        meta['params']['sign_grad'] = args_dict['sign_grad']
        meta['params']['momentum_adj'] = args_dict['momentum_adj']
        meta['params']['early_stop'] = args_dict['early_stop']
        model_metadata.append((meta, stats))
    
    
    def print_meta(meta):
        for k, v in meta.items():
            # if isinstance(v, dict):
            #     print(f"\t{k}:")
            #     for kk, vv in v.items():
            #         print(f"\t\t{kk}: {vv[0]:.3f} ({vv[1]:.3f})")
            if isinstance(v, tuple):
                print(f"\t{k}: {v[0]:.3f} ({v[1]:.3f})")
            else:
                print(f"\t{k}: {v}")
    
    print(f"Trained {args_dict['model_type']} with {args_dict['init_graph']} graph")

    # import ipdb; ipdb.set_trace()
    # best model by marglik
    idx_ml = np.argmax([x['marglik']['marglik'][0] for x, _ in model_metadata])
    # idx = np.argmax([x['marglik'][0] for x in model_metadata])
    print("Best model by Marglik: ")
    print_meta(model_metadata[idx_ml][0]['marglik'])
    print("Hyperparameters: ")
    print_meta(model_metadata[idx_ml][0]['params'])

    print('-' * 50)

    # best model by val loss
    idx_vl = np.argmin([x['valloss']['mean val loss'][0] for x, _ in model_metadata])
    print("Best model by Val Loss: ")
    print_meta(model_metadata[idx_vl][0]['valloss'])
    print("Hyperparameters: ")
    print_meta(model_metadata[idx_vl][0]['params'])

    # save metadata
    # rst_file = osp.join(out_dir, f'{args_dict['init_graph']}_{args_dict['model_type']}_rst.pt')
    rst_file = osp.join(out_dir, f'{args_dict["init_graph"]}_{args_dict["model_type"]}_rst.pkl')
    with open(rst_file, 'wb') as f:
        import pickle
        pickle.dump({
            'best_marglik': model_metadata[idx_ml],
            'best_valloss': model_metadata[idx_vl],
        }, f)
        # torch.save(model_metadata, f)

    # save final embeddings
    # model.load_state_dict(best_model_dict)
    # model.eval()
    # with torch.no_grad():
    #     emb = model(torch.arange(data.x.size(0)).to(device)).cpu()
    # emb_fn = osp.join(out_dir, f'{args_dict['init_graph']}_{args_dict['model_type']}_emb.pt')
    # torch.save(emb, emb_fn)
    
    # print(f"Saved results metadata to {rst_file}")
    # print(f"Saved final embeddings to {emb_fn}")
    # # print(f"Best model saved to {model_fn}")
    # print(f"Intermediate graphs saved to {learned_graphs_dir}")
    
    print('-' * 20, args_dict['dataset'], '-' * 20)
    keys = ['marglik', 'mean val loss', 'mean val acc', 'mean test loss',
            'mean test acc', 'homophily', 'num edges', 'best model epoch']
    print('marglik')
    for k in keys:
        v = model_metadata[idx_ml][0]['marglik'][k]
        print(f'{v[0]:.3f} ({v[1]:.3f})')
    print('valloss')
    for k in keys:
        v = model_metadata[idx_vl][0]['valloss'][k]
        print(f'{v[0]:.3f} ({v[1]:.3f})')
    
    print("Mean test acc:")
    print("Best marglik model:", [list(map(lambda x: round(x, 5), y))
                                  for y in model_metadata[idx_ml][1]['marglik']['mean test acc']])
    print("Best valloss model:", [list(map(lambda x: round(x, 5), y))
                                  for y in model_metadata[idx_vl][1]['valloss']['mean test acc']])          
    
    # avg_acc_margliks = [np.mean(y) for y in model_metadata[idx_ml][1]['marglik']['mean test acc']]
    # avg_acc_valloss = [np.mean(y) for y in model_metadata[idx_vl][1]['valloss']['mean test acc']]
    
    # log test_accs
    fn = osp.join(args_dict['base_out_dir'], f'{args_dict["dataset"]}_{args_dict["init_graph"]}_test_accs.pkl')
    if osp.exists(fn):
        with open(fn, 'rb') as f:
            test_accs = pickle.load(f)
    else:
        test_accs = dict()
    test_accs[args_dict['model_type']] = {
        'marglik_models': model_metadata[idx_ml][1]['marglik']['mean test acc'],
        'valloss_models': model_metadata[idx_vl][1]['valloss']['mean test acc'],
    }
    if args_dict['n_repeats'] == 10:
        with open(fn, 'wb') as f:
            pickle.dump(test_accs, f)
        print("Saved test accs to:", fn)

    # model.load_state_dict(best_model_dicts['valloss'])
    # pred = torch.nn.functional.softmax(model(torch.arange(data.x.size(0)).to(device)), dim=1)
    # torch.save(pred, '/home/anita/work/laplace-gnn-results/hetero_moon_pred.pt')
    # import ipdb; ipdb.set_trace()
    # torch.save(model.state_dict(), '/home/anita/work/laplace-gnn-results/hetero_moon_model_dict.pt')