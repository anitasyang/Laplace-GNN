import os
import argparse
from tqdm import tqdm
import os.path as osp

import pickle
import numpy as np
import torch
from scipy.sparse import csr_matrix

from GSL.data import *
from GSL.model import LDS, IDGL, SUBLIME, NodeFormer
from GSL.utils import accuracy

from utils import load_data, edge_index_to_adj, unused_gpu


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, nargs='+',
    choices=['cora', 'citeseer', 'pubmed',
             'actor', 'texas', 'wisconsin', 'cornell',
             'squirrel', 'chameleon'])
parser.add_argument(
    '--model_type', type=str,
    choices=['lds', 'idgl', 'sublime', 'nodeformer'])
parser.add_argument(
    '--n_repeats', type=int, default=1)
parser.add_argument(
    '--n_splits', type=int, default=10)
parser.add_argument(
    '--graph', type=str,
    choices=['original', 'knng'])
args = parser.parse_args()

gpu_num = unused_gpu() or 0
device = torch.device(f"cuda:{gpu_num}"
                      if torch.cuda.is_available() else "cpu")


def get_data(dataset):
    data = Dataset(root='/tmp/', name=dataset, use_mettack=False)

    # replace data
    data2 = load_data(dataset=dataset, n_rand_splits=10)
    data.features = data2.x.to(device)
    data.labels = data2.y.to(device)
    data.train_masks = [x.to(device) for x in data2.train_mask.t()]
    data.val_masks = [x.to(device) for x in data2.val_mask.t()]
    data.test_masks = [x.to(device) for x in data2.test_mask.t()]
    adj = edge_index_to_adj(data2.edge_index, data2.x.size(0))
    data.adj = adj.float().to(device)
    return data

# import ipdb; ipdb.set_trace()
# load a perturbed graph dataset
# data = Dataset(root='/tmp/', name='cora', use_mettack=True, ptb_rate=0.05)


metric = 'acc'


# the hyper-parameters are recorded in config
config_path = './configs/{}/{}_config.yaml'.format(
    args.graph, args.model_type.lower())
# config_path = './configs/knng/lds_hyper_search/lds_config_1.yaml'
# config_path = f'./configs/original/idgl_hyper_search'

if os.path.isdir(config_path):
    config_paths = [os.path.join(config_path, f)
                    for f in os.listdir(config_path)]
else:
    config_paths = [config_path]


model_type = {
    'lds': LDS,
    'idgl': IDGL,
    'sublime': SUBLIME,
    'nodeformer': NodeFormer,
}

model_specific_args = {
    'lds': {'params': None},
    'idgl': {'params': None},
    'sublime': {},
    'nodeformer': {'params': None},
}

rst = {}
mean_accs = {}
for dataset in args.dataset:
    for config_path in config_paths:
        print("using:", config_path)
        data = get_data(dataset)
        common_args = {
            'metric': accuracy,
            'config_path': config_path,
            'device': device,
            'num_features': data.num_feat,
            'num_classes': data.num_class,
            'dataset_name': dataset,
        }

        all_test_results = list()
        all_val_losses = list()
        for i in range(args.n_splits):
            test_results = list()
            val_losses = list()
            for j in tqdm(range(args.n_repeats)):
                print('-' * 20,
                    f'Split {i + 1} / {args.n_splits} (Repeat {j})',
                    '-' * 20)
                model = model_type[args.model_type](
                    **common_args, **model_specific_args[args.model_type])
                if args.graph == 'knng':
                    val_loss = model.fit(data, split_num=i, knng=True, k=3)
                else:
                    val_loss = model.fit(data, split_num=i)

                result = model.best_result
                test_results.append(result)
                val_losses.append(val_loss)
            all_test_results.append(test_results)
            all_val_losses.append(val_losses)

        all_test_results = np.array(all_test_results)
        test_mean_acc = np.mean(all_test_results)
        test_std_acc = np.std(all_test_results)
        all_val_losses = np.array(all_val_losses)
        val_mean_loss = np.mean(all_val_losses)
        val_std_loss = np.std(all_val_losses)
        
        split_mean_acc = np.mean(all_test_results, axis=1)
        mean_accs[(dataset, os.path.split(config_path)[1])] = all_test_results
        # import ipdb; ipdb.set_trace()
        # test_mean_acc = np.mean(test_results)
        # test_std_acc = np.std(test_results)
        # val_mean_loss = np.mean(val_losses)
        # val_std_loss = np.std(val_losses)

        print(f"[{dataset} {args.graph} {args.model_type}] "
            f"Mean accuracy: {test_mean_acc * 100:.4f} ({test_std_acc * 100:.4f}) "
            f"Mean val loss: {val_mean_loss:.4f}")
        
        rst[(dataset, os.path.split(config_path)[1])] = {
            'Test acc': (test_mean_acc, test_std_acc),
            'Val loss': (val_mean_loss, val_std_loss),}

# print('Mean accuracy:')
# for k, v in mean_accs.items():
#     print(f"{k[0]} {k[1]}: {[_v * 100 for _v in v]}")

import ipdb; ipdb.set_trace()

# log test_accs

# save
if args.n_repeats == 10:
    for (dataset, config), v in mean_accs.items():
        fn = osp.join('/home/anita/work/laplace-gnn-results', f'{dataset}_{args.graph}_test_accs.pkl')
        if osp.exists(fn):
            with open(fn, 'rb') as f:
                test_accs = pickle.load(f)
        else:
            test_accs = dict()
        # test_accs[args.model_type] = {
        #     'acc': [x * 100 for x in np.mean(v, axis=1)]
        # }
        test_accs[args.model_type] = {'acc': v}
        with open(fn, 'wb') as f:
            pickle.dump(test_accs, f)
        print("Saved test accs to:", fn)

import ipdb; ipdb.set_trace()

for k, v in rst.items():
    print(k)
    for _k, _v in v.items():
        if 'acc' in _k:
            print(f'\t{_k}: {_v[0] * 100:.4f} ({_v[1] * 100:.4f})')
        else:
            print(f'\t{_k}: {_v[0]:.4f} ({_v[1]:.4f})')
    # print(f'{k}: {v[0] * 100:.4f} ({v[1] * 100:.4f})')

import ipdb; ipdb.set_trace()
