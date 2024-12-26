# from functools import lru_cache
import torch


def power_adj(adj, power):
    _adj = adj.clone()
    for _ in range(power - 1):
        _adj @= adj
    return _adj


def symmetrize_adj(adj):
    adj += adj.T
    adj[adj > 1] = 1
    return adj


def train_edge_mask(adj, n_layers, train_nodes,
                   frac_train_edges=0.5):
    """
    Get gradient mask for adjacency matrix.

    Parameters
    ----------
    receptive_field_adj : torch.Tensor
        receptive field adjacency matrix (i.e. `\sum_i^{n_layer} A ** i`)
    """

    train_nodes = train_nodes.cpu()
    adj = adj.clone().detach().cpu()
    
    adj_mask = torch.ones_like(adj)
    adj_mask[train_nodes[:, None], train_nodes] = 0
    return adj_mask

class BinarizeSTE(torch.autograd.Function):
    """
    Straight-through estimator with thresholding.
    """ 
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
        ctx.threshold = inputs[1]
        ctx.mask = inputs[2]

    @staticmethod
    def forward(input, threshold: float, mask=None):
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mask is not None:  # gradient masking on the adj matrix
            # import ipdb; ipdb.set_trace()
            grad_output *= ctx.mask
        # import ipdb; ipdb.set_trace()
        # symmetrize the gradient
        # grad_output = (grad_output + grad_output.T) / 2
        # grad_output = (grad_output + torch.einsum('ij->ji', grad_output)) / 2
        # import ipdb; ipdb.set_trace()
        return grad_output, None, None


class Clipping(torch.autograd.Function):
    """
    Clipping adjacency matrix.
    """
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def forward(input):
        return torch.clamp(input, 0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.clamp(grad_output, 0, 1)


def normalize_adj(adj):
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = rowsum.pow(-0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return (adj @ d_mat_inv_sqrt).T @ d_mat_inv_sqrt


def sample_neigh_adj(adj, k=None, seed=0):
    if k is None:
        return adj
    _, col_idx = adj.nonzero(as_tuple=True)    
    
    neighbors = torch.split(col_idx, adj.sum(dim=1).int().tolist())
    torch.manual_seed(seed)
    
    sampled_adj = torch.zeros_like(adj)
    for i, neigh in enumerate(neighbors):
        n_neigh = neigh.size(0)
        if n_neigh > 0:
            sampled_neigh = neigh[
                torch.randperm(n_neigh)
                [:min(n_neigh, k)]]
            sampled_adj[i, sampled_neigh] = 1
    return sampled_adj
    
