import torch
from torch import nn


class BinarizeSTE(torch.autograd.Function):
    """
    Straight-through estimator with thresholding.
    """
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
        ctx.threshold = inputs[1]

    @staticmethod
    def forward(input, threshold: float):
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, init_adj: torch.Tensor, X: torch.Tensor,
                 threshold: float = 0.5, update_adj: bool = True,
                 dropout_p: float = 0.5):
        super(SimpleGCN, self).__init__()
        self.lin1 = nn.Linear(
            in_channels, hidden_channels)
        self.lin2 = nn.Linear(
            hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_p)

        self.adj = nn.Parameter(init_adj, requires_grad=update_adj)
        self.X = X
        self.threshold = threshold
        self.update_adj = update_adj


    def forward(self, x_indices: torch.Tensor):
        if self.update_adj:
            adj = (self.adj + self.adj.T) / 2  # Symmetric
            adj = BinarizeSTE.apply(self.adj, self.threshold)
            adj.fill_diagonal_(1)
        else:
            adj = self.adj

        deg_A = torch.diag(adj.sum(axis=1).pow(-0.5))
        aug_A = torch.mm(torch.mm(deg_A, adj), deg_A)

        X = self.dropout(self.X)
        X = self.lin1(X)
        X = aug_A @ X
        X = nn.functional.relu(X)

        X = self.dropout(X)
        X = self.lin2(X)
        X = aug_A @ X
        # X = torch.nn.functional.relu(X)

        return X[x_indices]


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.lin2(x)
        return x