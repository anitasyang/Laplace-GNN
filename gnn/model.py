import torch


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


# class BaseGCN(torch.nn.Module):


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, init_adj: torch.Tensor, X: torch.Tensor,
                 threshold: float = 0.5, update_adj: bool = True):
        super(SimpleGCN, self).__init__()
        self.lin1 = torch.nn.Linear(
            in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(
            hidden_channels, out_channels)

        self.adj = torch.nn.Parameter(init_adj, requires_grad=update_adj)
        self.X = X
        self.threshold = threshold


    def forward(self, x_indices: torch.Tensor):
        adj = (self.adj + self.adj.T) / 2  # Symmetric
        adj = BinarizeSTE.apply(self.adj, self.threshold)
        adj.fill_diagonal_(1)

        deg_A = torch.diag(adj.sum(axis=1).pow(-0.5))
        aug_A = torch.mm(torch.mm(deg_A, adj), deg_A)

        X = self.lin1(self.X)
        X = aug_A @ X
        X = torch.nn.functional.relu(X)

        X = self.lin2(X)
        X = aug_A @ X
        X = torch.nn.functional.relu(X)
        return X[x_indices]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.relu(self.lin2(x))
        return x