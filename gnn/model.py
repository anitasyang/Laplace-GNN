import math
import torch
from torch import nn
from utils import preprocess_adj, normalize_adj


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
        # import ipdb; ipdb.set_trace()
        return torch.clamp(grad_output, 0, 1)
        # return grad_output


class BaseGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, init_adj: torch.Tensor,
                 X: torch.Tensor, dropout_p: float = 0.5):
        super(BaseGCN, self).__init__()
        
        self.lin1 = nn.Linear(
            in_channels, hidden_channels)
        self.lin2 = nn.Linear(
            hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_p)

        self.adj = nn.Parameter(
            init_adj, requires_grad=True)
        self.X = X
    
    def forward_adj(self, *args) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, x_indices: torch.Tensor):        
        adj = self.forward_adj()

        adj_normalized = normalize_adj(adj)

        X = self.lin1(self.X)
        X = adj_normalized @ X
        X = nn.functional.relu(X)
        X = self.dropout(X)
        
        X = self.lin2(X)
        X = adj_normalized @ X

        return X[x_indices]


class GCN(BaseGCN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, init_adj, X,
                 dropout_p = 0.5):
        super(GCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            init_adj, X, dropout_p)
        
        init_adj.fill_diagonal_(1)
        self.adj: torch.Tensor = nn.Parameter(
            init_adj, requires_grad=False)
        
    def forward_adj(self):
        return self.adj


class STEGCN(BaseGCN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, init_adj: torch.Tensor, X: torch.Tensor,
                 dropout_p: float = 0.5, threshold: float = 0.5):
        super(STEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            init_adj, X, dropout_p)
        
        self.threshold = threshold
    
    def forward_adj(self):
        adj = (self.adj + self.adj.T) / 2
        adj = BinarizeSTE.apply(adj, self.threshold)
        adj.fill_diagonal_(1)  # add self-loops
        return adj


class LoRASTEGCN(BaseGCN):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            init_adj: torch.Tensor,
            X: torch.Tensor,
            r: int,
            lora_alpha: float,
            dropout_p: float = 0.5,
            threshold: float = 0.5):
        super(LoRASTEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            init_adj, X, dropout_p)
        
        self.threshold = threshold
        
        # Freeze the original adjacency matrix
        self.adj.requires_grad = False
        N = self.adj.shape[0]

        self.r = r
        self.lora_alpha = lora_alpha

        self.adj_lora_A = nn.Parameter(self.adj.new_zeros((r, N)))
        self.adj_lora_B = nn.Parameter(self.adj.new_zeros((N, r)))
        self.scaling = self.lora_alpha / self.r

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.adj_lora_A, a=math.sqrt(5))
        nn.init.normal_(self.adj_lora_B)
        
    def forward_adj(self):
        adj = self.adj + (self.adj_lora_B @ self.adj_lora_A) * self.scaling
        adj = (adj + adj.T) / 2
        # import ipdb; ipdb.set_trace()
        adj = BinarizeSTE.apply(adj, self.threshold)
        adj.fill_diagonal_(1)
        return adj


class ClipGCN(BaseGCN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, init_adj: torch.Tensor, X: torch.Tensor,
                 dropout_p: float = 0.5):
        super(ClipGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            init_adj, X, dropout_p)
    
    def forward_adj(self):
        adj = (self.adj + self.adj.T) / 2
        adj = Clipping.apply(adj)
        adj.fill_diagonal_(1)
        return adj
        # return torch.clamp(self.adj, 0, 1)

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