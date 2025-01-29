import copy

import torch.nn as nn
import torch

from typing import Union, Optional, Dict, Any, Callable

from torch_geometric.nn.resolver import activation_resolver


class BaseGNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 X: torch.Tensor,
                 init_adj: torch.Tensor,
                 dropout_p: float = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 update_adj: bool = False,
                 norm: Optional[str] = None,
                 res: bool = False,
                 symmetric: bool = False,
                 **kwargs,
                 ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input features.
        hidden_channels : int
            Number of hidden features.
        out_channels : int
            Number of output features.
        num_layers : int
            Number of layers.
        X : torch.Tensor
            Node feature matrix.
        init_adj : torch.Tensor
            Initial adjacency matrix.
        dropout_p : float
            Dropout probability.
        act : str or callable or None
            Activation function. Default is 'relu'.
        act_kwargs : dict or None
            Additional arguments for activation function.
        update_adj : bool
            Update adjacency matrix. Normal GNNs do not
            require update of the adjacency matrix.
        norm : str or None
            Normalization type (layer or batch). Default is None.
        res : bool
            Residual connection. Default is False.
        symmetric : bool
            Symmetrize adjacency matrix (i.e. treat as undirected).
            Default is False.
        """
        super(BaseGNN, self).__init__()

        self.X: torch.Tensor = X
        self.init_adj: torch.Tensor = init_adj
        
        self.update_adj = update_adj
        
        self.symmetric = symmetric
        if symmetric:  # symmetrize adjacency matrix
            adj = init_adj + torch.einsum('ij->ji', init_adj)
            adj[adj > 1] = 1
        else:
            adj = init_adj.clone()
        assert torch.all(torch.all((adj == 0) | (adj == 1)))

        self.adj = nn.Parameter(
            adj, requires_grad=update_adj)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout_p)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        
        if norm == "layer":
            norm_layer = nn.LayerNorm(hidden_channels)
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d(hidden_channels)
        elif norm in [None, "none"]:
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm}")
        self.norms = nn.ModuleList([
            copy.deepcopy(norm_layer) for _ in range(num_layers - 1)])

        self.convs = nn.ModuleList()
        self.res = nn.ModuleList()

        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if res:
                self.res.append(
                    nn.Linear(in_channels, hidden_channels))
            in_channels = hidden_channels

        for _ in range(num_layers - 2):
            import ipdb; ipdb.set_trace()
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if res:
                self.res.append(
                    nn.Linear(in_channels, hidden_channels))
        
        if out_channels is not None:
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
            
    def reset_parameters(self):
        with torch.no_grad():
            self.adj.copy_(self.init_adj)
        for conv in self.convs:
            conv.reset_parameters()
    
    def init_conv(self, in_channels: int,
                  out_channels: int, **kwargs):
        raise NotImplementedError
    
    def forward_adj(self):
        raise NotImplementedError
    
    def full_adj(self):
        return self.adj
        
    def forward(self, x_indices: torch.Tensor):
        adj = self.forward_adj()
        # if torch.isnan(adj).any() or torch.isinf(adj).any():
        #     raise ValueError("NaN or Inf in adjacency matrix.")
        x = self.X
        for i in range(self.num_layers - 1):
            if i < len(self.res):
                x = self.res[i](x) + self.convs[i](adj, x)
            else:
                x = self.convs[i](adj, x)
            # if torch.isnan(x).any() or torch.isinf(x).any():
            #     import ipdb; ipdb.set_trace()
            #     raise ValueError("NaN or Inf in hidden layer.")
            x = self.norms[i](x)
            # if torch.isnan(x).any() or torch.isinf(x).any():
            #     raise ValueError("NaN or Inf in normalization layer.")
            x = self.act(x)
            # x = x / x.norm(dim=-1, keepdim=True)
            # if torch.isnan(x).any() or torch.isinf(x).any():
            #     raise ValueError("NaN or Inf in activation layer.")
            x = self.dropout(x)
            # import ipdb; ipdb.set_trace()
        x = self.convs[-1](adj, x)
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     raise ValueError("NaN or Inf in output layer.")
        return x[x_indices]

