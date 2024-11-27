import copy
import math
import torch.nn as nn
import torch

from typing import Union, Optional, Dict, Any, Callable
from torch_geometric.nn.resolver import activation_resolver

from .utils import normalize_adj, BinarizeSTE
from .layers import GCNConv, GATConv


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
                 **kwargs,
                 ):
        super(BaseGNN, self).__init__()

        self.X: torch.Tensor = X
        self.init_adj: torch.Tensor = init_adj
        self.init_adj.fill_diagonal_(1)  # add self-loop
        
        self.update_adj = update_adj
        self.adj = nn.Parameter(
            init_adj, requires_grad=update_adj)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout_p)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        # if norm == "layer":
        #     norm_layer = nn.LayerNorm(hidden_channels)
        # elif norm == "batch":
        #     norm_layer = nn.BatchNorm1d(hidden_channels)
        # elif norm is None:
        #     norm_layer = nn.Identity()
        # else:
        #     raise ValueError(f"Unknown normalization type: {norm}")
        # self.norms = nn.ModuleList([
        #     copy.deepcopy(norm_layer) for _ in range(num_layers - 1)])

        self.convs = nn.ModuleList()
        # self.res = nn.ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            # self.res.append(
            #     self.init_res_conn(in_channels, hidden_channels))
            in_channels = hidden_channels
        
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            # self.res.append(
            #     self.init_res_conn(in_channels, hidden_channels))
        
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
    
    # def init_res_conn(self, in_channels: int,
    #                   out_channels: int):
    #     if self.use_res_conn:
    #         return nn.Linear(in_channels, out_channels)
    #     return nn.Identity()
    
    def forward(self, x_indices: torch.Tensor):
        adj = self.forward_adj()
        x = self.X
        # for i, (conv, norm, res_conn) in enumerate(
        #     zip(self.convs, self.norms, self.res)):
        for i, conv in enumerate(self.convs):
            x = conv(adj, x)
            if i < self.num_layers - 1:
                # x = res_conn(x) + x
                # x = norm(x)
                x = self.act(x)
                x = self.dropout(x)        
        return x[x_indices]


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs,
                 ):
        super(GCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False, **kwargs)
        
    def forward_adj(self):
        return normalize_adj(self.adj)
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)
    

class STEGCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 threshold: float = 0.5,
                 **kwargs,
                 ):
        super(STEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True, **kwargs)
        self.threshold = threshold
    
    def forward_adj(self):
        adj = (self.adj + self.adj.T) / 2
        adj = BinarizeSTE.apply(adj, self.threshold)
        adj.fill_diagonal_(1)  # add self-loops
        return normalize_adj(adj)
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)


class LoRASTEGCN(BaseGNN):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 X: torch.tensor,
                 init_adj: torch.tensor,
                 r: int,
                 lora_alpha: float,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 threshold: float = 0.5,
                 **kwargs,
                 ):
        super(LoRASTEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True, **kwargs)
        self.threshold = threshold

        self.r = r
        self.lora_alpha = lora_alpha

        N = self.adj.size(0)
        self.adj_lora_A = nn.Parameter(self.adj.new_zeros((r, N)))
        self.adj_lora_B = nn.Parameter(self.adj.new_zeros((N, r)))
        self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.adj_lora_A, a=math.sqrt(5))
        nn.init.normal_(self.adj_lora_B)
    
    def forward_adj(self):
        adj = self.adj + (self.adj_lora_B @ self.adj_lora_A) * self.scaling
        adj = (adj + adj.T) / 2
        adj = BinarizeSTE.apply(adj, self.threshold)
        adj.fill_diagonal_(1)
        return normalize_adj(adj)

    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)
    

class GAT(BaseGNN):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 X,
                 init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs,
                 ):
        super(GAT, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False, **kwargs)

    def forward_adj(self):
        return self.adj
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        heads = kwargs.pop("heads", 1)
        concat = kwargs.pop("concat", True)

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        return GATConv(in_channels, out_channels,
                       heads=heads, concat=concat, **kwargs)


