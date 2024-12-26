import copy
import math
import torch.nn as nn
import torch

from typing import Union, Optional, Dict, Any, Callable
from torch_geometric.nn.resolver import activation_resolver

from .utils import normalize_adj, BinarizeSTE, \
    train_edge_mask, sample_neigh_adj
from .layers import GCNConv, GATConv, GraphSAGEConv


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
        
        self.update_adj = update_adj
        
        # symmetrize adjacency matrix
        adj = init_adj + torch.einsum('ij->ji', init_adj)
        adj[adj > 1] = 1
        assert torch.all(torch.logical_and(adj == 0, adj == 1))

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
        
    def forward(self, x_indices: torch.Tensor):
        adj = self.forward_adj()
        x = self.X
        for i in range(self.num_layers - 1):
            if i < len(self.res):
                x = self.res[i](x) + self.convs[i](adj, x)
            else:
                x = self.convs[i](adj, x)
            x = self.norms[i](x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.convs[-1](adj, x)
        return x[x_indices]


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(1)  # add self-loops
        super(GCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False, **kwargs)

    def forward_adj(self):
        return normalize_adj(self.adj)
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 num_sampled_nodes_per_hop: None | int,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(0)  # remove self-loops
        super(GraphSAGE, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False, **kwargs)

        self.num_sampled_nodes_per_hop = num_sampled_nodes_per_hop

    
    def forward_adj(self, full_adj=False):
        if not full_adj:
            return self.adj * sample_neigh_adj(self.adj, self.num_sampled_nodes_per_hop)
        return self.adj
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return GraphSAGEConv(in_channels, out_channels, **kwargs)


class STEGCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 threshold: float = 0.5,
                 frac_train_edges: Optional[float] = None,
                 train_nodes: Optional[torch.Tensor] = None,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(1)  # add self-loops
        super(STEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True, **kwargs)
        self.threshold = threshold
        self.frac_train_edges = frac_train_edges
        self.train_nodes = train_nodes
        if self.frac_train_edges is not None and self.train_nodes is None:
            raise ValueError("If 'frac_train_edges' is provided, "
                             "then 'train_nodes' must be provided.")
    
    def forward_adj(self):
        # adj = (self.adj + self.adj.T) / 2
        if self.frac_train_edges is not None:
            adj_mask = train_edge_mask(adj, self.num_layers, self.train_nodes,
                            frac_train_edges=self.frac_train_edges)
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    adj_mask.to(adj.device))
        else:
            adj = BinarizeSTE.apply(adj, self.threshold)
        adj.fill_diagonal_(1)  # add self-loops
        return normalize_adj(adj)
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)


class STEGraphSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 num_sampled_nodes_per_hop: None | int,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 threshold: float = 0.5,
                 frac_train_edges: Optional[float] = None,
                 train_nodes: Optional[torch.Tensor] = None,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(0)  # remove self-loops
        super(STEGraphSAGE, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True, **kwargs)
        self.threshold = threshold

        self.num_sampled_nodes_per_hop = num_sampled_nodes_per_hop
        
        self.frac_train_edges = frac_train_edges
        self.train_nodes = train_nodes
        if self.frac_train_edges is not None and self.train_nodes is None:
            raise ValueError("If 'frac_train_edges' is provided, "
                             "then 'train_nodes' must be provided.")
    
    def forward_adj(self, full_adj=False):
        # adj = (self.adj + self.adj.T) / 2
        adj = self.adj
        if self.frac_train_edges is not None:  # mask adj update
            grad_adj_mask = train_edge_mask(adj, self.num_layers,self.train_nodes,
                            frac_train_edges=self.frac_train_edges)
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    grad_adj_mask.to(adj.device))
        else:
            adj = BinarizeSTE.apply(adj, self.threshold)
        if not full_adj:  # sample neighbors
            adj *= sample_neigh_adj(adj, self.num_sampled_nodes_per_hop)
        return adj
    
    def init_conv(self, in_channels, out_channels, **kwargs):
        return GraphSAGEConv(in_channels, out_channels, **kwargs)


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


