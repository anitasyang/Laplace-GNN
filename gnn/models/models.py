import copy
import math
import torch.nn as nn
import torch

from typing import Union, Optional, Dict, Any, Callable

from .utils import normalize_adj, BinarizeSTE, \
    train_edge_mask, sample_neigh_adj
from .layers import GCNConv, GATConv, GraphSAGEConv
from .base_gnn import BaseGNN


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 symmetric: bool = False,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(1)  # add self-loops
        super(GCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False,
            symmetric=symmetric, **kwargs)

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
                 symmetric: bool = False,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(0)  # remove self-loops
        super(GraphSAGE, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False,
            symmetric=symmetric, **kwargs)

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
                 symmetric: bool = False,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(1)  # add self-loops
        super(STEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True,
            symmetric=symmetric, **kwargs)
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
                 symmetric: bool = False,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(0)  # remove self-loops
        super(STEGraphSAGE, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True,
            symmetric=symmetric, **kwargs)
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
                 symmetric: bool = False,
                 **kwargs,
                 ):
        super(LoRASTEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True,
            symmetric=symmetric, **kwargs)
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
                 symmetric: bool = False,
                 **kwargs,
                 ):
        super(GAT, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False,
            symmetric=symmetric, **kwargs)

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


