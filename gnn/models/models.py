import copy
import math
import torch.nn as nn
import torch

from typing import Union, Optional, Dict, Any, Callable

from .utils import normalize_adj, BinarizeSTE, \
    train_adj_mask, sample_neigh_adj
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
                 train_masked_update: bool = False,
                 train_nodes: Optional[torch.Tensor] = None,
                 symmetric: bool = False,
                 sign_grad: bool = False,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(1)  # add self-loops
        super(STEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True,
            symmetric=symmetric, **kwargs)
        self.threshold = threshold
        self.sign_grad = sign_grad
        self.train_masked_update = train_masked_update
        self.train_nodes = train_nodes
        if self.train_masked_update and self.train_nodes is None:
            raise ValueError("'train_nodes' must be provided, "
                             "to use train_masked_update.")
        if self.train_masked_update:
            mask = train_adj_mask(
                self.adj.size(0), self.train_nodes)
            mask[mask == 0] = 0.1  # soft mask
            # import ipdb; ipdb.set_trace()
            self.register_buffer("grad_adj_mask", mask)
    
    def full_adj(self):
        adj = super().full_adj()
        return (adj > self.threshold).float()

    def forward_adj(self):
        adj = self.adj
        if self.symmetric:
            adj = (adj + adj.T) / 2
        if self.train_masked_update:
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    self.grad_adj_mask,
                                    self.sign_grad)
        else:
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    None,
                                    self.sign_grad)
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
                 train_masked_update: bool = False,
                 train_nodes: Optional[torch.Tensor] = None,
                 symmetric: bool = False,
                 sign_grad: bool = False,
                 **kwargs,
                 ):
        init_adj.fill_diagonal_(0)  # remove self-loops
        super(STEGraphSAGE, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=True,
            symmetric=symmetric, **kwargs)
        self.threshold = threshold
        self.sign_grad = sign_grad

        self.num_sampled_nodes_per_hop = num_sampled_nodes_per_hop
        
        self.train_masked_update = train_masked_update

        self.train_nodes = train_nodes
        if self.train_masked_update and self.train_nodes is None:
            raise ValueError("If 'train_nodes' must be provided, "
                             "to use train_masked_update.")
        if self.train_masked_update:
            self.register_buffer("grad_adj_mask", train_adj_mask(
                self.adj.size(0), self.train_nodes))
    
    def full_adj(self):
        adj = super().full_adj()
        return (adj > self.threshold).float()
    
    def forward_adj(self, full_adj=False):
        adj = self.adj
        if self.symmetric:
            adj = (adj + adj.T) / 2
        # import ipdb; ipdb.set_trace()
        if self.train_masked_update:  # mask adj update
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    self.grad_adj_mask,
                                    self.sign_grad)
            # import ipdb; ipdb.set_trace()
        else:
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    None,
                                    self.sign_grad)
        # if not full_adj:  # sample neighbors
        #     # import ipdb; ipdb.set_trace()
        #     adj *= sample_neigh_adj(adj, self.num_sampled_nodes_per_hop)
        #     # import ipdb; ipdb.set_trace()
        assert not torch.isnan(adj.data).any()
        assert not torch.isinf(adj.data).any()
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
        if self.symmetric:
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
        init_adj.fill_diagonal_(1)  # add self-loops
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


class AttSTEGCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, X, init_adj,
                 dropout_p = 0.5,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 threshold: float = 0.5,
                 train_masked_update: bool = False,
                 train_nodes: Optional[torch.Tensor] = None,
                 symmetric: bool = False,
                 d_k: int = 8,
                 **kwargs,
                 ):
        super(AttSTEGCN, self).__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, X, init_adj, dropout_p, act,
            act_kwargs, update_adj=False,
            symmetric=symmetric, **kwargs)
        self.threshold = threshold

        self.train_masked_update = train_masked_update
        self.train_nodes = train_nodes
        if self.train_masked_update and self.train_nodes is None:
            raise ValueError("If 'train_nodes' must be provided, "
                             "to use train_masked_update.")
        if self.train_masked_update:
            self.register_buffer("grad_adj_mask", train_adj_mask(
                self.adj.size(0), self.train_nodes))
        
        # self.adj_src = nn.Linear(in_channels, d_k, bias=False)
        # self.adj_dst = nn.Linear(in_channels, d_k, bias=False)
        self.adj_W = nn.Linear(in_channels, d_k, bias=False)
        # self.adj_lin = nn.Linear(2 * d_k, 1, bias=False)
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # self.adj_src.reset_parameters()
        # self.adj_dst.reset_parameters()
        self.adj_W.reset_parameters()
        # self.adj_lin.reset_parameters()
    
    def construct_adj(self):
        # src = self.adj_src(self.X)
        # dst = self.adj_dst(self.X)
        src = dst = self.adj_W(self.X)

        # alpha = torch.cat([src, dst], dim=-1)
        # alpha = self.adj_lin(alpha).squeeze(-1)
        # import ipdb; ipdb.set_trace()
        # alpha = torch.nn.functional.leaky_relu(alpha)

        score = torch.matmul(src, dst.transpose(0, 1)) / self.scale
        # score = torch.nn.functional.leaky_relu(score)
        # return nn.functional.softmax(score, dim=-1)
        # return torch.nn.functional.sigmoid(score)
        return torch.nn.functional.hardtanh(score, min_val=0, max_val=1)
    
    def forward_adj(self):
        adj = self.construct_adj()

        if self.symmetric:
            adj = (adj + adj.T) / 2
        
        if self.train_masked_update:
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    self.grad_adj_mask,)
        else:
            adj = BinarizeSTE.apply(adj, self.threshold,
                                    None,)
        
        adj.fill_diagonal_(1)
        return normalize_adj(adj)

    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)