import torch
import torch.nn as nn


class GraphSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 bias: bool = True):
        super(GraphSAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(
            in_channels * 2, out_channels, bias=bias)
                    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def mean_agg(self, adj, x):
        adj /= adj.sum(dim=1, keepdims=True)
        return adj @ x
    
    def forward(self, adj, x):
        x_neigh = self.mean_agg(adj, x)  # aggregate neighbors
        x = torch.cat([x, x_neigh], dim=-1)
        return self.lin(x)


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 bias: bool = True):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(
            in_channels, out_channels, bias=bias)
            
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, adj, x):
        return adj @ self.lin(x)


class GATConv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 heads: int,
                 negative_slope: float = 0.2,
                 concat: bool = True,
                 bias: bool = True,):
        """
        Parameters
        ----------
        in_channels: int
            Number of input features.
        out_channels: int
            Number of output features.
        heads: int
            Number of attention heads.
        negative_slope: float
            LeakyReLU negative slope.
        concat: bool
            If set to `False`, the multi-head outputs are averaged.
        bias: bool
            If set to `False`, the layer will not learn an additive
            bias.
        """
        super(GATConv, self).__init__()

        self.negative_slope = negative_slope
        self.heads = heads
        self.concat = concat

        self.in_channels = in_channels
        self.out_channels = out_channels
    
        self.lin = nn.Linear(
            in_channels, heads * out_channels, bias=False,)
        
        total_out_channels = out_channels * (heads if concat else 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)
                
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, adj, x):
        x_src = x_dst = self.lin(x).view(
            -1, self.heads, self.out_channels)
        
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)

        # sum over the neighbors
        adj = torch.stack([adj] * self.heads, dim=-1)
        alpha = adj * alpha_src.unsqueeze(0) + adj * alpha_dst.unsqueeze(1)
        alpha = nn.functional.leaky_relu(
            alpha, negative_slope=self.negative_slope)
        alpha = torch.where(adj > 0, torch.exp(alpha), 0.)
        alpha /= torch.sum(alpha, dim=1, keepdim=True)

        out = torch.einsum('bij,bjk->bkj', alpha, x_dst)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out += self.bias
        return out