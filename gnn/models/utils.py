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