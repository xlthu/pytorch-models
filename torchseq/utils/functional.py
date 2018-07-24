import torch

def softmax_mask(input, mask, dim=1, epsilon=1e-12):
    shift, _ = torch.max(input, dim, keepdim=True)
    shift = shift.expand_as(input)

    exp = torch.exp(input - shift) * mask

    norm = torch.sum(exp, dim, keepdim=True).expand_as(exp)
    softmax = exp / (norm + epsilon)

    return softmax