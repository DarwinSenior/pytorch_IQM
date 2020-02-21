import torch

Tensor = torch.Tensor

RBG2YIQ = torch.tensor([
    0.299, 0.587, 0.114,
    0.596, -0.274, -0.322,
    0.211, -0.523, 0.312 ]).view(3,3)


def rgb2yiq(x: Tensor):
    shape = x.shape
    return torch.bmm(x.view(-1, 3), RBG2YIQ).view(*x.size())
