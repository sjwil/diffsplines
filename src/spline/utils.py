import torch


def torch_binomial(k, i):
    # Binomial coefficient https://github.com/pytorch/pytorch/issues/47841
    return torch.exp(torch.lgamma(k + 1) -
                     torch.lgamma((k - i) + 1) - torch.lgamma(i + 1))
