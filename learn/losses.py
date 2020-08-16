import torch
import torch.nn as nn


def weighted_L1(x, y, bias=1., dim=(0, -1)):
    feat_l1 = nn.L1Loss(reduction="none")(x, y).sum(dim=dim, keepdim=True)
    feat_w = bias + feat_l1 / (feat_l1.sum() + 1e-12)
    return (feat_w * feat_l1).sum()


def prop_L1(x, y):
    L = nn.L1Loss(reduction="sum")(x, y)
    return 100 * L / y.sum()


def cos_loss(x, y):
    nume = (x * y).sum(axis=-1)
    x_norm = torch.norm(x, p=2, dim=-1)
    y_norm = torch.norm(y, p=2, dim=-1)
    y_norm = torch.where(y_norm <= torch.finfo(y.dtype).eps, torch.ones_like(y_norm), y_norm)
    denom = x_norm * y_norm
    out = - nume / denom
    return out.mean()


def euclidean_loss(x, y):
    out = torch.sqrt(((x - y) ** 2).sum(axis=1))
    return out.sum()



