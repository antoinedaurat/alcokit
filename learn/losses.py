import torch
import torch.nn as nn


def weighted_L1(x, y, bias=1.):
    feat_l1 = nn.L1Loss(reduction="none")(x, y).sum(dim=0)
    feat_w = bias + feat_l1 / feat_l1.sum()
    return (feat_w * feat_l1).sum()


def dot_loss(x, y):
    return - (x * y).sum(axis=0).sum()


def cos_loss(x, y):
    nume = (x * y).sum(axis=1)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    y_norm = torch.where(y_norm <= torch.finfo(y.dtype).eps, torch.ones_like(y_norm), y_norm)
    denom = x_norm * y_norm
    out =  - nume / denom
    return out.mean()


def euclidean_loss(x, y):
    out = torch.sqrt(((x - y) ** 2).sum(axis=1))
    return out.sum()


def L1_log(x, y):
    L =  nn.L1Loss(reduction="none")(x, y)
    return (1 + L).log().sum() + L.sum()


def loged_L1(x, y):
    return nn.L1Loss(reduction="sum")((x+1).log(), (y+1).log())


def exped_L1(x, y):
    beta = 1 + torch.exp(- torch.arange(x.size(1)).float().to(x))
    return (nn.L1Loss(reduction="none")(x, y) * beta).sum()


def bce(x, y):
    pass

