import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_pca(z, tags=None):
    encoded = PCA(n_components=2).fit_transform(z)
    if tags is None:
        tags = [str(i) for i in range(len(z))]
    plt.figure(figsize=(12, 12))
    plt.scatter(encoded[:, 0], encoded[:, 1])
    for n, txt in enumerate(tags):
        plt.text(encoded[n, 0], encoded[n, 1], txt)
    plt.show()


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    eps = torch.finfo(low.dtype).eps
    theta = torch.acos((low_norm*high_norm).sum(-1))
    sign = torch.sign(2 * torch.sign(0.5 * np.pi - theta % np.pi) + 1)
    theta += sign * eps
    so = torch.sin(theta)
    if len(theta.shape) > 0:
        val = val.unsqueeze(-1)
    rv = (torch.sin((1.0-val)*theta)/so).unsqueeze(-1)*low + (torch.sin(val*theta)/so).unsqueeze(-1) * high
    return rv


def slerp_np(val, low, high):
    low_norm = low / np.linalg.norm(low, axis=-1, keepdims=True)
    high_norm = high / np.linalg.norm(high, axis=-1, keepdims=True)
    eps = np.finfo(low.dtype).eps
    theta = np.arccos((low_norm*high_norm).sum(-1))
    sign = np.sign(2 * np.sign(0.5 * np.pi - theta % np.pi) + 1)
    theta += sign * eps
    so = np.sin(theta)
    if theta.shape != ():
        val = np.expand_dims(val, axis=-1)
    rv = np.expand_dims(np.sin((1.0-val)*theta)/so, axis=-1)*low + \
        np.expand_dims(np.sin(val*theta)/so, axis=-1) * high
    return rv