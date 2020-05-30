import torch
import numpy as np
from cafca import DEVICE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython import get_ipython


# Array <==> Tensor ops :

def numcpu(y):
    if isinstance(y, torch.Tensor):
        if y.requires_grad:
            return y.detach().cpu().numpy()
        return y.cpu().numpy()
    else:  # tuples
        return tuple(numcpu(x) for x in y)


def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(DEVICE)
    elif isinstance(x, torch.Tensor):
        if x.device == DEVICE:
            return x.float()
        else:
            return x.float().to(DEVICE)
    return x


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
    rv = (torch.sin((1.0-val)*theta)/so).unsqueeze(-1)*low + (torch.sin(val*theta)/so).unsqueeze(-1) * high
    return rv


def is_notebook():
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        return True
    elif shell == 'TerminalInteractiveShell':
        return False
    else:
        return False
