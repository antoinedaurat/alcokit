import numpy as np
from scipy.stats import rv_histogram
from scipy.ndimage import generic_filter
from alcokit.transform.spectrum import frame


# Low level functions
def categorical_split(X, n_bins=200, min_size=100, ignore_zeros=True):
    """
    fit an empirical distribution D to X and return the points in the domain of X
    where the sign of the derivative of the pdf of D is 0
    and where the value of this derivative is maximum.
    This corresponds to partionning X into n sub-domains where the pdf of the sub-domain_i
    is best approximated by a uniform distribution.
    """
    x_min, x_max = X.min(), X.max()
    if X.size < min_size or (x_max - x_min) == 0:
        return np.r_[x_min, x_min, x_max]
    hist = np.histogram(X, bins=n_bins)
    dist = rv_histogram(hist)
    xs = np.linspace(x_min, x_max, n_bins)
    uniform_cdf = xs * (1 / (xs[-1] - xs[0]))
    uniform_cdf -= uniform_cdf.min()
    dev = uniform_cdf - dist.cdf(xs)
    dev_sign = np.sign(dev)
    dev_abs = abs(dev)
    # X is already "almost" uniform
    if dev_abs.max() <= 1e-1:  # OR np.std(dev_abs) ?
        sp = (x_min, x_min, x_max)
    # X hasn't any zero crossings
    elif ignore_zeros or (np.all(dev_sign[1:-1] >= 0) or np.all(dev_sign[1:-1] <= 0)):
        sp = (x_min, xs[dev_abs.argmax()], x_max)
    # return Zero crossings and their respective maxes
    else:
        diffs = abs(np.diff(dev_sign[1:-1], prepend=0))
        zeros = np.where(diffs == 2)[0] + 1
        maxes = [x.argmax() + i for i, x in zip(np.r_[0, zeros], np.split(dev_abs, zeros))]
        sp = (x_min, *tuple(np.sort(xs[np.r_[zeros, maxes]])), x_max)
    return np.r_[sp]


def csplit_right(X, splits, **kwargs):
    new = categorical_split(X[X >= splits[-2]], **kwargs)
    return np.r_[splits[:-1], new[1:]]


def csplit_left(X, splits, **kwargs):
    new = categorical_split(X[X <= splits[1]], **kwargs)
    return np.r_[new[:-1], splits[1:]]


def k_csplits(X, k, side="right", **kwargs):
    splits = categorical_split(X, **kwargs)
    if side == "right":
        for n in range(max([0, k - 2])):
            splits = csplit_right(X, splits, **kwargs)
    else:
        for n in range(max([0, k - 2])):
            splits = csplit_left(X, splits, **kwargs)
    return splits


def csplit_along_axis(X, k, side="right", axis=None, **kwargs):
    if axis is None:
        return k_csplits(X.flat[:], k, side, **kwargs)
    return np.apply_along_axis(k_csplits, int(not axis), X, k, side, **kwargs)


def tag(X, csplits):
    tags = np.zeros_like(X, dtype=np.int32) - 1
    if len(csplits.shape) == 1:
        return np.sum(np.stack(tuple(X >= s for s in csplits[:-1])), axis=0, out=tags) - 1
    else:
        matching_dim = (np.r_[X.shape] == np.r_[csplits.shape]).nonzero()[0][0]
        func = lambda s: X >= (s if matching_dim == 1 else s[:, None])
        return np.sum(np.stack(tuple(func(s) for s in (csplits if matching_dim == 1 else csplits.T))),
                      axis=0, out=tags) - 1


class SupportTagger(object):
    def __init__(self, data, k=2, ignore_zeros=True, side="right"):
        self.ignore_zeros = ignore_zeros
        self.side = side
        self.splits = k_csplits(data.flat[:], k, side, ignore_zeros=ignore_zeros)

    def tag(self, X):
        return tag(X, self.splits)


def smooth_tags(tags, axis=None, kernel=None, window=2, agg=np.mean):
    if axis is not None and np.any(kernel is not None):
        raise ValueError("either axis or kernel has to be None")
    if axis is not None:
        framed = frame(tags, window, 1, "reflect", p_axis=axis, f_axis=axis)
        return agg(framed, axis=-1 if axis == 0 else 0)
    if np.any(kernel):
        return generic_filter(tags, agg, footprint=kernel)
    else:
        raise ValueError("both axis and kernel are None. Can not compute anything.")


def discrete_affinity(x, ref, glob_min, glob_max):
    """
    return the absolute affinity from x to ref
    as a proportion of (ref - glob_min) if x < ref
    or as a proportion of (glob_max - x) if x > ref.
    useful for mixing tagged elements in an array.
    """
    if x < glob_min or x > glob_max:
        return 0
    if (ref - x) > 0:
        return 1 - (ref - x) / (ref - glob_min)
    if (ref - x) < 0:
        return 1 - (x - ref) / (glob_max - ref)
    # ref == x
    return 1


def mix(X, tags, affinity_functions=(), normalize=True):
    K = len(affinity_functions)
    if K == 0:
        raise ValueError("the argument affinity_functions is empty")
    if X.shape != tags.shape:
        raise ValueError("X and tags must have the same shape")
    filtr = np.zeros_like(X, dtype=X.dtype)
    for func in affinity_functions:
        filtr += func(tags)
    return X * (filtr / (K if normalize else 1))
