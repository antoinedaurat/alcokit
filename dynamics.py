import numpy as np
from scipy.stats import rv_histogram
from scipy.ndimage import generic_filter
from cafca.util import frame


# Low level functions
def uni_split_point(X, n_bins=200, min_size=100, ignore_zeros=False, uni_thresh=1e-1):
    """
    fit an empirical distribution D to X and return the points in the domain of X
    where the sign of the derivative of the pdf of D is 0
    and where the value of this derivative is maximum.
    This corresponds to partionning X into n sub-domains where the pdf of the sub-domain_i
    is best approximated by a uniform distribution.
    """
    x_min, x_max = X.min(), X.max()
    if X.size < min_size or (x_max - x_min) == 0:
        return x_min, x_max
    hist = np.histogram(X, bins=n_bins)
    dist = rv_histogram(hist)
    xs = np.linspace(x_min, x_max, n_bins)
    uniform_cdf = xs * (1 / (xs[-1] - xs[0]))
    uniform_cdf -= uniform_cdf.min()
    dev = uniform_cdf - dist.cdf(xs)
    dev_sign = np.sign(dev)
    dev_abs = abs(dev)
    # X is already "almost" uniform
    if dev_abs.max() <= uni_thresh:
        return x_min, x_max
    # X hasn't any zero crossings
    if ignore_zeros or (np.all(dev_sign[1:-1] >= 0) or np.all(dev_sign[1:-1] <= 0)):
        return x_min, xs[dev_abs.argmax()], x_max
    # return Zero crossings and their respective maxes
    diffs = abs(np.diff(dev_sign[1:-1], prepend=0))
    zeros = np.where(diffs == 2)[0] + 1
    maxes = [x.argmax() + i for i, x in zip(np.r_[0, zeros], np.split(dev_abs, zeros))]
    sp = (x_min, *tuple(xs[np.r_[zeros, maxes]].sort()), x_max)
    return sp


def split_right(X, splits, n_bins=200, min_size=100, ignore_zeros=False, uni_thresh=1e-1):
    new = uni_split_point(X[X >= splits[-2]],
                          n_bins=n_bins, min_size=min_size, ignore_zeros=ignore_zeros,
                          uni_thresh=uni_thresh)
    return (*splits[:-1], *new[1:])


def split_left(X, splits, n_bins=200, min_size=100, ignore_zeros=False, uni_thresh=1e-1):
    new = uni_split_point(X[X <= splits[1]],
                          n_bins=n_bins, min_size=min_size, ignore_zeros=ignore_zeros,
                          uni_thresh=uni_thresh)
    return (*new[:-1], *splits[1:])


def sub_domains_from_splits(X, splits, keepdims=False):
    # make sure we have min and max as first and last values 
    if splits[0] > X.min():
        splits = (X.min(), *splits)
    if splits[-1] < X.max():
        splits = (*splits, X.max())
    sp = splits
    if keepdims:
        masks = [(X >= sp[i]) & (X < sp[i + 1]) for i in range(len(sp) - 1)]
        return [X[mask] for mask in masks if np.any(mask)]
    splited = [X[(X >= sp[i]) & (X < sp[i + 1])] for i in range(len(sp) - 1)]
    return [x for x in splited if x.size > 0]


def get_priors(sub_domains):
    sizes = [s.size for s in sub_domains]
    return np.array(sizes) / sum(sizes)


def tag(X, splits):
    tags = np.zeros_like(X, dtype=np.int32) - 1
    return np.sum(np.stack(tuple(X >= s for s in splits)), axis=0, out=tags) - 1


# Higher level functions

def n_uniform_partitions(X, min_n=2, min_size=100):
    """
    tries to return at least min_n continuous intervalls in the domain of X that seem,
    from the data, to be uniform.
    If further splitting would result in a partition of size < min_size,
    the algorithm stops and only n < min_n sub-domains are returned.
    """
    N = X.size
    if min_n >= N:
        raise ValueError("the minimum number of partitions can not be >= to X's size.")
    splits = uni_split_point(X, n_bins=min_size, min_size=min_size)
    subs = sub_domains_from_splits(X, splits)
    priors = np.array([x.size / N for x in subs])
    n = priors.size
    if n >= min_n:
        return n, subs, priors, splits
    while n < min_n:
        mode = priors.argmax()
        splits = (*uni_split_point(subs[mode],
                                   n_bins=min_size, min_size=min_size,
                                   # we already found all splits where p(X = x) = 0,
                                   # and where x is a local mode.
                                   # Now, we only want to partition curves into linear segments, hence :
                                   ignore_zeros=True),
                  *splits)
        subs = sub_domains_from_splits(X, splits)
        priors = get_priors(subs)
        new_n = priors.size
        # make sure we are not trying to split hairs in 2
        if new_n == n:
            print(("Warning : Further splitting would make partitions smaller than min_size=%i." % min_size) +
                  ("Stopped partitioning at n = %i" % n))
            break
        n = new_n
    return n, subs, priors, splits


def nd_n_uniform_partitions(X, axis="glob", min_n=2, min_size=100):
    if axis == "glob":
        _, _, _, splits = n_uniform_partitions(X.flat, min_n=min_n, min_size=min_size)
        return tag(X, splits)
    if axis == "horz":
        return np.stack(tuple(tag(X[i], n_uniform_partitions(X[i], min_n=min_n, min_size=min_size))
                              for i in range(X.shape[0])))
    if axis == "vert":
        return np.hstack(tuple(tag(X.T[j], n_uniform_partitions(X.T[j], min_n=min_n, min_size=min_size))
                               for j in range(X.shape[1])))
    else:
        raise ValueError("axis must be one of 'glob', 'horz' or 'vert'")


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
    usefull for mixing tagged elements in an array.
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
