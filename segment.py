import numpy as np
from librosa.segment import recurrence_matrix
from librosa.util import localmax, softmask
from scipy.ndimage import convolve

from cafca.util import normalize, logistic_map


def checker(N):
    block = np.zeros((N * 2 + 1, N * 2 + 1), dtype=np.int32)
    for k in range(-N, N + 1):
        for l in range(-N, N + 1):
            block[k + N, l + N] = np.sign(k) * np.sign(l)
    return block / abs(block).sum()


def segment_from_recurrence_matrix(X,
                                   L=6,
                                   k=None,
                                   sym=True,
                                   bandwidth=1.,
                                   thresh=0.2,
                                   min_dur=4):
    R = recurrence_matrix(
        X, metric="cosine", mode="affinity",
        k=k, sym=sym, bandwidth=bandwidth, self=True)
    # intensify checker-board-like entries
    R_hat = convolve(R, checker(L), mode="constant")
    # extract them along the main diagonal
    dg = np.diag(R_hat, 0)
    mx = localmax(dg * (dg > thresh)).nonzero()[0]
    # filter out maxes less than min_dur frames away of the previous max
    mx = mx * (np.diff(mx, prepend=0) > min_dur)
    mx = mx[mx > 0]
    slices = np.split(np.arange(R.shape[0]), mx)
    return R, dg, slices


def block_reduce(R, slices):
    """
    sum blocks of indices in a matrix to reduce its dimensionality
    """
    n = len(slices)
    out = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i, n):
            r, s = slices[i], slices[j]
            # sum and normalize the block
            out[i, j] = R[r.min():r.max() + 1, s.min():s.max() + 1].sum() / (r.size * s.size)
            out[j, i] = out[i, j]
    return out


def stack_from_arrays(X, mode="max"):
    if mode == "max":
        max_len = max(x.shape[1] for x in X)
        return np.stack([np.pad(x, ((0, 0), (0, max_len - x.shape[1])))
                         for x in X])
    elif mode == "min":
        min_len = min(x.shape[1] for x in X)
        return np.stack([x[:, :min_len] for x in X])
    elif mode == "med":
        med_len = int(np.median(np.array(tuple(x.shape[1] for x in X))))
        stack = []
        for x in X:
            if x.shape[1] > med_len:
                stack += [x[:, :med_len]]
            elif x.shape[1] < med_len:
                stack += [np.pad(x, ((0, 0), (0, med_len - x.shape[1])))]
            else:
                stack += [x]
        return np.stack(stack)


def sequence_from_slices(X, slices):
    return X[:, np.concatenate(slices)]


def matrix2dict(A, thresh=0.):
    """
    extract nonzero column entries above thresh and pack them in a dict
    with their row indice as key.
    typically, `A` is an affinity/similarity matrix and we want to extract 'families' from it
    """
    M = A.copy()
    M[M < thresh] = 0
    kids = {}
    for i in range(M.shape[0]):
        nz = set(M[i].nonzero()[0])
        if nz:
            kids[i] = nz
    return kids


def dict2kerns(d, X, agg=np.median):
    return [agg(np.stack([X[i] for i in [k, *v]]), axis=0) for k, v in d.keys()]


def recurrent_softmask(X, window=None, hop_length=None, margin=1.,
                       aggregated=True, normalized=False):
    """
    accumulates the softmask zi := softmask(zi-1, xi * margin) * xi
    along the first dimension of X. X should be a 'stack' of examples
    (or typically nearest neighbors)
    """
    if aggregated:
        z = X[0].copy()
        for x in X[1:]:
            z = softmask(z, x * margin) * x
        return z if not normalized else z / z.sum()
    else:
        z = np.zeros_like(X)
        z[0] = X[0].copy()
        for i in range(1, X.shape[0]):
            xi = X[i]
            z[i] = softmask(z[i - 1], xi * margin) * xi
        return z if not normalized else z / z.sum(axis=1)


def mask_from_frame(x, strength=1):
    mask = np.log(normalize(x))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = logistic_map(mask, mask.mean(), strength=strength)
    if len(mask.shape) == 1:
        mask[:, None]
    return mask


def cluster_twins(M, thresh=2 / 3):
    """
    connects indices of M (a symetrical similarity matrix) where Mi,j > thresh and i ≠ j
    it is important that M be symetrical because once j is merged to i, j is "removed" from M
    and will not be a key in the returned dictionary (while `i` will)
    This constraint ensures that the returned dictionary contains less indices than M,
    thus offering the possibility to reduce M to a smaller number of components at a lesser cost.
    """
    rg = np.arange(M.shape[0])
    A = M.copy()
    A[rg, rg] = 0
    maxes = A > thresh
    meta = {}
    merged = np.zeros(A.shape[0], dtype=np.bool)
    while not np.all(merged):
        first_nz = (merged == 0).nonzero()[0][0]
        if np.any(maxes[first_nz]):
            meta[first_nz] = maxes[first_nz].nonzero()[0]
            merged[first_nz] = True
            merged[meta[first_nz]] = True
        else:
            meta[first_nz] = np.array([], dtype=np.int32)
            merged[first_nz] = True
    return meta


def connected_components(A, thresh=.5):
    visited = set()
    components = {}
    to_visit = list(range(A.shape[0]))
    cur_comp_idx = -1

    def visit(i, R, thresh, components, to_visit, visited, cur_comp_idx):
        new = (R[i] > thresh).nonzero()[0]
        if i in visited:
            return components, to_visit
        components.setdefault(cur_comp_idx, []).append(i)
        visited.add(i)
        for n in new:
            if n not in visited:
                components.setdefault(cur_comp_idx, []).append(n)
                to_visit.remove(n)
                visited.add(n)
                components, to_visit = visit(n, R, thresh, components, to_visit, visited, cur_comp_idx)
        return components, to_visit

    while to_visit:
        i = to_visit.pop(0)
        cur_comp_idx += 1
        components, to_visit = visit(i, A, thresh, components, to_visit, visited, cur_comp_idx)

    return {v[0]: v[1:] for v in components.values()}


def alphabet(X, family_func=cluster_twins, thresh=2 / 3, k=None, sym=True, bandwidth=1., agg=np.median):
    alphas = X.copy()
    R = recurrence_matrix(X, metric="cosine", mode="affinity",
                          k=k, sym=sym, bandwidth=bandwidth)
    d = family_func(R, thresh=thresh)
    while len(d) < R.shape[0]:
        alphas = dict2kerns(d, X, agg)
        R = recurrence_matrix(np.stack(alphas).T, k=k, sym=sym, bandwidth=bandwidth)
        d = family_func(R, thresh=thresh)
    return alphas
