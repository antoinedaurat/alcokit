import numpy as np


def bounded(X, min_x=None, max_x=None, keepdims=False, fill=0):
    if X.size == 0:
        return X
    if min_x is None:
        min_x = X.min()
    if max_x is None:
        max_x = X.max()
    if keepdims:
        X[X > max_x] = fill
        X[X < min_x] = fill
        return X
    return X[(X <= max_x) & (X >= min_x)]


def bounded_nz(x, min_i, max_i):
    if x.size == 0:
        return x
    nz = x.nonzero()[0]
    return bounded(nz, min_i, max_i, False)


def group_by_cc(x, eps=1., max_cons=None):
    """
    group xs by chains of consecutive points.
    Two points are consecutive iff their diff is less than or equal to eps.
    """
    if x.size <= 1:
        return x
    if np.all(np.diff(x) <= eps):
        return x
    a = np.sort(x)
    chains = [[a[0]]]
    for i in range(1, a.size):
        if (a[i-1] + eps >= a[i]) and \
                (True if not max_cons else len(chains[-1]) < max_cons):
            chains[-1] += [a[i]]
        else:
            chains += [[a[i]]]
    return chains


def center(indices, data):
    weights = data[indices] / (1e-9+data[indices].sum())
    return np.sum(indices * weights)


# Optimization

def sparse_mean(x, fill=np.array([]), axis=0):
    """
    helper function to take the mean only of the finite values > 0 of an array
    """
    xabs = abs(x)
    if len(x.shape) == 1:
        to_reduce = x[(xabs > 0) & (xabs < np.inf)]
        if np.any(to_reduce):
            return to_reduce.mean()
        else:
            return fill
    else:
        return np.apply_along_axis(sparse_mean, int(not axis), xabs)


def retune(centers, eps=.1, max_iter=25):
    if centers.size <= 1:
        return centers

    def _retune(centers, eps=eps):
        # ESTIMATION STEP
        K = square_ratios(centers)
        ideal_ratios = (np.rint(K) * almost_ints(K, .1))
        ideal_ratios[ideal_ratios <= 1.] = np.inf
        # what the spectrum would look like, if partial_i were readjusted to integer ratios of partials_j
        should_horz = centers * ideal_ratios
        should_horz = np.asarray([sparse_mean(should_horz[i], centers[i]) \
                                for i in range(centers.size)])
        # what the fundamental SHOULD have as a spectrum if centers[i] were the true fundamental:
        should_vert = centers[:, None] / ideal_ratios
        should_vert = np.asarray([sparse_mean(should_vert[:, j], centers[j]) \
                                for j in range(centers.size)])
        # MAXIMIZATION STEP
        return np.stack((should_horz, should_vert)).mean(axis=0)
    for _ in range(max_iter):
        centers = _retune(centers, eps=eps)
    return centers


def square_ratios(x):
    """returns the matrix x / x.T """
    x = np.asarray(x)
    return np.tril(x[:, None] / x)


def almost_ints(x, gamma=1 / 8):
    """
    indicator function
    """
    return abs(np.rint(x) - x) <= (np.log10(x + 1) * gamma)


def ideal_ints(centers, gamma=1/8):
    K = square_ratios(centers)
    mask = almost_ints(K, gamma=gamma)
    return np.rint(K * mask) + np.eye(centers.size, dtype=np.int32)


def ratios(centers, gamma=1/8):
    K = square_ratios(centers)
    K_p = ideal_ints(K, gamma=gamma)
    return K, K_p


def ideal_centers(centers, ideal_ratios):
    return centers * np.tril(ideal_ratios)


def harm_factors(K):
    """
    returns the element-wise harmonicity factors of a matrix of ratios
    """
    k_int = np.rint(K)
    where_int = k_int >= 2
    k_int[where_int] = np.log(k_int[where_int])
    out = np.zeros_like(K, dtype=np.float)
    out[where_int] = np.log(K[where_int]) / k_int[where_int]
    return out


def harmonic_graph(centers, gamma=1/8):
    K = square_ratios(centers)
    return almost_ints(K, gamma=gamma).astype(np.int32)


def locodis(G):
    """
    find the roots of the harmonic graph G.
    @arg `G`: a graph where an edge (i, j) means Gi % Gj = 0 + eps ≈ 0
    returns the indices of the vertices which are the lowest common divisors
    to the rest of the series.
    """
    n = G.shape[0]
    D = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            # avoid having ands at higher bins than i and j (i.e. where bin_k = bin_i * bin_j)
            # i -> j IFF bins[:j] have edges in G
            ands = np.sum(np.logical_and(G[:j+1, i], G[:j+1, j]))
            D[i, j] = ands
    sums = D.sum(axis=1)
    candidates = list((sums > 0).nonzero()[0])
    for i, c in enumerate(candidates[:-1]):
        intersect = set(D[i].nonzero()[0]) & set(candidates[i+1:])
        if len(intersect) > 0:
            for x in intersect:
                candidates.remove(x)
    return np.asarray(candidates, dtype=np.int32)


class HarmonicSpectrum(object):
    def __init__(self, S,
                 mask=None,
                 max_r=100,
                 min_b=5, max_b=None,
                 gamma=1/5):
        if len(S.shape) != 1:
            raise ValueError("`spectrum` must be a 1d-array")
        self.S = S
        nz = bounded_nz(mask or S > S.mean() + 1, min_b, max_b)
        self.chains = group_by_cc(nz)
        self.centers = np.array([center(chain, S) for chain in self.chains])
        self.K, self.K_prime = ratios(self.centers, gamma=gamma)
        self.ideal_centers = self.centers * self.K_prime
        self.inharmonicity = harm_factors(self.K)
        self.graph = ((self.K_prime != 0) & (self.K_prime <= max_r)).astype(np.int32)
        self.roots_idx = locodis(self.graph)
        self.harmonics = [i for i in range(self.centers.size) if self.graph[i, :i].sum() > 1]
        self.residuals = [i for i in range(self.centers.size)
                          if self.graph[i, :i].sum() == 1 and i not in self.centers]
        amp_w = np.array([S[chain].sum() for chain in self.chains])
        self.amp_w = amp_w / amp_w.sum()
        self.harm_w = self.graph.sum(axis=0) - 1

    def as_graph(self):
        graph = []
        centers = np.round(b2m(self.centers), decimals=3)
        amps = np.round(self.amp_w, decimals=3)
        for r in self.roots_idx:
            harms = self.graph[:, r].nonzero()[0][1:]
            if np.any(harms):
                K = np.round(self.K[harms, r], decimals=3)
                res = np.round(self.K[self.roots_idx, r], decimals=3)
                d = {
                    "i_c": (r, centers[r]),
                    "h_a": (self.harm_w[r], amps[r]),
                    "harms": harms,
                    "ratios": K,
                    "residual_ratios": res,
                    "comp": self.completeness(r)
                }
                graph += [d]
        return sorted(graph, key=lambda d: d['comp'])[::-1]

    def completeness(self, idx):
        K = self.K_prime[:, idx]
        K = np.unique(K[K != 0].astype(np.int32)[1:])
        return (((K.size ** 2) / K.max()) / K.min()) if K.size else 0






# def retune_up(centers, r=40):
#     mask = np.tril(ideal_ratios(centers, r=r))
#     return sparse_mean(centers * mask, axis=0)
#
#
# def retune_down(centers, r=40):
#     mask = np.tril(ideal_ratios(centers, r=r))
#     # avoid division by 0
#     mask[mask == 0] = -1
#     res = centers / mask.T
#     res[res < 0] = 0
#     return sparse_mean(res, axis=0)



# def f0s(S, eps=.1, max_iter=25, min_a=0., min_b=1, max_b=None):
#     Sabs = abs(S)
#     chains = [group_by_cc(bounded_nz(Sabs[:, t], min_a, None))
#               for t in range(Sabs.shape[1])]
#     centers = [np.array([center(c, Sabs[:, t]) for c in chain])
#                for t, chain in enumerate(chains)]
#     centers = [retune(c, eps=eps, max_iter=max_iter)
#                for c in centers]
#     edges = [almost_int(floatp_ratios(c), eps)
#              if c.size else np.array([]) for c in centers]
#     locos_idx = [locodis(E) for E in edges]
#     locos_weight = [(E.sum(axis=0) / E.sum())[idx] \
#                     if E.size else np.array([0], dtype=np.int32) \
#                     for idx, E in zip(locos_idx, edges)]
#     locos = [list(zip(c[idx], w)) for c, idx, w in zip(centers, locos_idx, locos_weight)]
#     return locos

#



