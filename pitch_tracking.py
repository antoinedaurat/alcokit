import numpy as np
from cafca.util import b2m


"""

Graph of Local Centroid 

"""


# Low level functions

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


def square_ratios(x):
    """returns the matrix x / x.T """
    x = np.asarray(x)
    return np.tril(x[:, None] / x)


# Middle Level Functions

def almost_ints(x, gamma=1 / 8):
    """
    indicator function
    """
    return abs(np.rint(x) - x) <= (np.log10(x + 1) * gamma)


def ideal_ints(K, gamma=1/8):
    mask = almost_ints(K, gamma=gamma)
    return np.rint(K * mask) + np.eye(K.shape[0], dtype=np.int32)


def ratios(centers, gamma=1/8):
    K = square_ratios(centers)
    K_p = ideal_ints(K, gamma=gamma)
    return K, K_p


def harm_factors(K):
    """
    returns the element-wise harmonicity factors of a matrix a of ratios K
    """
    k_int = np.rint(K)
    where_int = k_int >= 2
    k_int[where_int] = np.log(k_int[where_int])
    out = np.zeros_like(K, dtype=np.float)
    out[where_int] = np.log(K[where_int]) / k_int[where_int]
    return out


# High Level functions

def harmonic_graph_ints(K, gamma=1/8):
    return almost_ints(K, gamma=gamma).astype(np.int32)


def harmonic_graph_dist(K, max_dist=.05):
    factors = harm_factors(K)
    diffs = abs(np.rint(factors) - factors)
    edges = ((diffs <= max_dist) & (diffs > 0)).astype(np.int32)
    return edges + np.eye(K.shape[0], dtype=np.int32)


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


# Interfaces

class HarmonicSpectrum(object):
    def __init__(self, S,
                 mask=None,
                 min_b=5, max_b=None,
                 max_dist=.05,
                 gamma=1/4):
        if len(S.shape) != 1:
            raise ValueError("`spectrum` must be a 1d-array")
        self.S = S
        if mask is None:
            print("mask is None")
            mask = S > S.mean()
        nz = bounded_nz(mask, min_b, max_b)
        self.chains = group_by_cc(nz)
        self.centers = np.array([center(chain, S) for chain in self.chains])
        self.n = self.centers.size
        self.K, self.K_prime = ratios(self.centers, gamma=gamma)
        self.harmonicity = harm_factors(self.K)
        self.graph = harmonic_graph_dist(self.K, max_dist=max_dist)
        self.roots_idx = locodis(self.graph)
        self.is_harmonic = [i for i in range(self.n)
                          if self.graph[i, :i].sum() > 1]
        self.has_hamonics = [i for i in range(self.n)
                            if self.graph[:, i].sum() > 1]
        self.residuals = [i for i in range(self.n)
                          if i not in np.r_[self.centers, self.is_harmonic, self.has_hamonics]]
        amp_w = np.array([S[chain].sum() for chain in self.chains])
        self.amp_w = amp_w / amp_w.sum()
        self.amp_h = [self.amp_w[self.graph[:, i].nonzero()[0]].sum() for i in range(self.n)]
        self.harm_w = self.graph.sum(axis=0) - 1

    def as_graph(self):

        def round3(x):
            return np.round(x, decimals=3)

        graph = []
        centers = round3(b2m(self.centers))
        amps = round3(self.amp_h)
        for r in self.roots_idx:
            harms = self.graph[:, r].nonzero()[0][1:]
            inharms = round3(self.harmonicity[harms, r])
            K = round3(self.K[harms, r])
            d = {
                "i_c": (r, centers[r]),
                "h_a": (self.harm_w[r], amps[r]),
                "harms": harms,
                "ratios": np.stack((K, inharms)).T,
                "comp": self.completeness(r)
            }
            graph += [d]
        return sorted(graph, key=lambda d: d['h_a'][0] * d['h_a'][1])[::-1]

    def completeness(self, idx):
        K = self.K_prime[:, idx]
        K = np.unique(K[K != 0].astype(np.int32)[1:])
        return (((K.size ** 2) / K.max()) / K.min()) if K.size else 0


