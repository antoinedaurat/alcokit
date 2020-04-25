import sklearn.neighbors as knn
from multiprocessing import cpu_count
from cafca.extract.utils import reduce_2d
import numpy as np


def _get_estimator(param=1, mode="best", n_jobs=cpu_count()):
    if mode == "best":
        param = int(param)
        return knn.KNeighborsTransformer(metric="cosine",
                                         n_neighbors=param,
                                         n_jobs=n_jobs)
    elif mode == "radius":
        return knn.RadiusNeighborsTransformer(metric="cosine",
                                              radius=param,
                                              n_jobs=n_jobs)
    else:
        raise ValueError("value for `mode` argument = '{}' not understood.".format(mode) +
                         "`mode` should be one of 'best' or 'radius'")


def cos_sim_graph(X, Y=None, param=1, mode="best", n_jobs=cpu_count()):
    if Y is None:
        Y = X
    nn = _get_estimator(param, mode, n_jobs)
    nn.fit(Y)
    return nn.transform(X)


def segments_sim(X, splits_x, Y=None, splits_y=None, param=1, mode="best", n_jobs=cpu_count()):
    """
    # TODO : handle sparse-reduce, return distances
    @param X:
    @param splits_x:
    @param Y:
    @param splits_y:
    @param param:
    @param mode:
    @param n_jobs:
    @return: a list of arrays where each array_i corresponds to the neighbors (indices in the array splits_y)
            of the segment splits_x_i-1:splits_x_i
    """
    G = cos_sim_graph(X, Y, param, mode, n_jobs)
    G = reduce_2d(G, splits_x, splits_y, np.mean, subst_zeros=None, n_jobs=n_jobs)
    locs = None
    if mode == "best":
        idx = np.argsort(G, axis=1)[:, :param]
        locs = [neighbors for neighbors in idx]
    elif mode == "radius":
        locs = []
        for row in G:
            idx = np.argsort(row)
            less_than = idx[row[idx] <= param]
            locs += [less_than]
    return locs


def cos_sim_neighbors(X, Y=None, param=1, mode="best", n_jobs=cpu_count(), return_distances=True):
    if Y is None:
        Y = X
    nn = _get_estimator(param, mode, n_jobs)
    nn.fit(Y)
    return nn.kneighbors(X, return_distances)
