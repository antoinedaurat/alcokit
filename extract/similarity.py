import sklearn.neighbors as knn
from multiprocessing import cpu_count


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


def cos_sim_neighbors(X, Y=None, param=1, mode="best", n_jobs=cpu_count(), return_distances=True):
    if Y is None:
        Y = X
    nn = _get_estimator(param, mode, n_jobs)
    nn.fit(Y)
    return nn.kneighbors(X, return_distances)
