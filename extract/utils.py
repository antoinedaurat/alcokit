import numpy as np
from multiprocessing import cpu_count, Pool
from scipy.sparse import issparse


def _mp_block_reduce(arr, splits_j, reduce_func, subst_zeros=None):
    X = arr.A if issparse(arr) else arr
    if subst_zeros is not None:
        X[X == 0] = subst_zeros(X)
    X = np.split(X, splits_j, axis=1)
    return np.array([reduce_func(x) for x in X])


def reduce_2d(arr_2d, splits_i, splits_j, reduce_func=np.mean, subst_zeros=None, n_jobs=cpu_count()):
    Ax = [arr_2d[xi:xj] for xi, xj in zip(np.r_[0, splits_i[:-1]], splits_i)]
    args = [(ax, splits_j[:-1], reduce_func, subst_zeros) for ax in Ax]
    with Pool(n_jobs) as p:
        rows = p.starmap(_mp_block_reduce, args)
    return np.stack(rows)
