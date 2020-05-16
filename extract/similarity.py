from multiprocessing import cpu_count
import numpy as np
from sklearn.metrics import pairwise_distances
from cafca.hdf.iter_utils import dts
from cafca.extract.utils import mp_foreach


def sim_graph(x_tuple, y_tuple, metric, reduce_func=None, n_jobs=None):
    dbx, dby = x_tuple[0], y_tuple[0]
    item_x, item_y = x_tuple[1], y_tuple[1]
    slice_x, slice_y = dts(item_x), dts(item_y)
    data_x, data_y = np.concatenate(dbx[item_x], axis=int(dbx.T)), np.concatenate(dby[item_y], axis=int(dby.T))
    data_x = data_x.T if dbx.T else data_x
    data_y = data_y.T if dby.T else data_y
    G = pairwise_distances(data_x, data_y, metric=metric, n_jobs=n_jobs)
    if reduce_func is not None:
        G = np.stack([reduce_func(block, axis=0)
                      for block in np.split(G, slice_x)[:-1]])
        G = np.hstack([reduce_func(block, axis=1)[:, None]
                       for block in np.split(G, slice_y, axis=1)[:-1]])
    return G


def sort_graph(G, mode, param):
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


def segments_sim(feat_x, item_x, feat_y, item_y,
                 param=1, mode="best", metric="cosine",
                 reduce_func=np.mean, batch_size=None,
                 return_graph=False, n_cores=cpu_count()):
    N, M = len(item_x), len(item_y)

    def as_batch(rg, n):
        i = 0
        while i < len(rg):
            yield rg[i:i + n]
            i += n
    if batch_size is None:
        # Todo 2 : make the batch size dependent of memory ?
        batch_size = min(np.ceil(np.sqrt(N * M)), 500)
        print("batching with size", batch_size)

    args = [((feat_x, item_x.iloc[batch_x]), (feat_y, item_y.iloc[batch_y]), metric, reduce_func)
            for batch_x in as_batch(list(range(N)), batch_size)
            for batch_y in as_batch(list(range(M)), batch_size)]

    # Todo : make the following to a queue + consumer structure so as to pack the args in a generator!
    G = mp_foreach(sim_graph, args, n_cores)
    # if N % batch_size > 0, then we have to concatenate 2 groups of results
    axes0 = np.r_[tuple(x.shape[0] for x in G)]
    split = np.where(axes0[:-1] != axes0[1:])[0]
    if len(split) > 0:
        split = split[0]
        # g2 = the last batches of x
        g1, g2 = G[:split + 1], G[split + 1:]
        G = np.vstack((np.hstack(g1).reshape(-1, M),
                       np.hstack(g2).reshape((-1, M))))
        G = np.ascontiguousarray(G)
    else:
        G = np.concatenate(G, axis=1).reshape(N, M)
    locs = sort_graph(G, mode, param)
    return (locs, G) if return_graph else locs
