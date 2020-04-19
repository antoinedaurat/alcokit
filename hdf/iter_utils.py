import numpy as np
from multiprocessing import cpu_count, Pool
from scipy.sparse import issparse


def dts(item):
    """
    Durations To Splits
    @param item: a pd.DataFrame with a "duration" column (typically the metadata of a db)
    @return: splits : the split-indices in a range that correspond to the input series of durations
    """
    splits = np.cumsum(item["duration"].values)
    return splits


def ssts(item):
    """
    Start-Stop To Slices
    @param item: a pd.DataFrame with a "start" and a "stop" column (typically the metadata of a db)
    @return: 1d array of slices for retrieving each element in `item`
    """
    arr = np.atleast_2d(item[['start', 'stop']].values)
    slices = np.apply_along_axis(lambda a: slice(a[0], a[1]), 1, arr)
    return slices


def irbol(item, batch_size=64, shuffle=False, drop_last=False):
    """
    In Random Batches Of Length
    """
    if shuffle:
        rg = np.arange(len(item))
        np.random.shuffle(rg)
        item = item.iloc[rg]
    slices = ssts(item)
    N = len(slices)
    n_batch = N // batch_size
    rem = N % batch_size
    if drop_last:
        return np.split(slices[:-rem], batch_size * (np.arange(n_batch) + 1))
    else:
        return np.split(slices, np.r_[batch_size * (np.arange(n_batch) + 1), N-rem])


def ibol(item, batch_size=64, drop_last=False):
    """
    In Batches Of Length
    """
    return irbol(item, batch_size, False, drop_last)


def irbod(item, batch_dur=1024, shuffle=False, drop_last=False):
    """
    In Random Batches Of Duration
    """
    if shuffle:
        rg = np.arange(len(item))
        np.random.shuffle(rg)
        item = item.iloc[rg]
    slices = ssts(item)
    dur = dts(item) % batch_dur
    batch_idx = np.where(dur[:-1] > dur[1:])[0]
    batches = np.split(slices, batch_idx + 1)
    return batches[:-1] if drop_last else batches


def ibod(item, batch_dur=1024, drop_last=False):
    """
    In Batches Of Duration
    """
    return irbod(item, batch_dur, False, drop_last)


def _mp_block_reduce(arr, splits_j, reduce_func):
    X = arr.A if issparse(arr) else arr
    X = np.split(X, splits_j, axis=1)
    return np.array([reduce_func(x) for x in X])


def reduce_2d(arr_2d, splits_i, splits_j, reduce_func=np.mean, n_jobs=cpu_count()):
    Ax = [arr_2d[xi:xj] for xi, xj in zip(np.r_[0, splits_i[:-1]], splits_i)]
    args = [(ax, splits_j[:-1], reduce_func) for ax in Ax]
    with Pool(n_jobs) as p:
        rows = p.starmap(_mp_block_reduce, args)
    return np.stack(rows)


def iterate(item, level, batch_type, shuffle=False, drop_last=True):
    if level == "frame":
        if batch_type == "dur":
            pass
        elif batch_type == "len":
            pass
        return None
    elif level == "segment":
        pass
    elif level == "file":
        pass
    else:
        pass
