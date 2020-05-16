import numpy as np


def dts(item):
    """
    Durations To Splits
    @param item: a pd.DataFrame with a "duration" column (typically the metadata of a db)
    @return: splits : the corresponding split-indices
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


def batch_of(feature, m_frame, batch_size, shuffle=True):
    n = len(m_frame)
    n_batches = n // batch_size
    gen = feature.gen_item(m_frame if not shuffle else m_frame.sample(frac=1.))
    next_batch = [next(gen) for _ in range(batch_size)]
    for i in range(n_batches-1):
        yield next_batch
        next_batch = [next(gen) for _ in range(batch_size)]
    yield next_batch
    yield [next(gen) for _ in range(n % batch_size)]


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