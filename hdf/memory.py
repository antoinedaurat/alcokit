from psutil import virtual_memory
import numpy as np


def available():
    return virtual_memory().available


def row_size(x):
    item_size = np.dtype(x.dtype).itemsize
    row_length = np.prod(x.shape[1:])
    return item_size * row_length


def max_batch_size(x, take=.90):
    mem = available()
    per_row = row_size(x)
    return int(np.floor(take * mem / per_row))

