import numpy as np


def split_length(x, min_n, max_n):
    mod = min(range(min_n, max_n + 1), key=lambda m: x % m if min_n <= x % m <= max_n else (x % m) * max_n)
    return [mod for _ in range(x // mod)] + ([x % mod] if x % mod >= 1 else [])


def constrain_lengths(lengths, min_d, max_d):
    """
    splits or joins the duration returned by an iterator to constrain them between min_d and max_d
    @param lengths:
    @param min_d:
    @param max_d:
    @return:
    """
    lengths = iter(lengths) if not getattr(lengths, "__next__", False) else lengths
    while True:
        try:
            d = next(lengths)
        except StopIteration:
            break
        if min_d <= d <= max_d:
            yield d

        elif max_d < d:
            for di in split_length(d, min_d, max_d):
                yield di

        elif d < min_d:
            while d < min_d:
                try:
                    nxt = next(lengths)
                except StopIteration:
                    break
                d += nxt
            if d <= max_d:
                yield d
            else:
                for di in split_length(d, min_d, max_d):
                    yield di


def nearest_multiple(x, m):
    """
    quantize a sequence x to a given modulo m
    """
    return (m * np.maximum(np.rint(x / m), 1)).astype(np.int)



