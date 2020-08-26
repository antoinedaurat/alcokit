import numpy as np
import scipy.stats as stats
import librosa
from alcokit.util import normalize
from alcokit.transform.spectrum import frame, running_agg
from librosa.segment import recurrence_matrix
from librosa.util import localmax
from scipy.ndimage import convolve


def from_recurrence_matrix(X,
                           L=6,
                           k=None,
                           sym=True,
                           bandwidth=1.,
                           thresh=0.2,
                           min_dur=4):
    def checker(N):
        block = np.zeros((N * 2 + 1, N * 2 + 1), dtype=np.int32)
        for k in range(-N, N + 1):
            for l in range(-N, N + 1):
                block[k + N, l + N] = np.sign(k) * np.sign(l)
        return block / abs(block).sum()

    R = recurrence_matrix(
        X, metric="cosine", mode="affinity",
        k=k, sym=sym, bandwidth=bandwidth, self=True)
    # intensify checker-board-like entries
    R_hat = convolve(R, checker(L),
                     mode="constant")  # Todo : check if mode="reflect" would avoid peaking at the end
    # extract them along the main diagonal
    dg = np.diag(R_hat, 0)
    mx = localmax(dg * (dg > thresh)).nonzero()[0]
    # filter out maxes less than min_dur frames away of the previous max
    mx = mx * (np.diff(mx, append=R.shape[0]) >= min_dur)
    mx = mx[mx > 0]
    stops = np.r_[mx, R.shape[0]]
    return stops


def as_novelty_f(graph, width=3, order=1, axis=0):
    if width < 3:
        grad = np.diff(normalize(graph), n=width, prepend=np.zeros(width))
    else:
        grad = librosa.feature.delta(normalize(graph), width=width, order=order, axis=axis)
    grad[grad < 0] = 0
    return normalize(grad)


def reduce_sequence(data, n=1, t_axis=-1, agg=lambda x, axis: x, J=None):
    agged = running_agg(data, agg=agg, window=n, hop_length=1, mode="edge",
                        p_axis=t_axis, f_axis=t_axis, a_axis=1)
    T = data.shape[t_axis]
    costs = np.zeros(T)
    if t_axis == 0:
        for t in range(n, T):
            costs[t] = J(data, agged, t, n)
    else:
        for t in range(n, T):
            costs[t] = J(data.T, agged.T, t, n)
    return costs


def KL(P, Q, t, n):
    if P[t].sum() < 5e-5 or Q[t-n].sum() < 5.e-5:
        return 0.  # ignore frames where the sum of energy is below 5.e-5
    else:
        return stats.entropy(P[t], Q[t-n])


def KL_update(P, Q, t, n):
    prior = np.mean(Q[t-n], axis=0)
    posterior = np.mean(np.vstack((Q[t-n], P[t])), axis=0)
    if prior.sum() > 5e-5 or posterior.sum() < 5.e-5:
        return 0.  # ignore frames where the sum of energy is below 5.e-5
    else:
        return stats.entropy(posterior, prior)


def self_perplexity(data, n=1, t_axis=-1, agg=np.mean, width=3, order=1):
    perplexity = reduce_sequence(data, n=n, t_axis=t_axis, agg=agg, J=KL)
    return as_novelty_f(perplexity, width=width, order=order, axis=0)


def self_perplexity_update(data, n=1, t_axis=-1, agg=lambda x, axis: x, width=3, order=1):
    perplexity = reduce_sequence(data, n=n, t_axis=t_axis, agg=agg, J=KL_update)
    return as_novelty_f(perplexity, width=width, order=order, axis=0)


# Peak Picking :

def max_in_range(y, rg=4, hop=1):
    framed = frame(y, m_frames=rg, hop_length=hop, p_axis=0, f_axis=0)
    idx = (framed.argmax(axis=1) + (np.arange(framed.shape[0]) + 1 - (rg//2)))[framed.max(axis=1) > 0]
    items, counts = np.unique(idx, return_counts=True)
    return items[counts >= rg]


def mad_outliers(y,thresh=3.5):
    """
    find outliers with median absolute deviation
    taken from https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
    """
    m = max(np.median(y), 1e-6)
    abs_dev = np.abs(y - m)
    left_mad = max(np.median(abs_dev[y <= m]), 1e-6)
    right_mad = max(np.median(abs_dev[y >= m]), 1e-6)
    y_mad = m * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    modified_z_score[np.isnan(modified_z_score) | np.isinf(modified_z_score)] = 0
    return modified_z_score > thresh


# Freq Bands Splitter :
def split_bands(data, *freqs):
    F, T = data.shape
    bins_f = librosa.fft_frequencies(n_fft=(F-1) * 2, sr=22500)
    k = 0
    bands = np.zeros(bins_f.shape, dtype=np.int32)
    for i, f in enumerate(bins_f):
        if f >= freqs[k]:
            k += 1
            bands[i] = k
        else:
            bands[i] = k
    bands = np.flatnonzero(np.diff(bands))
    return np.split(data, bands, axis=0)