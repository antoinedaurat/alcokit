from itertools import combinations

import librosa
import numpy as np
from librosa.sequence import dtw
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics import pairwise_distances

from alcokit.fft import FFT
from alcokit.util import t2f


def harmonic_percussive(S, margin=1):
    D_h, D_p = librosa.decompose.hpss(S, margin=margin)

    if isinstance(S, FFT):
        D_h = FFT(D_h, S.hop_length, S.sr)
        D_p = FFT(D_p, S.hop_length, S.sr)
    return D_h, D_p


def REP_SIM(S,
            aggregate=np.median,
            metric='cosine',
            width_sec=2.,
            margin_b=2, margin_f=10, power=2):
    S_full, _ = librosa.magphase(S)
    w = int(t2f(width_sec, sr=22050, hop_length=1024))
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=aggregate,
                                           metric=metric,
                                           width=w)

    S_filter = np.minimum(S_full, S_filter)

    mask_b = librosa.util.softmask(S_filter,
                                   margin_b * (S_full - S_filter),
                                   power=power)

    mask_f = librosa.util.softmask(S_full - S_filter,
                                   margin_f * S_filter,
                                   power=power)

    S_foreground = mask_f * S_full
    S_background = mask_b * S_full

    if isinstance(S, FFT):
        S_foreground = FFT(S_foreground, S.hop_length, S.sr)
        S_background = FFT(S_background, S.hop_length, S.sr)

    return S_background, S_foreground


def optimal_path(x, y):
    return dtw(C=pairwise_distances(abs(x.T), abs(y.T), metric='cosine'))[1][::-1]


def align(*ffts):
    path_dict = {}
    N = len(ffts)
    for i, j in combinations(range(N), r=2):
        path_dict[(i, j)] = optimal_path(ffts[i], ffts[j])
        print(i, j)
    return path_dict


def decompose(X,
              n_components=50,
              comp_length=1,
              max_iter=200,
              regularization=0.,
              seed=1,
              ):
    if X.dtype == np.complex64:
        X = abs(X)

    F, T = X.shape

    if comp_length > 1:
        X = np.pad(X, ((0, 0), (0, T % comp_length)))
        X = X.reshape(F * comp_length, -1)

    nmf = lambda S: non_negative_factorization(
        X=X, W=None, H=None, n_components=n_components, init=None,
        update_H=True, solver="mu",
        max_iter=max_iter, alpha=regularization,
        random_state=seed)
    C, S, _ = nmf(X)

    return C, S


def score_for(X, C, max_iter=200, regularization=0., seed=1, ):
    nmf = lambda X: non_negative_factorization(
        X=X.T, W=None, H=C.T,
        n_components=C.shape[1], init="custom",
        update_H=False, solver="mu",
        max_iter=max_iter, alpha=regularization,
        random_state=seed)
    S, _, _ = nmf(abs(X))
    return S.T


def components_for(X, S, max_iter=200, regularization=0., seed=1):
    if X.shape[1] < S.shape[1]:
        S = S[:, :X.shape[1]]
    elif X.shape[1] > S.shape[1]:
        X = X[:, :S.shape[1]]
    nmf = lambda X: non_negative_factorization(
        X=X, W=None, H=S,
        n_components=S.shape[0], init="custom",
        update_H=False, solver="mu",
        max_iter=max_iter, alpha=regularization,
        random_state=seed)
    C, _, _ = nmf(abs(X))
    return C


def recompose(C, S, F=0):
    res = C @ S
    if F > 0:
        return res.reshape(F, -1)
    return res
