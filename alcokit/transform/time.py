from alcokit import HOP_LENGTH, SR
from alcokit.transform.spectrum import griffinlim
from alcokit.util import signal
from librosa import stft, phase_vocoder
import numpy as np
from pyrubberband.pyrb import time_stretch as rb_stretch
from scipy.interpolate import RectBivariateSpline as RBS


def trim(S, left=0, right=0):
    return S[:, left:-right] if right else S[left:]


def pad(S, left=0, right=0, **kwargs):
    return np.pad(S, ((0, 0), (left, right)), **kwargs)


def split(S, indices):
    return np.split(S, indices, axis=1)


def concat(*spectra):
    return np.concatenate(spectra, axis=1)


def reverse(S):
    return np.fliplr(S)


def add_start_end_tokens(S, start, end):
    pass


def _stretch_hop(S, ratio):
    y = signal(S)
    hop = int(HOP_LENGTH * ratio)
    return stft(y, hop_length=hop)


def _stretch_rbs(S, ratio):
    if S.shape[1] <= 1 or S.shape[0] <= 1:
        return S
    time_indices = np.linspace(0, S.shape[1]-1, int(np.rint(S.shape[1]/ratio)))
    if S.dtype in (np.complex64, np.complex128):
        mag, phase = abs(S), np.imag(S)
    else:
        mag, phase = S, None
    spline = RBS(np.arange(S.shape[0]), np.arange(S.shape[1]), mag)
    interp = spline.ev(np.arange(S.shape[0])[:, None], time_indices)
    if S.dtype in (np.complex64, np.complex128):
        return griffinlim(interp, n_iter=32)
    else:
        return interp


def _stretch_vocoder(S, ratio):
    return phase_vocoder(S, ratio, hop_length=HOP_LENGTH)


def _stretch_rubberband(S, ratio):
    y = signal(S)
    return stft(rb_stretch(y, SR, ratio), hop_length=HOP_LENGTH)


def stretch(S, ratio, method="rbs"):
    funcs = {"rbs": _stretch_rbs, "hop": _stretch_hop, "rubber": _stretch_rubberband, "voc": _stretch_vocoder}
    if method not in funcs:
        raise ValueError("'method' value '{}' not understood."
                         "'method' arg should be one of ['hop', 'rbs', 'voc', 'rubber']")
    func = funcs[method]
    return func(S, ratio)


def align(X, Y, method="rbs"):
    N, M = X.shape[1], Y.shape[1]
    k = (N + M) / 2
    x, y = stretch(X, N/k, method), stretch(Y, M/k, method)
    assert x.shape[1] == y.shape[1]
    return x, y


def stretch_map(S, map, method="rbs"):
    pass

