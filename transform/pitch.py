from cafca import SR, HOP_LENGTH
from cafca.util import signal, f2s
import numpy as np
from librosa import resample, phase_vocoder, util, stft
from pyrubberband.pyrb import pitch_shift as rb_shift


def _shift_translate(S, k_bins):
    n = S.shape[0]
    k = k_bins
    rng = np.arange(n)
    i_rng = rng if k == 0 else (rng[:-k] if k > 0 else rng[-k:])
    j_rng = rng if k == 0 else (rng[k:] if k > 0 else rng[:k])
    D = np.zeros((n, n))
    D[i_rng, j_rng] = 1
    return D.T @ S


def _shift_scale(S, rate):
    n = S.shape[0]
    i_range = np.arange(n)
    j = i_range * rate
    D = np.zeros((n, n))

    for i, alpha in zip(i_range, j):
        floor = int(np.floor(alpha))
        ceil = floor + 1
        if floor >= n:
            break
        col = [floor, ceil] if ceil < n else floor
        val = [ceil - alpha, alpha - floor] if ceil < n else ceil - alpha
        D[i, col] = val
    return D.T @ S


def _steps2rate(n_steps, bins_per_octave=12):
    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.integer):
        raise ValueError('bins_per_octave must be a positive integer.')
    return 2.0 ** (-float(n_steps) / bins_per_octave)


def _shift_vocoder(S, rate):
    y = signal(phase_vocoder(S, rate, hop_length=HOP_LENGTH))
    y_shift = resample(y, float(SR) / rate, SR,
                       res_type="kaiser_best")

    # Crop to the same dimension as the input
    return util.fix_length(y_shift, f2s(S.shape[1], HOP_LENGTH))


def _shift_rubber(S, intv):  # TODO : DOES THIS SUPPORT QUARTER-TONES ??
    y = signal(S)
    return stft(rb_shift(y, SR, intv), hop_length=HOP_LENGTH)


def shift(S, intv, method="voc"):
    pass


def rotate(S, k, clip=True):
    pass


def flip(S):
    return np.flipud(S)


def repitch(S):
    pass


def rephase(S, eps):
    pass
