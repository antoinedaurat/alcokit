from cafca import SR, HOP_LENGTH
from cafca.util import signal, f2s
import numpy as np
from librosa import resample, phase_vocoder, util, stft
from pyrubberband.pyrb import pitch_shift as rb_shift


def _shift_scale_freq(S, intv):
    pass


def _shift_vocoder(S, n_steps, bins_per_octave=12):
    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.integer):
        raise ValueError('bins_per_octave must be a positive integer.')

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

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
