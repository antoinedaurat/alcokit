import librosa
import numpy as np
import os
from librosa.display import specshow
import matplotlib.pyplot as plt
import IPython.display as ipd
from cafca import HOP_LENGTH, SR, N_FFT


# OS
def is_audio_file(file):
    return file.split(".")[-1] in ("wav", "aif", "aiff", "mp3", "m4a") and "._" not in file


def flat_dir(directory):
    files = []
    for root, dirname, filenames in os.walk(directory):
        for f in filenames:
            if is_audio_file(f):
                files += [os.path.join(root, f)]
    return sorted(files)


# Conversion

normalize = librosa.util.normalize
a2db = lambda S: librosa.amplitude_to_db(abs(S), ref=S.max())
s2f = librosa.samples_to_frames
s2t = librosa.samples_to_time
f2s = librosa.frames_to_samples
f2t = librosa.frames_to_time
t2f = librosa.time_to_frames
t2s = librosa.time_to_samples
hz2m = librosa.hz_to_midi
m2hz = librosa.midi_to_hz


def m2b(m, sr=SR, n_fft=N_FFT):
    step = (sr / 2) / (n_fft // 2)
    return m2hz(m) / step


def b2m(b, sr=SR, n_fft=N_FFT):
    step = (sr / 2) / (n_fft // 2)
    return hz2m(b * step)


def delta_b(b, delta_m=1, sr=SR, n_fft=N_FFT):
    """
    returns the size in bins of the interval delta_m (in midi) at bin `b`
    """
    params = dict(sr=sr, n_fft=n_fft)
    return b - m2b(b2m(b, **params) - delta_m, **params)


def unit_scale(x):
    return (x - x.min()) / (x.max() - x.min())


# Useful formulas

def logistic_map(X, thresh=.1, strength=20):
    y = X - thresh
    return 1 / (1 + np.exp(- y * strength))


# Debugging utils

def signal(S, hop_length=HOP_LENGTH):
    if S.dtype in (np.complex64, np.complex128):
        return librosa.istft(S, hop_length=hop_length)
    else:
        return librosa.griffinlim(S, hop_length=hop_length)


def audio(S, hop_length=HOP_LENGTH, sr=SR):
    if len(S.shape) > 1:
        y = signal(S, hop_length)
        return ipd.display((ipd.Audio(y, rate=sr),))
    else:
        return ipd.display((ipd.Audio(S, rate=sr),))


def show(S, figsize=(), to_db=True, y_axis="linear", x_axis='frames', title=""):
    S_hat = S.db() if to_db else S
    S_hat = S_hat if S.t_axis == 1 else S_hat.T
    if figsize:
        plt.figure(figsize=figsize)
    ax = specshow(S_hat, x_axis=x_axis, y_axis=y_axis, sr=SR)
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    return ax


# Sequences

def frame(a, m_frames, hop_length=1, mode='edge', p_axis=-1, f_axis=-1):
    a = librosa.util.pad_center(a, a.shape[p_axis] + m_frames - 1, mode=mode, axis=p_axis)
    if f_axis == 0:
        a = np.ascontiguousarray(a)
    else:
        a = np.asfortranarray(a)
    return librosa.util.frame(a, frame_length=m_frames, hop_length=hop_length, axis=f_axis)


def running_agg(a, agg=lambda x, axis: x, window=10, hop_length=1, mode='edge',\
                p_axis=-1, f_axis=-1, a_axis=1):
    framed = frame(a, m_frames=window, hop_length=hop_length, p_axis=p_axis, f_axis=f_axis)
    return agg(framed, axis=a_axis)
