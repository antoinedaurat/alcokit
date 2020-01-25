import librosa
from librosa.display import specshow, waveplot
from librosa.core.spectrum import griffinlim
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import os


# OS

def flat_dir(directory):
    files = []
    for root, dirname, filenames in os.walk(directory):
        for f in filenames:
            if f.split(".")[-1] in ("wav", "mp3", "aif", "aiff"):
                files += [os.path.join(root, f)]
    return files


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


def m2b(m, sr=22050, n_fft=2048):
    step = (sr / 2) / (n_fft // 2)
    return m2hz(m) / step


def b2m(b, sr=22050, n_fft=2048):
    step = (sr / 2) / (n_fft // 2)
    return hz2m(b * step)


def delta_b(b, delta_m=1, sr=22050, n_fft=2048):
    """
    returns the size in bins of the interval delta_m (in midi) at bin `b`
    """
    params = dict(sr=sr, n_fft=n_fft)
    return b - m2b(b2m(b, **params) - delta_m, **params)


# Useful formulas

def logistic_map(X, thresh=.1, strength=20):
    y = X - thresh
    return 1 / (1 + np.exp(- y * strength))


# Sequences

def frame(a, m_frames, hop_length=1, mode='edge', p_axis=-1, f_axis=-1):
    a = librosa.util.pad_center(a, a.shape[p_axis] + m_frames - 1, mode=mode, axis=p_axis)
    if f_axis == 0:
        a = np.ascontiguousarray(a)
    else:
        a = np.asfortranarray(a)
    return librosa.util.frame(a, frame_length=m_frames, hop_length=hop_length, axis=f_axis)


def running_agg(a, agg=lambda x, axis: x, window=10, hop_length=1, mode='edge', \
                p_axis=-1, f_axis=-1, a_axis=1):
    framed = frame(a, m_frames=window, hop_length=hop_length, p_axis=p_axis, f_axis=f_axis)
    return agg(framed, axis=a_axis)


# Display / Debug


def crop(S, minf=0, maxf=-1, mint=0, maxt=-1):
    maxf = S.shape[0] if maxf < 0 else maxf
    maxt = S.shape[1] if maxt < 0 else maxt
    return S[minf:maxf, mint:maxt]


def dbspec(S):
    S_hat = None
    if S.dtype == np.complex64:
        S_hat = a2db(abs(S)) + 40
    elif S.min() >= 0 and S.dtype in (np.float, np.float32, np.float64, np.float_):
        S_hat = a2db(S) + 40
    return S_hat


def show(S, figsize=(), y_axis="log", x_axis='frames', title=""):
    S_hat = dbspec(S)
    # make sure we have only reals in db
    if figsize:
        plt.figure(figsize=figsize)
    ax = specshow(S_hat, x_axis=x_axis, y_axis=y_axis, sr=22050)
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    return ax


def signal(S, hop_length=1024, gain=1.):
    if S.dtype == np.complex64:
        return librosa.istft(S, hop_length=hop_length) * gain
    else:
        return griffinlim(S, hop_length=hop_length) * gain


def audio(S, hop_length=1024, gain=1.):
    if len(S.shape) > 1:
        y = signal(S, hop_length=hop_length, gain=gain)
        return ipd.display(ipd.Audio(y, rate=22050))
    else:
        return ipd.display(ipd.Audio(S * gain, rate=22050))
