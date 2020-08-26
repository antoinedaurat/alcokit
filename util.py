import librosa
import numpy as np
import os
from librosa.display import specshow
import matplotlib.pyplot as plt
import IPython.display as ipd
from alcokit import HOP_LENGTH, SR, N_FFT
import pickle


def save_pickle(obj, path):
    with open(path, "wb") as f:
        f.write(pickle.dumps(obj))
    return None


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj


# OS
def is_audio_file(file):
    return file.split(".")[-1] in ("wav", "aif", "aiff", "mp3", "m4a", "mp4") and "._" not in file


def flat_dir(directory, ext_filter=is_audio_file):
    files = []
    for root, dirname, filenames in os.walk(directory):
        for f in filenames:
            if ext_filter(f):
                files += [os.path.join(root, f)]
    return sorted(files)


def fs_dict(root, extension_filter=is_audio_file):
    root_name = os.path.split(root.strip("/"))[-1]
    items = [(d, list(filter(extension_filter, f))) for d, _, f in os.walk(root)]
    if not items:
        raise ValueError("no audio files found on path %s" % root)
    return root_name, dict(item for item in items if len(item[1]) > 0)


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


# Debugging utils

def db(S):
    if S.dtype == np.complex64:
        S_hat = a2db(S.abs) + 40
    elif S.min() >= 0 and S.dtype in (np.float, np.float32, np.float64, np.float_):
        S_hat = a2db(S) + 40
    else:
        S_hat = a2db(S)
    return S_hat


def signal(S, hop_length=HOP_LENGTH):
    if S.dtype in (np.complex64, np.complex128):
        return librosa.istft(S, hop_length=hop_length)
    else:
        return librosa.griffinlim(S, hop_length=hop_length, n_iter=32)


def audio(S, hop_length=HOP_LENGTH, sr=SR):
    if len(S.shape) > 1:
        y = signal(S, hop_length)
        if y.size > 0:
            return ipd.display(ipd.Audio(y, rate=sr))
        else:
            return ipd.display(ipd.Audio(np.zeros(hop_length*2), rate=sr))
    else:
        return ipd.display(ipd.Audio(S, rate=sr))


def playlist(iterable):
    for seg in iterable:
        audio(seg)
    return


def playthrough(iterable, axis=1):
    rv = np.concatenate(iterable, axis=axis)
    return audio(rv)


def show(S, figsize=(), to_db=True, y_axis="linear", x_axis='frames', title=""):
    S_hat = db(S) if to_db else S
    if figsize:
        plt.figure(figsize=figsize)
    ax = specshow(S_hat, x_axis=x_axis, y_axis=y_axis, sr=SR)
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    return ax
