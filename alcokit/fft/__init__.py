import numpy as np
import librosa
from alcokit import HOP_LENGTH, N_FFT, SR
from alcokit.util import db, show, audio, signal


class FFT(np.ndarray):

    def __new__(cls, array, hop_length=HOP_LENGTH, sr=SR, file="", t_axis=1):
        obj = array.view(cls)
        obj.sr = sr
        obj.hop_length = hop_length
        obj.file = file
        obj.t_axis = t_axis
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.sr = getattr(obj, 'sr', SR)
        self.hop_length = getattr(obj, 'hop_length', HOP_LENGTH)
        self.file = getattr(obj, 'file', None)
        self.t_axis = getattr(obj, 't_axis', 1)

    @staticmethod
    def stft(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
        y, sr = librosa.load(file, sr=sr)
        return FFT(librosa.stft(y, n_fft=n_fft, hop_length=hop_length), hop_length, sr, file)

    @property
    def n_fft(self):
        f_axis = int(not bool(self.t_axis))
        return (self.shape[f_axis] - 1) * 2

    @property
    def abs(self):
        return np.abs(self)

    @property
    def angle(self):
        return np.angle(self)

    @property
    def db(self):
        return db(self)

    @property
    def signal(self):
        return signal(self, self.hop_length)

    @property
    def audio(self):
        return audio(self, self.hop_length, self.sr)

    def show(self, figsize=(), to_db=True, y_axis="linear", x_axis='frames', title=""):
        return show(self, figsize, to_db, y_axis, x_axis, title if title else self.file)


def instance_params_else_globals(array, params):
    _globals = {"n_fft": N_FFT, "hop_length": HOP_LENGTH, "sr": SR}
    rv = {}
    for param in params:
        val = getattr(array, param) if getattr(array, param) else _globals[param]
        rv[param] = val
    return rv
