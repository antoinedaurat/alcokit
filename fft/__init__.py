import numpy as np
import librosa
from cafca import HOP_LENGTH, N_FFT, SR
from cafca.util import a2db, show, audio, signal


class FFT(np.ndarray):

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        obj = super(FFT, cls).__new__(cls, shape, dtype,
                                      buffer, offset, strides,
                                      order)
        obj.sr = SR
        obj.hop = HOP_LENGTH
        obj.file = None
        obj.t_axis = 1
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.sr = getattr(obj, 'sr', SR)
        self.hop = getattr(obj, 'hop', HOP_LENGTH)
        self.file = getattr(obj, 'file', None)
        self.t_axis = getattr(obj, 't_axis', 1)

    @staticmethod
    def to_fft(array, hop_length=HOP_LENGTH, sr=SR, file=None, t_axis=1):
        rv = array.view(FFT)
        rv.hop = hop_length
        rv.sr = sr
        rv.file = file
        rv.t_axis = t_axis
        return rv

    @staticmethod
    def stft(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
        y, sr = librosa.load(file, sr=sr)
        return FFT.to_fft(librosa.stft(y, n_fft=n_fft, hop_length=hop_length), hop_length, sr, file)

    @property
    def n_fft(self):
        f_axis = int(not bool(self.t_axis))
        return (self.shape[f_axis] - 1) * 2

    @property
    def abs(self):
        return np.abs(self)

    @property
    def phi(self):
        return np.exp(1.j * np.angle(self))

    def db(self):
        S_hat = None
        if self.dtype == np.complex64:
            S_hat = a2db(self.abs) + 40
        elif self.min() >= 0 and self.dtype in (np.float, np.float32, np.float64, np.float_):
            S_hat = a2db(self) + 40
        else:
            S_hat = a2db(self)
        return S_hat

    def signal(self):
        return signal(self, self.hop)

    def audio(self):
        return audio(self, self.hop, self.sr)

    def show(self, figsize=(), to_db=True, y_axis="linear", x_axis='frames', title=""):
        return show(self, figsize, to_db, y_axis, x_axis, title)
