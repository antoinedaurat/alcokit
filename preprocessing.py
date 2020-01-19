import librosa
import numpy as np
from scipy.ndimage import convolve
from scipy.linalg import block_diag
from cafca.util import running_agg


def selfmasked(y, hop=128, n_small=256, n_big=2048, margin=4, power=100):
    S_small = librosa.stft(y, n_fft=n_small, hop_length=hop, center=False)
    S_big = librosa.stft(y, n_fft=n_big, hop_length=hop, center=False)
    a = n_big // n_big
    S_small = S_small[:, (S_small.shape[1] - S_big.shape[1]):]
    block = np.ones(a) * margin
    I = block_diag(*[block[:, None]] * (S_small.shape[0] - 1))
    S_hat = np.pad(I.dot(S_small[1:]), ((1, 0), (0, 0)))
    S_hat = S_big * librosa.util.softmask(abs(S_big), abs(S_hat), power=power)
    return S_hat


def preprocess_for_pitch(X, ga=1000, hw=2, vw=2):
    S = abs(X)
    S = convolve(S,
                 np.array([[-.65, -.25, -.65], [.75, 1.5, .75], [-.65, -.25, -.65]]))
    S -= S.min()
    S = np.log1p(S)
    # vert : maxpooling to get wide clusters
    S = running_agg(S, window=vw, agg=np.max, p_axis=0, f_axis=0, a_axis=1)
    # horz : min pooling to only keep long clusters
    S = running_agg(S, window=hw, agg=np.min, p_axis=-1, f_axis=-1, a_axis=1)
    S[S <= (S.mean() + (S.mean() / ga))] = 0
    return S
