import numpy as np
from librosa.core import istft, stft
import librosa
import warnings


def griffinlim(S, n_iter=32, hop_length=None, win_length=None, window='hann',
               center=True, dtype=np.float32, length=None, pad_mode='reflect',
               momentum=0.99, init='random', random_state=None):
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise TypeError("type '{}' not valid for argument 'random_state'".format(type(random_state)))

    if momentum > 1:
        warnings.warn('Griffin-Lim with momentum={} > 1 can be unstable. '
                      'Proceed with caution!'.format(momentum))
    elif momentum < 0:
        raise ValueError('griffinlim() called with momentum={} < 0'.format(momentum))

    # Infer n_fft from the spectrogram shape
    n_fft = 2 * (S.shape[0] - 1)

    # using complex64 will keep the result to minimal necessary precision
    angles = np.empty(S.shape, dtype=np.complex64)
    if init == 'random':
        # randomly initialize the phase
        angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ValueError("init={} must either None or 'random'".format(init))

    # And initialize the previous iterate to 0
    rebuilt = 0.

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = istft(S * angles, hop_length=hop_length, win_length=win_length,
                        window=window, center=center, dtype=dtype, length=length)

        # Rebuild the spectrogram
        rebuilt = stft(inverse, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window, center=center,
                       pad_mode=pad_mode)

        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    # Return the final phase estimates
    return S * angles


def exponentiate(S, thresh=.1, strength=20):
    y = S - thresh
    return 1 / (1 + np.exp(- y * strength))


def softmask(X, X_ref, power=1, split_zeros=False):
    return librosa.util.softmask(X, X_ref, power=power, split_zeros=split_zeros)


def frame(a, m_frames, hop_length=1, mode='edge', p_axis=-1, f_axis=-1):
    a = librosa.util.pad_center(a, a.shape[p_axis] + m_frames - 1, mode=mode, axis=p_axis)
    if f_axis == 0:
        a = np.ascontiguousarray(a)
    else:
        a = np.asfortranarray(a)
    return librosa.util.frame(a, frame_length=m_frames, hop_length=hop_length, axis=f_axis)


def running_agg(a, agg=lambda x, axis: x, window=10, hop_length=1, mode='edge',\
                p_axis=-1, f_axis=-1, a_axis=1):
    framed = frame(a, m_frames=window, hop_length=hop_length, p_axis=p_axis, f_axis=f_axis, mode=mode)
    return agg(framed, axis=a_axis)


def smooth_2d(S, hw=2, vw=2, hagg=np.min, vagg=np.max):
    out = running_agg(S, window=vw, agg=vagg, p_axis=0, f_axis=0, a_axis=1)
    out = running_agg(out, window=hw, agg=hagg, p_axis=-1, f_axis=-1, a_axis=1)
    return out



