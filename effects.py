import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import non_negative_factorization

from cafca.util import t2f, t2s, f2t, show, signal, audio


def harmonic_percussive(S, margin=1, figsize=(16, 4)):
    D_h, D_p = librosa.decompose.hpss(S, margin=margin)

    if figsize is not None:
        show(S, figsize=figsize, title="original")
        show(D_h, figsize=figsize, title="harmonic")
        show(D_p, figsize=figsize, title="percussive")
    print("original")
    audio(S)
    print("harmonic")
    audio(D_h)
    print("percussive")
    audio(D_p)
    return D_h, D_p


def REP_SIM(S,
            aggregate=np.median,
            metric='cosine',
            width_sec=2.,
            margin_b=2, margin_f=10, power=2,
            figsize=(16, 4)):
    S_full, _ = librosa.magphase(S)
    w = int(t2f(width_sec, sr=22050, hop_length=1024))
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=aggregate,
                                           metric=metric,
                                           width=w)

    S_filter = np.minimum(S_full, S_filter)

    mask_b = librosa.util.softmask(S_filter,
                                   margin_b * (S_full - S_filter),
                                   power=power)

    mask_f = librosa.util.softmask(S_full - S_filter,
                                   margin_f * S_filter,
                                   power=power)

    S_foreground = mask_f * S_full
    S_background = mask_b * S_full

    if figsize is not None:
        show(S_full, figsize=figsize, title="original")
        show(S_background, figsize=figsize, title="background")
        show(S_foreground, figsize=figsize, title="foreground")

    print("original")
    audio(S_full)
    print("background")
    audio(S_background)
    print("foreground")
    audio(S_foreground)
    return S_background, S_foreground


def alignement(X, Y,
               compute_chroma=True,
               metric='cosine',
               figsize=(10, 10),
               ):
    if compute_chroma:
        x = librosa.feature.chroma_stft(S=abs(X), tuning=0, norm=2)
        y = librosa.feature.chroma_stft(S=abs(Y), tuning=0, norm=2)
    else:
        x = abs(X)
        y = abs(Y)

    D, wp = librosa.sequence.dtw(X=x, Y=y, metric=metric, subseq=True)

    if figsize is not None:
        show(D, figsize=figsize, title="costs matrix", x_axis="frames", y_axis="frames")
        plt.plot(wp[:, 1], wp[:, 0], label='Optimal Path', color='green')
        plt.legend()

    return D, wp[::-1]


def random_path(X, Y, D, wp,
                mode="path",
                start_with=0,  # i.e X ; 1 for Y
                start_frame=100,
                n_crossing=3,
                min_step_duration=2.,
                max_step_duration=2.,  # or float
                xfade_dur=.1,
                ordered=True,
                show_result=False,
                ):
    min_step_duration = int(t2f(min_step_duration, sr=22050, hop_length=1024))
    if max_step_duration is None:
        max_step_duration = X.shape[1] // n_crossing
    else:
        max_step_duration = int(t2f(max_step_duration, sr=22050, hop_length=1024))

    crossing_durs = np.random.randint(min_step_duration, max_step_duration, size=n_crossing)

    if mode == "path":
        idx_getter = lambda k, t, trim: wp[t, k]
    elif mode == "max":
        agg = np.argmax
        idx_getter = lambda k, t, trim: int(agg(D[t, :-trim])) if k == 0 else int(agg(D.T[t, :-trim]))
    elif mode == "min":
        agg = np.argmin
        idx_getter = lambda k, t, trim: int(agg(D[t, :-trim])) if k == 0 else int(agg(D.T[t, :-trim]))
    else:
        raise ValueError("value %s for 'mode' is not recognized" % mode)

    maxt_x, maxt_y = D.shape
    last_i = start_frame
    xfade_dur = int(t2s(xfade_dur, sr=22050)) // 2
    pieces = np.zeros(xfade_dur)
    fade_in = np.arange(xfade_dur) / xfade_dur
    fade_out = np.arange(xfade_dur)[::-1] / xfade_dur
    for i, d in enumerate(crossing_durs, int(bool(start_with))):
        k = i % 2

        print(["X", "Y"][k], "from frame", last_i, "to", last_i + d, " (= %.3f seconds)" % f2t(d))

        z = [X, Y][k][:, last_i:last_i + d]
        last_i += d
        last_i = idx_getter(k, last_i % [maxt_x, maxt_y][k], d + 1)

        # join the audios :
        z = signal(z)
        z[:xfade_dur] *= fade_in
        z[-xfade_dur:] *= fade_out
        pieces[-xfade_dur:] += z[:xfade_dur]
        pieces = np.concatenate((pieces, z[xfade_dur:]))

    return audio(pieces)


def decompose(X,
              mode=0,  # 0 = learn to output component, 1 = learn to output score
              n_components=50,
              comp_length=1,
              max_iter=200,
              regularization=0.,
              seed=1,
              figsize=(12, 4),
              ):
    if X.dtype == np.complex64:
        X = abs(X)

    F, T = X.shape

    if comp_length > 1:
        X = np.pad(X, ((0, 0), (0, T % comp_length)))
        X = X.reshape(F * comp_length, -1)

    nmf = lambda S: non_negative_factorization(
        X=X, W=None, H=None, n_components=n_components, init=None,
        update_H=True, solver="mu",
        max_iter=max_iter, alpha=regularization,
        random_state=seed)
    C, S, _ = nmf(X)
    rec = recompose(C, S, F)

    if figsize is not None:
        show(X, title="original", figsize=figsize)
        show(C, title="components", figsize=figsize, y_axis="linear")
        show(S, title="score", figsize=figsize, x_axis=None, y_axis="linear")
        show(rec, title="reconstruction", figsize=figsize)
    print("reconstruction")
    audio(rec)
    return C, S


def score_for(X, C, max_iter=200, regularization=0., seed=1, ):
    nmf = lambda X: non_negative_factorization(
        X=X.T, W=None, H=C.T,
        n_components=C.shape[1], init="custom",
        update_H=False, solver="mu",
        max_iter=max_iter, alpha=regularization,
        random_state=seed)
    S, _, _ = nmf(abs(X))
    return S.T


def components_for(X, S, max_iter=200, regularization=0., seed=1):
    if X.shape[1] < S.shape[1]:
        S = S[:, :X.shape[1]]
    elif X.shape[1] > S.shape[1]:
        X = X[:, :S.shape[1]]
    nmf = lambda X: non_negative_factorization(
        X=X, W=None, H=S,
        n_components=S.shape[0], init="custom",
        update_H=False, solver="mu",
        max_iter=max_iter, alpha=regularization,
        random_state=seed)
    C, _, _ = nmf(abs(X))
    return C


def recompose(C, S, F=0):
    res = C @ S
    if F > 0:
        return res.reshape(F, -1)
    return res
