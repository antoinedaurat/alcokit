from alcokit.util import normalize, m2hz
import numpy as np
import torch

# WIP


def spectra(S, midi_pitches,
            peaks="laplace",  # add ["sin", "saw", "sqr", ]
            h_dist=1., tuning=0., decay_len=1.,
            peak_width=.25, max_partial=4.5):
    """
    @param S:
    @param midi_pitches:
    @param peaks:
    @param h_dist: harmonic distorsion -> partial k = f0 * k * h_dist
    @param tuning: deviation in semi-tones to the standard tuning
    @param decay_len: ~ how many octaves it takes to kill the spectrum to 0. amplitude
    @param peak_width:
    @param max_partial:
    @return:
    """
    frequencies = np.linspace(0, 22050 / 2, S.shape[0])
    mid_col = m2hz(midi_pitches + tuning)
    ratios = (frequencies / mid_col[:, None]) ** (1/h_dist)
    half_col = .5 * mid_col[:, None]
    d = (((half_col + np.maximum(frequencies, half_col)) % mid_col[:, None]) - half_col)
    # normalize each peak by forcing regions near 0 to be even closer to 0
    eps = (frequencies[1] - frequencies[0]) / 2
    if peaks == "gauss":
        less_than_eps = np.sqrt(d ** 2) <= eps
        d[less_than_eps] = 0
        d = np.exp(-(1 / (peak_width ** 2)) * (d ** 2))
    elif peaks == "laplace":
        less_than_eps = abs(d) <= eps
        d[less_than_eps] = 0
        d = np.exp(-(1 / peak_width) * abs(d))

    # exponential decay of the spectrum
    d[(ratios > max_partial)] = 0
    d *= np.exp(- ratios * (1 / decay_len))
    d = normalize(d, axis=1)
    return d


def update_score(y, instrument, score, loss_fn, loss_t):
    with torch.no_grad():
        arange = torch.arange(score.size(1))
        temp = score + torch.zeros_like(score)
        mins = score.grad.min(dim=0)
        update = mins.values / torch.norm(score.grad, p=2, dim=0)
        temp[mins.indices, arange] -= update
        new_loss = loss_fn(y, torch.mm(instrument, temp))
        updatable = new_loss < loss_t
        score[mins.indices, arange][updatable] -= update[updatable]
        score.grad.zero_()
    return score


def optim_loop(X, loss_fn, n_iter):
    Y = abs(X)
    y_nz = Y.sum(axis=0) > 0
    instru = spectra(Y, np.arange(128),
                     "laplace", tuning=0.08, h_dist=1.015,
                     decay_len=np.linspace(2, 2, 128)[:, None],
                     peak_width=np.linspace(24, 24, 128)[:, None],
                     max_partial=22.5)

    T, P = Y.shape[1], instru.shape[0]

    Yt = torch.tensor(Y)
    instrument = torch.tensor(instru.T).float().requires_grad_(False)
    score = torch.zeros(P, T).float().requires_grad_(True)

    losses = []
    predictions = torch.mm(instrument, score)
    loss = loss_fn(Yt, predictions)
    L = loss.mean()
    L.backward(retain_graph=True)
    losses += [L.item()]
    for i in range(n_iter):
        print("iter:", i)
        score = update_score(Yt, instrument, score, loss_fn, loss)
        predictions = torch.mm(instrument, score)
        loss = loss_fn(Yt, predictions)
        L = loss.mean()
        L.backward(retain_graph=True)
        losses += [L.item()]
