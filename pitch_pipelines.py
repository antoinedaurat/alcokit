from cafca.pitch_tracking import *
from cafca.util import b2m
import librosa
from cafca.preprocessing import selfmasked, preprocess_for_pitch
from cafca.dynamics import uni_split_point, split_right, tag


class PitchPipeline(object):
    def __init__(self,
                 preproc_step=None,
                 split_step=None,
                 graph_step=None,
                 pred_step=None,
                 post_step=None
                 ):
        self.preproc_step = preproc_step
        self.split_step = split_step
        self.graph_step = graph_step
        self.pred_step = pred_step
        self.post_step = post_step

    def call1d(self, S):
        chains = self.split_step(S)
        centers = np.array(center(chain, S) for chain in chains)
        K = square_ratios(centers)
        graph = self.graph_step(K)
        return self.pred_step(graph=graph, K=K, centers=centers, chains=chains, S=S)

    def __call__(self, input):
        if self.preproc_step is not None:
            input = self.preproc_step(input)
        preds = []
        for t in range(input.shape[1]):
            preds += [self.call1d(input[:, t])]
        if self.post_step is not None:
            preds = self.post_step(preds)
        return preds

    @staticmethod
    def _iterate_array(input, method):
        out = []
        for t in range(input.shape[1]):
            out += [method(input[:, t])]
        return out

    @staticmethod
    def _iterate_list(input, method):
        out = []
        for t in range(len(input)):
            out += [method(input[t])]
        return out

    def get_chains(self, input):
        return self._iterate_array(input, self.split_step)

    def get_centers(self, input):
        chains = self.get_chains(input)
        _get_centers = lambda chain: [center(ch, input[:, t])
                                      for t, ch in enumerate(chain)]
        return self._iterate_list(chains, _get_centers)

    def get_graphs(self, input):
        centers = self.get_centers(input)
        Ks = [square_ratios(c) for c in centers]
        return [self.graph_step(k) for k in Ks]

    def plot_chains(self, input):
        pass

    def plot_centers(self, input):
        pass

    def plot_graphs(self, input):
        pass

    def plot_preds(self, input):
        pass


class Preprocess(object):
    @staticmethod
    def from_audio(sr=22050, n_fft=2048, hop_length=1024):
        return lambda y: librosa.stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length)


class Split(object):

    by_saddle = split_at_saddles

    @staticmethod
    def by_categorized_chain(min_b=0, max_b=np.inf,
                             min_chain_size=0, max_chain_size=np.inf,
                             n_split_right=1,
                             n_bins=150, min_cat_size=50,
                             ignore_zeros=True):
        def _split(S):
            splits = uni_split_point(S, n_bins=n_bins, min_size=min_cat_size,
                                     ignore_zeros=ignore_zeros)
            for _ in range(1, n_split_right):
                splits = split_right(S, splits,
                                     n_bins=n_bins, min_size=min_cat_size,
                                     ignore_zeros=ignore_zeros)
            mask = S >= splits[-2]
            chains = group_by_cc(S * mask)
            return [chain for chain in chains
                    if all([chain.min() >= min_b,
                            chain.max() <= max_b,
                            min_chain_size <= chain.size <= max_chain_size])]
        return _split


class Graph(object):
    @staticmethod
    def by_ints(gamma):
        return lambda S: harmonic_graph_ints(S, gamma=gamma)

    @staticmethod
    def by_dist(max_dist):
        return lambda S: harmonic_graph_dist(S, max_dist=max_dist)


class Predict(object):
    @property
    def by_heaviest_root(self):
        def _predict(graph=None, K=None, centers=None, chains=None, S=None):
            roots = locodis(graph)
            amp_w = np.array([S[chain].sum() for chain in chains])
            amp_w = amp_w / amp_w.sum()
            amp_h = [amp_w[graph[:, i].nonzero()[0]].sum() for i in range(centers.size)]
            return b2m(centers[amp_h[roots].argsort()[::-1]])
        return _predict


class Post(object):
    @staticmethod
    def asarray(n_pitch=1, min_midi=0, max_midi=127):
        def _asarray(preds):
            out = np.zeros((len(preds), n_pitch), dtype=np.float)
            for t, pred in enumerate(preds):
                pred = np.asarray(pred)
                pred = pred[(pred >= min_midi) & (pred <= max_midi)]
                pred = pred[:n_pitch]
                if pred.size < n_pitch:
                    pred = np.pad(pred, (0, n_pitch - pred.size), constant_values=-1)
                out[:, t] = pred
            return out
        return _asarray

