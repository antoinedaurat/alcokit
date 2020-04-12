import librosa

from cafca.dynamics import uni_split_point, split_right
from cafca.pitch_tracking import *
from cafca.util import b2m


class PitchPipeline(object):
    def __init__(self,
                 preproc_step=None,
                 split_step=None,
                 prune_step=None,
                 graph_step=None,
                 pred_step=None,
                 post_step=None,
                 cache_output=False
                 ):
        self.preproc_step = preproc_step
        self.split_step = split_step
        self.prune_step = prune_step
        self.graph_step = graph_step
        self.pred_step = pred_step
        self.post_step = post_step
        self.cache_output = cache_output

    def call1d(self, S):
        chains = self.split_step(S)
        if self.prune_step is not None:
            chains = self.prune_step(S, chains)
        centers = np.array([center(chain, S) for chain in chains])
        K = square_ratios(centers)
        graph = self.graph_step(K)
        return self.pred_step(graph=graph, K=K, centers=centers, chains=chains, S=S)

    def __call__(self, input):
        if self.preproc_step is not None:
            input = self.preproc_step(input)
        preds = []
        for t in range(input.shape[1]):
            print(t, end=" ")
            preds += [self.call1d(input[:, t])]
        if self.post_step is not None:
            preds = self.post_step(preds)
        if self.cache_output:
            self.output = preds
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
        centers = [[center(c, input[:, t]) for c in ch] for t, ch in enumerate(chains)]
        return centers

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
    @staticmethod
    def by_cc(min_b=0, max_b=np.inf):
        def _split(St):
            mask = St > St.mean()
            nz = bounded_nz(mask, min_b, max_b)
            return group_by_cc(nz)
        return _split

    @staticmethod
    def by_saddle(min_b=0, max_b=np.inf,
                  min_chain_size=2, max_chain_size=np.inf):
        def _split(S):
            chains = split_at_saddles(S)
            chains = [c for c in chains
                      if c.any() and c.min() >= min_b and
                      c.max() <= max_b and
                      min_chain_size <= c.size <= max_chain_size]
            return chains

        return _split

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


class Prune(object):
    @staticmethod
    def by_skewness(window_size=3, ratio_thresh=.5, maxb=400):
        def _prune(St, chains):
            scores = [peak_skewness(St, c, w=window_size) for c in chains]
            return [c for c, score in zip(chains, scores) if c.max() <= maxb and score >= ratio_thresh]
        return _prune


class Graph(object):
    @staticmethod
    def by_ints(gamma):
        return lambda S: harmonic_graph_ints(S, gamma=gamma)

    @staticmethod
    def by_dist(max_dist):
        return lambda S: harmonic_graph_dist(S, max_dist=max_dist)


class Predict(object):
    @staticmethod
    def by_heaviest_root():
        def _predict(graph=None, K=None, centers=None, chains=None, S=None):
            """
            compute the sums of the amplitudes for each component (i.e. root)
            and return sort the roots accordingly
            """
            roots = locodis(graph)
            amp_w = np.array([S[chain].sum() for chain in chains])
            amp_w = amp_w / amp_w.sum()
            amp_h = np.array([amp_w[graph[:, i].nonzero()[0]].sum() for i in range(centers.size)])
            amp_h = amp_h / (amp_h.sum() + 1e-9)
            if roots.size == 0:
                return np.array([])
            preds = centers[amp_h[roots].argsort()[::-1]]
            # if roots.size > 0:
            #     print(list(zip(b2m(preds, n_fft=2048), np.sort(amp_h[roots])[::-1])))
            return preds
        return _predict


class Post(object):
    @staticmethod
    def asarray(n_pitch=1, min_midi=0, max_midi=127):
        def _asarray(preds):
            out = np.zeros((n_pitch, len(preds)), dtype=np.float)
            for t, pred in enumerate(preds):
                pred = np.asarray(pred)
                pred = pred[(pred >= min_midi) & (pred <= max_midi)]
                pred = pred[:n_pitch]
                if pred.size < n_pitch:
                    pred = np.pad(pred, (0, n_pitch - pred.size), constant_values=-1)
                out[:, t] = pred
            return out

        return _asarray
