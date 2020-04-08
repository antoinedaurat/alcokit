import matplotlib.pyplot as plt
import numpy as np
from librosa.segment import recurrence_matrix
from librosa.util import localmax, softmask
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.ndimage import convolve
from cafca.util import normalize, logistic_map, audio


def checker(N):
    block = np.zeros((N * 2 + 1, N * 2 + 1), dtype=np.int32)
    for k in range(-N, N + 1):
        for l in range(-N, N + 1):
            block[k + N, l + N] = np.sign(k) * np.sign(l)
    return block / abs(block).sum()


def segment_from_recurrence_matrix(X,
                                   L=6,
                                   k=None,
                                   sym=True,
                                   bandwidth=1.,
                                   thresh=0.2,
                                   min_dur=4,
                                  plot=True):
    R = recurrence_matrix(
        X, metric="cosine", mode="affinity",
        k=k, sym=sym, bandwidth=bandwidth, self=True)
    # intensify checker-board-like entries
    R_hat = convolve(R, checker(L), mode="constant")
    # extract them along the main diagonal
    dg = np.diag(R_hat, 0)
    if plot:
        plt.figure(figsize=(14, 14))
        plt.imshow(R)
        plt.figure(figsize=(14, 14))
        plt.imshow(R_hat)
        plt.figure(figsize=(20, 8))
        plt.plot(dg)
    mx = localmax(dg * (dg > thresh)).nonzero()[0]
    # filter out maxes less than min_dur frames away of the previous max
    mx = mx * (np.diff(mx, prepend=0) >= min_dur)
    mx = mx[mx > 0]
    # print(mx)
    # first index is leading silence and last index is always a peak
    slices = np.split(np.arange(R.shape[0]), mx)
    return mx, R, dg, slices


def slices_stats(slices):
    for i in range(len(slices)):
        lens = np.array([s.size for s in slices[i]])
        print(i, "-->", "min:", lens.min(), "max:", lens.max(),
              "mean:", lens.mean(), "median:", np.median(lens))
        print("    n, count:")
        n, counts = np.unique(lens, return_counts=True)
        for j in range(n.size):
            print("    ", n[j], counts[j])
    return None


def expected_len(all_lens):
    n, counts = np.unique(all_lens, return_counts=True)
    return (n * counts / counts.sum()).sum()


def nearest_multiple(x, m):
    return int(m * max([1, np.rint(x / m)]))


def is_multiple(x, m):
    return x != (np.rint(x / m) * m)


def stretched_range(n, k):
    return np.linspace(0, n - 1, k)


def time_stretch(S, time_indices):
    if S.dtype in (np.complex64, np.complex128):
        mag, phase = abs(S), np.imag(S)
    else:
        mag, phase = S, None
    spline = RBS(np.arange(S.shape[0]), np.arange(S.shape[1]), mag)
    interp = spline.ev(np.arange(S.shape[0])[:, None], time_indices)
    if S.dtype in (np.complex64, np.complex128):
        # small random phases since to be working best...
        interp_p = np.random.randn(*interp.shape) * .001
        return (interp + (interp_p * 1j)).astype(S.dtype)
    else:
        return interp


def stretch_slices(o_lenghts, t_lengths):
    slices = []
    for o, t in zip(o_lenghts, t_lengths):
        if o == t:
            slices += np.arange(o)
        else:
            slices += stretched_range(o, t)
    return slices


class Segment(object):
    def __init__(self, data):
        self.data = data
        self.n = data.shape[1]

    def nearest_len(self, lens):
        return sorted(lens, key=lambda n: abs(n - self.n))[0]

    def nearest_len_by_mod(self, lens):
        return sorted(lens, key=lambda n: abs(nearest_multiple(self.n, n) - self.n))[0]

    def _check_target(self, n, mod=False):
        if isinstance(n, int):
            return n
        elif getattr(n, "__iter__", None) is not None:
            return self.nearest_len(n) if not mod else self.nearest_len_by_mod(n)

    def stretch(self, m):
        m = self._check_target(m, mod=False)
        s_range = stretched_range(self.n, m)
        self.data = time_stretch(abs(self.data), s_range)
        return self

    def stretch_to_mod(self, m):
        m = self._check_target(m, mod=True)
        s_range = stretched_range(self.n, nearest_multiple(self.n, m))
        self.data = time_stretch(abs(self.data), s_range)  # taking the abs simplifies stretching
        return self

    def trim(self, n):
        n = self._check_target(n, mod=False)
        self.data = self.data[:, :n]
        self.n = n
        return self

    def trim_to_mod(self, m):
        m = self._check_target(m, mod=True)
        n_hat = nearest_multiple(self.n, m)
        return self.trim(n_hat)

    def pad(self, n, mode="edge"):
        n = self._check_target(n, mod=False)
        self.data = np.pad(self.data, ((0, 0), (0, abs(self.n - n))), mode=mode)
        self.n = n
        return self

    def pad_to_mod(self, m, mode="edge"):
        m = self._check_target(m, mod=True)
        n_hat = nearest_multiple(self.n, m)
        return self.pad(n_hat, mode)

    def trim_or_pad(self, n, mode="edge"):
        n = self._check_target(n, mod=False)
        if n == self.n:
            return self
        elif n < self.n:
            return self.trim(n)
        else:
            return self.pad(n, mode)

    def trim_or_pad_to_mod(self, m, mode="edge"):
        m = self._check_target(m, mod=True)
        n = nearest_multiple(self.n, m)
        if n == self.n:
            return self
        elif n < self.n:
            return self.trim(n)
        else:
            return self.pad(n, mode)

    def as_dataset(self, seg_len):
        return self.data.T.reshape(-1, seg_len, self.data.shape[0])

    def to_wav(self, hop_length=512):
        pass


class SegmentMap(object):
    
    def __init__(self, sample=None, L=6, k=200, sym=True, bandwidth=5, thresh=.2, min_dur=5, plot=True):
        self.sample = sample  # fft
        self.T = sample.shape[1]
        _, _, _, self.slices = segment_from_recurrence_matrix(abs(self.sample), L, k, sym, bandwidth, thresh, min_dur, plot=plot)
        # self.idx, self.slices = self.idx[:-1], self.slices[:-1]
        self.segs = [Segment(sample[:, slice]) for slice in zip(self.slices)]
        print("found", len(self.segs), "segments")
        self.o_lens_list = [s.size for s in self.slices]

    def __iter__(self):
        for seg in self.segs:
            yield seg
    
    def slices_stats(self):
        n, counts = np.unique([s.size for s in self.slices], return_counts=True)
        print("segment's length || frequency")
        for x, c in zip(n, counts):
            print("          ", x, " " * (10 - len(str(x))), c)

    def mod_standardize(self, seg_len=None):
        if seg_len is None:
            seg_len = self.expected_length()
        print("snapping all lengths to multiples of", seg_len)
        self.segs = [seg.stretch_to_mod(seg_len) for seg in self]
        return self

    def standardize(self, seg_len=None):
        if seg_len is None:
            seg_len = self.expected_length()
        print("snapping all lengths to length", seg_len)
        self.segs = [seg.stretch(seg_len) for seg in self]
        return self

    def trim_or_pad(self, seg_len=None, mode="edge"):
        if seg_len is None:
            seg_len = self.expected_length()
        print("trimming or padding all lengths to length", seg_len)
        self.segs = [s.trim_or_pad(seg_len, mode) for s in self]
        return self

    def group_by_len(self):
        di = {}
        for s in self:
            di.setdefault(s.n, []).append(s.data)
        for n, data in di.items():
            di[n] = np.stack(data)
        return di

    def map(self, f):
        return [f(s.data) for s in self]

    def substitute(self, indices, segments):
        for i, seg in zip(indices, segments):
            self.segs[i] = Segment(seg) if isinstance(seg, np.ndarray) else seg
        return self

    def playlist(self):
        for seg in self:
            audio(seg.data, hop_length=512)
        return

    def playthrough(self):
        rv = np.concatenate([s.data for s in self], axis=1)
        return audio(rv, hop_length=512)

    def expected_length(self):
        return int(np.ceil(expected_len(self.o_lens_list)))
    
    def median_length(self):
        return int(np.ceil(np.median(self.o_lens_list)))
    
    def mode_length(self):
        x, counts = np.unique(self.o_lens_list, return_counts=True)
        return int(np.ceil(x[np.argmax(counts)]))

    def as_dataset(self, seg_len):
        """
        will throw an error if all the segments don't have the same shape, i.e. haven't been standardized...
        """
        return np.concatenate([s.as_dataset(seg_len) for s in self])

    def export(self, directory):
        pass

########################################################################################################################

# OLD SCRIPT

########################################################################################################################


def get_segment_slices(samples,
                       L=6,
                       k=200,
                       sym=True,
                       bandwidth=5,
                       thresh=.2,
                       min_dur=5,
                       plot=False):
    slices = {}
    print("segmenting sample", end=" ", flush=True)
    for i in range(len(samples)):
        print(i, end=", ", flush=True)
        _, _, dg, sl = segment_from_recurrence_matrix(abs(samples[i]),
                                                   L=L,
                                                   k=k,
                                                   sym=sym,
                                                   bandwidth=bandwidth,
                                                   thresh=thresh,
                                                   min_dur=min_dur)
        slices[i] = sl
        if plot:
            plt.figure(figsize=(samples[i].shape[1] / 30, 6))
            plt.plot(dg)
            plt.vlines([s[0] for s in sl], 0, dg.max())
            plt.title("sample " + str(i))
    print("\n")
    return slices


def standardize_slices(slices, max_length=None):
    if max_length is None:
        expected = expected_len([s.size for i in range(len(slices)) for s in slices[i]])
        m = np.ceil(expected)
    else:
        m = max_length

    true_dur = {i: [s.size + 0 for s in slices[i]] for i in range(len(slices))}
    factors = {i: [int(np.rint(n / m)) for n in true_dur[i]] for i in true_dur.keys()}

    stretched = {}
    time_idx = {}
    for i in range(len(slices)):
        sl = slices[i]
        stretched[i] = []
        time_idx[i] = []
        for j, piece in enumerate(sl):
            n = piece.size
            if is_multiple(n, m):
                # round n to the nearest multiple of m
                s_hat = stretched_range(n, nearest_multiple(n, m))
                # overwrite the slice in-place
                time_idx[i] += [s_hat]
                sl[j] = (piece, s_hat)
                stretched[i] += [True]
            else:
                time_idx[i] += [np.arange(n)]
                stretched[i] += [False]
    return slices, factors, stretched, true_dur, m


def stretch_and_stack(Xs, d, slices, factors, stretched):
    out = []
    stretch_factors = []
    print("normalizing the segments of sample", end=" ", flush=True)
    for i in Xs.keys():
        print(i, end=", ", flush=True)
        for j in range(len(slices[i])):
            if stretched[i][j]:
                original_time_idx = slices[i][j][0]
                new_time_idx = slices[i][j][1]
                if new_time_idx.size == 0:
                    print(slices[i][j])
                stretch_factors += [(original_time_idx.size, original_time_idx.size / new_time_idx.size)]
                x = Xs[i][:, original_time_idx]
                f = factors[i][j]
                new = time_stretch(x, new_time_idx)
                splits = ((1 + np.arange(f)) * int(d))
                splits = np.split(new, splits, axis=1)
                splited = np.stack(splits[:-1] if len(splits) > 1 else splits)
                out += [splited]
            else:
                f = factors[i][j]
                x = Xs[i][:, slices[i][j]]
                new = time_stretch(x, slices[i][j] - slices[i][j].min())
                stretch_factors += [(slices[i][j].size, 1)]
                f = factors[i][j]
                splits = ((1 + np.arange(f)) * int(d))
                splited = np.stack(np.split(new, splits, axis=1)[:-1])
                out += [splited]
    print("\n")
    return out, stretch_factors


def standardize_segments(samples, slices, max_length=None, encode_durs=False):
    new_slices, factors, stretched, true_dur, s = standardize_slices(slices, max_length=max_length)
    normed, durs = stretch_and_stack(samples, s, new_slices, factors, stretched)
    normed = np.concatenate(normed, axis=0)
    if encode_durs:
        return normed, true_dur
    else:
        return normed, durs, true_dur


def segment_samples(samples, mode="stand", encode_durs=False):
    slices = get_segment_slices(samples)
    if mode == "stand":
        return standardize_segments(samples, slices, encode_durs=encode_durs)
    elif mode == "raw":
        return {i: [samples[i][:, s] for s in slices] for i in samples.keys()}
    elif type(mode) == int:  # max segment length
        return standardize_segments(samples, slices, max_length=mode, encode_durs=encode_durs)
    else:
        raise NotImplementedError("No implementation available for `mode` " + str(mode))


def block_reduce(R, slices):
    """
    sum blocks of indices in a matrix to reduce its dimensionality
    """
    n = len(slices)
    out = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i, n):
            r, s = slices[i], slices[j]
            # sum and normalize the block
            out[i, j] = R[r.min():r.max() + 1, s.min():s.max() + 1].sum() / (r.size * s.size)
            out[j, i] = out[i, j]
    return out


def stack_from_arrays(X, mode="max"):
    if mode == "max":
        max_len = max(x.shape[1] for x in X)
        return np.stack([np.pad(x, ((0, 0), (0, max_len - x.shape[1])))
                         for x in X])
    elif mode == "min":
        min_len = min(x.shape[1] for x in X)
        return np.stack([x[:, :min_len] for x in X])
    elif mode == "med":
        med_len = int(np.median(np.array(tuple(x.shape[1] for x in X))))
        stack = []
        for x in X:
            if x.shape[1] > med_len:
                stack += [x[:, :med_len]]
            elif x.shape[1] < med_len:
                stack += [np.pad(x, ((0, 0), (0, med_len - x.shape[1])))]
            else:
                stack += [x]
        return np.stack(stack)


def sequence_from_slices(X, slices):
    return X[:, np.concatenate(slices)]


def matrix2dict(A, thresh=0.):
    """
    extract nonzero column entries above thresh and pack them in a dict
    with their row indice as key.
    typically, `A` is an affinity/similarity matrix and we want to extract 'families' from it
    """
    M = A.copy()
    M[M < thresh] = 0
    kids = {}
    for i in range(M.shape[0]):
        nz = set(M[i].nonzero()[0])
        if nz:
            kids[i] = nz
    return kids


def dict2kerns(d, X, agg=np.median):
    out = []
    for k, v in d.items():
        if type(v) is not dict:
            out += [agg(np.stack([X[i] for i in [k, *v]]), axis=0)]
        else:
            for n in dict2kerns(v, X, agg):
                out += [n]
    return out


def list2kerns(lst, X, agg=np.median):
    if type(lst[0]) is list:
        out = []
        for x in lst:
            out += [list2kerns(x, X, agg)]
        return out
    else:
        return agg(np.stack([X[i] for i in lst]), axis=0)


def recurrent_softmask(X, window=None, hop_length=None, margin=1.,
                       aggregated=True, normalized=False, axis=0):
    """
    accumulates the softmask zi := softmask(zi-1, xi * margin) * xi
    along the first dimension of X. X should be a 'stack' of examples
    (or typically nearest neighbors)
    """
    if aggregated:
        z = X[0].copy()
        for x in X[1:]:
            z = softmask(z, x * margin) * x
        return z if not normalized else z / z.sum()
    else:
        z = np.zeros_like(X)
        z[0] = X[0].copy()
        for i in range(1, X.shape[0]):
            xi = X[i]
            z[i] = softmask(z[i - 1], xi * margin) * xi
        return z if not normalized else z / z.sum(axis=1)


def mask_from_frame(x, strength=1):
    mask = np.log(normalize(x))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = logistic_map(mask, mask.mean(), strength=strength)
    if len(mask.shape) == 1:
        mask[:, None]
    return mask


def get_degrees(A, thresh=.5, axis=0):
    maxes = A > thresh
    return maxes.sum(axis=axis)


def connected_components(A, thresh=.5):
    visited = set()
    components = {}
    to_visit = list(range(A.shape[0]))
    cur_comp_idx = -1

    def visit(i, R, thresh, components, to_visit, visited, cur_comp_idx):
        new = (R[i] > thresh).nonzero()[0]
        if i in visited:
            return components, to_visit
        components.setdefault(cur_comp_idx, set()).add(i)
        visited.add(i)
        for n in new:
            if n in to_visit:
                components.setdefault(cur_comp_idx, set()).add(n)
                to_visit.remove(n)
                visited.add(n)
                components, to_visit = visit(n, R, thresh, components, to_visit, visited, cur_comp_idx)
        return components, to_visit

    while to_visit:
        i = to_visit.pop(0)
        cur_comp_idx += 1
        components, to_visit = visit(i, A, thresh, components, to_visit, visited, cur_comp_idx)

    return [list(v) for v in components.values()]


def cluster_twins(M, thresh=2 / 3):
    """
    connects indices of M (a symetrical similarity matrix) where Mi,j > thresh and i ≠ j
    it is important that M be symetrical because once j is merged to i, j is "removed" from M
    and will not be a key in the returned dictionary (while `i` will)
    This constraint ensures that the returned dictionary contains less indices than M,
    thus offering the possibility to reduce M to a smaller number of components at a lesser cost.
    """
    rg = np.arange(M.shape[0])
    A = M.copy()
    A[rg, rg] = 0
    maxes = A > thresh
    meta = {}
    merged = np.zeros(A.shape[0], dtype=np.bool)
    while not np.all(merged):
        first_nz = (merged == 0).nonzero()[0][0]
        if np.any(maxes[first_nz]):
            meta[first_nz] = maxes[first_nz].nonzero()[0]
            merged[first_nz] = True
            merged[meta[first_nz]] = True
        else:
            meta[first_nz] = np.array([], dtype=np.int32)
            merged[first_nz] = True
    return meta


def alphabet(X, family_func=connected_components,
             agg=np.median,
             thresh=.5, k=None, sym=True, bandwidth=1.):
    alphas = X.copy()
    R = recurrence_matrix(X, metric="cosine", mode="affinity",
                          k=k, sym=sym, bandwidth=bandwidth)
    d = family_func(R, thresh=thresh)
    while len(d) < R.shape[0]:
        alphas = dict2kerns(d, X, agg)
        R = recurrence_matrix(np.stack(alphas).T, k=k, sym=sym, bandwidth=bandwidth)
        d = family_func(R, thresh=thresh)
    return alphas


if __name__ == '__main__':
    from cafca.sampleset import SampleSet
    import os
    from cafca.util import complex2channels

    # directory = "../Segmentations_tests/Two and Three Part Inventions and Sinfonias (Glenn Gould)/"
    # directory = "../Arie/"
    #
    # samples = SampleSet(directory,
    #                     n_fft=2048, hop_length=512,
    #                     max_n_samples=-1, recursive=True)
    #
    # segmented, _, _ = segment_samples(samples.ffts)
    # segmented = complex2channels(segmented, chan_axis=1)
    # segmented = np.load(directory + "segmented.npy")
    # segmented = segmented[:, 0]
    # np.save(directory + "segmented", segmented)
    # print("saved array of shape", segmented.shape, "in", directory)

    # directory = "../data_by_technique/"
    #
    # for d in os.listdir(directory):
    #     print(d, directory+d, os.path.isdir(d))
    #     if not os.path.isdir(directory+d) or d not in ("vibrato", "vocal_fry"):
    #         continue
    #     samples = SampleSet(directory+d,
    #                         n_fft=2048, hop_length=512,
    #                         max_n_samples=-1, recursive=True)
    #
    #     segmented, _, _ = segment_samples(samples.ffts, mode=14)
    #     print(segmented.shape)
    #     np.save(directory+d+"/segmented", segmented)
    #     print("saved array of shape", segmented.shape, "in", directory)
