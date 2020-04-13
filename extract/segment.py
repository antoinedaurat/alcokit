import matplotlib.pyplot as plt
import numpy as np
from librosa.segment import recurrence_matrix
from librosa.util import localmax
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.ndimage import convolve
from cafca.util import audio, playlist, playthrough
from cafca.fft import FFT
from cafca.transform.time import stretch


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
                                   plot=False):
    R = recurrence_matrix(
        X, metric="cosine", mode="affinity",
        k=k, sym=sym, bandwidth=bandwidth, self=True)
    # intensify checker-board-like entries
    R_hat = convolve(R, checker(L), mode="constant")  # Todo : check if mode="reflect" would avoid peaking at the end
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


class Segment(FFT):
    def __new__(cls, fft, i):
        obj = fft.view(cls)
        obj.n = obj.shape[1]
        obj.i = None
        return obj

    def __array_finalize__(self, obj):
        super(Segment, self).__array_finalize__(obj)
        self.n = obj.shape[1]
        self.i = getattr(obj, "i", None)

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
        return time_stretch(abs(self), s_range)

    def stretch_to_mod(self, m):
        m = self._check_target(m, mod=True)
        s_range = stretched_range(self.n, nearest_multiple(self.n, m))
        return time_stretch(abs(self), s_range)  # taking the abs simplifies stretching

    def trim(self, n):
        n = self._check_target(n, mod=False)
        return self[:, :n].view(Segment)

    def trim_to_mod(self, m):
        m = self._check_target(m, mod=True)
        n_hat = nearest_multiple(self.n, m)
        return self.trim(n_hat)

    def pad(self, n, mode="edge"):
        n = self._check_target(n, mod=False)
        return np.pad(self, ((0, 0), (0, abs(self.n - n))), mode=mode)

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
        return self.T.reshape(-1, seg_len, self.shape[0])

    def to_wav(self, hop_length=512):
        pass


class SegmentList(list):

    def __init__(self, sample=None, **kwargs):
        _, _, _, slices = segment_from_recurrence_matrix(abs(sample), **kwargs)
        super(SegmentList, self).__init__(Segment(sample[:, slice], i) for i, slice in enumerate(slices))
        print("found", len(self), "segments")

    @property
    def slices(self):
        i = 0
        slices = []
        for seg in self:
            slices += [np.arange(seg.n) + i]
            i += seg.n
        return slices

    def slices_stats(self):
        n, counts = np.unique([s.n for s in self], return_counts=True)
        print("segment's length || frequency")
        for x, c in zip(n, counts):
            print("          ", x, " " * (10 - len(str(x))), c)

    def mod_standardize(self, seg_len=None):
        if seg_len is None:
            seg_len = self.expected_length()
        print("snapping all lengths to multiples of", seg_len)
        return SegmentList(seg.stretch_to_mod(seg_len) for seg in self)

    def standardize(self, seg_len=None):
        if seg_len is None:
            seg_len = self.expected_length()
        print("snapping all lengths to length", seg_len)
        return SegmentList(seg.stretch(seg_len) for seg in self)

    def trim_or_pad(self, seg_len=None, mode="edge"):
        if seg_len is None:
            seg_len = self.expected_length()
        print("trimming or padding all lengths to length", seg_len)
        return SegmentList(s.trim_or_pad(seg_len, mode) for s in self)

    def group_by_len(self):
        di = {}
        for s in self:
            di.setdefault(s.n, []).append(s)
        for n, data in di.items():
            di[n] = np.stack(data)
        return di

    def playlist(self):
        return playlist(self)

    def playthrough(self):
        return playthrough(self, axis=1)

    def expected_length(self):
        return int(np.ceil(expected_len([s.n for s in self])))

    def median_length(self):
        return int(np.ceil(np.median([s.n for s in self])))

    def mode_length(self):
        x, counts = np.unique([s.n for s in self], return_counts=True)
        return int(np.ceil(x[np.argmax(counts)]))

    def as_dataset(self, seg_len):
        """
        will throw an error if all the segments don't have the same shape, i.e. haven't been standardized...
        """
        return np.concatenate([s.as_dataset(seg_len) for s in self])

    def export(self, directory):
        pass

