import torch
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
import numpy as np
import h5py
from alcokit.hdf.api import FeatureProxy
from alcokit.sources import ScoredSourceMixin, NPSource


class FlatSampler(Sampler):
    """
    returns INT indices
    """
    def __init__(self, score, shuffle=True):
        super(FlatSampler, self).__init__([0])
        self.indices = score.flat
        self.N = self.indices.size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return self.N


class EventSampler(Sampler):
    """
    returns EVENT indices
    """
    def __init__(self, score, shuffle=True):
        super(EventSampler, self).__init__([0])
        self.score = score
        self.N = len(score)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.score = self.score.sample(frac=1.)
        return self.score.events

    def __len__(self):
        return self.N


class FrameSampler(Sampler):
    """
    returns SLICE indices
    """
    def __init__(self, N, k=1, stride=1, shifts=tuple(), shuffle=True):
        super(FrameSampler, self).__init__([0])
        self.base_idx = np.arange(N - k - sum(shifts) + 1, step=stride)
        self.k = k
        self.N = len(self.base_idx)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.base_idx)
        return iter(slice(i, i+self.k) for i in self.base_idx)

    def __len__(self):
        return self.N


def get_event(x, event, shift=0):
    if type(event) is slice or "Event" in str(type(event)):
        slc = slice(event.start+shift, event.stop+shift)
        return x[slc]
    return x[event+shift]


def zip_stack(batch_lists):
    """
    returns (stack(feat_1), stack(feat_2)...)
    """
    rv = tuple(torch.stack(x) for x in zip(*batch_lists))
    return rv


def zip_list(batch_lists):
    return tuple(x for x in zip(*batch_lists))


class DSBase(Dataset):
    def __init__(self):
        self.N = None

    def __len__(self):
        return self.N

    def set_length(self, N):
        self.N = N
        return self


class OpenedDS(DSBase):
    def __init__(self, file, ds_name, device="cpu", shift=0):
        self.file = h5py.File(file, "r")
        self.ds = self.file[ds_name]
        self.shape = self.ds.shape
        self.shift = shift
        self.device = device
        super(OpenedDS, self).__init__()

    def __getitem__(self, event):
        array = get_event(self.ds, event, self.shift)
        return torch.from_numpy(array)


class TensorSource(DSBase):
    def __init__(self, src, shift=0):
        self.src = src
        self.shape = self.src.shape
        self.shift = shift
        super(TensorSource, self).__init__()

    def __getitem__(self, event):
        return get_event(self.src, event, self.shift)


class MultiSource(DSBase):
    def __init__(self, *srcs):
        self.srcs = srcs
        super(MultiSource, self).__init__()

    def __getitem__(self, event):
        return tuple(src[event] for src in self.srcs)


def prepare(source, score, pre_cat=False, device="cpu", shift=0):
    if issubclass(type(source), ScoredSourceMixin):
        if pre_cat:
            tp = type(source)
            source = torch.from_numpy(source.colcat(score)).to(device)
            # TAKE CARE OF THE TIME AXIS with thos known transposing sources
            source = source.T if tp in (FeatureProxy, NPSource) else source
            source = TensorSource(source, shift)
        else:
            source = OpenedDS(source.h5_file, source.name, device, shift)
    else:
        if type(source) is np.ndarray:
            source = torch.from_numpy(source).to(device)
        if pre_cat:
            source = torch.cat([source[i] for i in score.slices(time_axis=0)])
        source = TensorSource(source, shift)
    return source


def load(features, score, mode,
         pre_cat=False, device="cpu", shifts=None,
         shuffle=True, batch_size=None, drop_last=False,
         collate_fn=zip_stack,
         num_workers=0,
         **fsamp_kwargs):
    if not isinstance(features, tuple):
        features = (features, )
    # prepare
    features = tuple(prepare(src, score, pre_cat, device, shft)
                     for src, shft in zip(features, shifts if shifts is not None else [0] * len(features)))
    if pre_cat:
        score.make_contiguous()
    # score -> Sampler
    if mode == "flat":
        sampler = FlatSampler(score, shuffle)
    elif mode == "event":
        sampler = EventSampler(score, shuffle)
    elif mode == "frame":
        if not pre_cat and not score.is_contiguous():
            print("WARNING: score is not contiguous and pre_cat is False which might"
                  " lead to errors or undesired results when using mode='frame'")
        sampler = FrameSampler(score.span, shifts=shifts, shuffle=shuffle, **fsamp_kwargs)
    else:
        raise ValueError("`mode` value '%s' not recognized. Must be one of 'flat', 'event'" % mode)

    # batches are ALWAYS tuples of tensor
    features = MultiSource(*features).set_length(len(sampler))

    # source + sampler -> DataLoader
    if batch_size is not None:
        sampler = BatchSampler(sampler, batch_size, drop_last)
        return DataLoader(features, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    return DataLoader(features, sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
