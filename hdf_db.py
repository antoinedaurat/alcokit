import h5py
import os
import pickle
import re
from copy import deepcopy
from sklearn.neighbors import KNeighborsTransformer
from cafca.util import is_audio_file
from cafca.fft import FFT
from cafca.extract.segment import SegmentList
import librosa
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch import from_numpy, cuda, device as torch_device
from torch.nn.utils.rnn import pack_sequence
import warnings

warnings.filterwarnings("ignore")

'''
@todo : never forget : "TypeError: h5py objects cannot be pickled"
'''




class HdBase(object):
    compression_args = dict(compression="gzip", compression_opts=4)

    @staticmethod
    def data_transform(abs_path):
        return FFT.stft(abs_path)

    def pre_process(self, root_dir, rel_file_path):
        print("processing", os.path.split(rel_file_path)[-1])
        abs_path = root_dir + rel_file_path
        data = self.data_transform(abs_path)
        tmp_db = ".".join(abs_path.split(".")[:-1] + ["h5"])
        with h5py.File(tmp_db, "w") as f:
            f.create_dataset(rel_file_path, shape=data.shape if data.shape else (1,), data=data,
                             **self.compression_args)
            f.close()
        return tmp_db

    def post_process(self, root_db, tmp_db):

        def _copy(name, data):
            if isinstance(data, h5py.Dataset):
                root_db.create_dataset(name, data=data[()], **self.compression_args)
            return None

        f = h5py.File(tmp_db, "r")
        f.visititems(_copy)
        f.close()
        os.remove(tmp_db)
        return None

    def make(self, root_directory,
             n_cores=cpu_count()):
        target_file = self.h5_file
        root_name, tree = audio_fs_dict(root_directory)
        print(root_name)
        args = [(root_directory, os.path.join(os.path.relpath(dir, root_directory), file))
                for dir, files in tree.items() for file in files]
        with Pool(n_cores) as p:
            tmp_dbs = p.starmap(self.pre_process, args)
        print("saving results in .h5 file")
        f = h5py.File(target_file, "w")
        # root_group = f.create_group(root_name)
        for db in tmp_dbs:
            self.post_process(f, db)
        f.close()
        print("Done")
        return HdBase(target_file)

    def __init__(self, h5_file):
        self.h5_file = h5_file
        # the h5 file object cannot be stored in an instance because it will break multiprocessing (unpicklable object...)
        try:
            f = h5py.File(h5_file, "r")
        except OSError:
            return
        self.ds = {}

        def register_ds(name, ds):
            if ds.__class__ == h5py.Dataset:
                self.ds[name] = len(ds)
                return None
            return None

        print("registering")
        f.visititems(register_ds)
        self.n = len(self.ds)
        self.N = sum(self.ds.values())
        f.close()
        print("Done!")

    def open(self, mode="r"):
        return h5py.File(self.h5_file, mode)

    def __getitem__(self, args):
        """
        returns the Datasets referenced or matched by `ds` in the h5py file
        @type ds: `str` or `re.Pattern`
        @return: the found Datasets if `ds` is a `str` and the list of the Datasets whose name matched `ds`
        if `ds` is a regex
        N.B. THE FILE HANDLE TO THE DB MUST BE OPENED PRIOR TO CALLING THIS METHOD!
        """
        f, ds = args
        if isinstance(ds, str):
            return f[ds]
        elif isinstance(ds, re.Pattern):
            return [f[k] for k in self.ds.keys() if ds.search(k) is not None]
        else:
            raise ValueError("method __getitem__ not implemented for the type `%s`" % str(type(ds)))

    def __iter__(self):
        """
        @yields: the Datasets stored in the h5py file
        """
        for ds in self.ds.keys():
            yield self[ds]


class HdfDB(HdBase):

    @staticmethod
    def collate_fn(alist):
        return from_numpy(np.concatenate(alist, axis=0))

    def __init__(self, h5_file, batch_size, cache_size, ds_filter=None, device=None):
        super(HdfDB, self).__init__(h5_file)
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.gen = None
        self.filtr=ds_filter
        self.cache = None
        self.device = device if device is not None else torch_device('cuda' if cuda.is_available() else 'cpu')
        self.init_sampling()
        self.bpe = self.N // batch_size
        print("ready to yield", self.bpe, "batches for", self.N, "datapoints")

    def size(self):
        return self.cache.size()

    def ds_gen(self):
        keys = list(self.ds.keys()) if self.filtr is None else list(filter(self.filtr, self.ds.keys()))
        for i in np.random.permutation(np.arange(len(keys))):
            yield keys[i]

    def init_sampling(self):
        self.gen = self.ds_gen()
        return self._refill_cache()

    def _refill_cache(self):
        f = self.open()
        N, i = 0, 0
        to_cache = []
        while N < self.cache_size:
            try:
                new = self[f, next(self.gen)][()].T
                to_cache += [new]
                N += new.shape[0]
            except StopIteration:
                break
        if N > 0:
            self.cache = self.collate_fn(to_cache).to(self.device)
        return N > 0

    def __iter__(self):
        keep_on = self.init_sampling()
        N = 0
        while keep_on:
            # print("updated cache to array of shape", self.cache.shape)
            iterator = DataLoader(self.cache, batch_size=self.batch_size, shuffle=True)
            for x in iterator:
                N += x.shape[0]
                yield x
            self.cache = []
            keep_on = self._refill_cache()
        print("yielded", N, "datapoints")

    def to_numpy(self):
        rv = []
        for ds in self.ds.keys():
            rv += [self[ds.value]]
        return np.concatenate(rv, axis=0)


class SegmentDB(HdBase):

    @staticmethod
    def collate_fn(alist):
        return pack_sequence([from_numpy(abs(x)) for x in alist])

    compression_args = dict(compression="gzip", compression_opts=4)

    @staticmethod
    def data_transform(abs_path, **kwargs):
        fft = FFT.stft(abs_path)
        chroma = librosa.feature.chroma_stft(S=fft, hop_length=fft.hop_length)
        slices = SegmentList(fft.abs, **kwargs).slices
        print("done with", abs_path)
        return fft.abs, chroma, np.array([s[0] for s in slices] + [slices[-1][-1]])

    def pre_process(self, root_dir, rel_file_path):
        print("processing", os.path.split(rel_file_path)[-1])
        abs_path = root_dir + rel_file_path
        fft, chroma, slices = self.data_transform(abs_path)
        tmp_db = ".".join(abs_path.split(".")[:-1] + ["h5"])
        with h5py.File(tmp_db, "w") as f:
            f.attrs["name"] = rel_file_path
            f.create_dataset("fft", shape=fft.shape, data=fft)
            f.create_dataset("chroma", shape=chroma.shape, data=chroma)
            f.create_dataset("segments", shape=slices.shape, data=slices)
        f.close()
        return tmp_db

    def make(self, root_directory,
             n_cores=cpu_count()):
        target_file = self.h5_file
        root_name, tree = audio_fs_dict(root_directory)
        print(root_name)
        args = [(root_directory, os.path.join(os.path.relpath(dir, root_directory), file))
                for dir, files in tree.items() for file in files]
        with Pool(n_cores) as p:
            tmp_dbs = p.starmap(self.pre_process, args)
        print("getting metadatas")
        metadata = []
        i = 0
        F, C = None, None
        f_dtype, c_dtype = None, None
        for file_idx, db in enumerate(tmp_dbs):
            db = h5py.File(db, "r")
            file_name = db.attrs["name"]
            ln = db["fft"].shape[1]
            if F is None:
                F, C = db["fft"].shape[0], db["chroma"].shape[0]
                f_dtype, c_dtype = db["fft"].dtype, db["chroma"].dtype
            segs = list(zip(db["segments"][:-1], np.diff(db["segments"][()])))
            # (file, global_index, ordinal_index, duration)
            metadata += [(file_name, False, file_idx, i, i+ln, ln)]
            metadata += [(file_name, True, ord_index, index+i, index+i+dur, dur) for ord_index, (index, dur) in enumerate(segs)]
            i += ln
            db.close()
        metadata = pd.DataFrame(metadata, columns=["name", "is_seg", "idx", "start", "stop", "dur"])
        print("saving results in .h5 file")
        f = h5py.File(target_file, "w")
        metadata.to_hdf(target_file, key="metadata", mode="r+")
        fft = f.create_dataset("fft", shape=(i, F), dtype=f_dtype, **self.compression_args)
        chroma = f.create_dataset("chroma", shape=(i, C), dtype=c_dtype, **self.compression_args)
        i = 0
        for db in tmp_dbs:
            db = h5py.File(db, "r")
            ln = db["fft"].shape[1]
            fft[i:i+ln] = db["fft"][()].T
            chroma[i:i+ln] = db["chroma"][()].T
            i += ln
            name = db.filename
            db.close()
            os.remove(name)
            f.flush()
        f.close()
        print("Done")
        return HdBase(target_file)

    def __getitem__(self, args):
        """
        overrides the parent's method in order to get items by reference
        """
        f, ds = args

        def slice_or_array(obj, name):
            if len(obj.shape) == 1:  # it's a slice!
                origin = name.split("/_s")[0] + "/frames"
                return f[origin][:, obj[()]]
            else:  # it's a 2d-array
                return obj

        if isinstance(ds, str):
            return slice_or_array(f[ds], ds)
        elif isinstance(ds, re.Pattern):
            return [slice_or_array(f[k], k) for k in self.ds.keys() if ds.search(k) is not None]
        else:
            raise ValueError("method __getitem__ not implemented for the type `%s`" % str(type(ds)))


class NeighborsForest(HdBase):
    compression_args = dict()

    estimator = KNeighborsTransformer()

    @staticmethod
    def data_transform(abs_path):
        fft = FFT.stft(abs_path)
        estimator = deepcopy(NeighborsForest.estimator)
        estimator.fit(fft)
        return np.array(pickle.dumps(estimator), dtype=np.bytes_)

    def __init__(self, h5_file):
        super(NeighborsForest, self).__init__(h5_file)

    def _kneighbors(self, X, name, k, return_distances=True):
        print("getting neighbors from", os.path.split(name)[-1])
        with h5py.File(self.h5_file, "r") as f:
            estimator = pickle.loads(f[name][()])
        return estimator.kneighbors(X, k, return_distances)

    def kneighbors(self, X, k, return_distances=True, n_cores=cpu_count()):
        args = [(X, ds, k, return_distances) for ds in self.ds.keys()]
        with Pool(n_cores) as p:
            results = p.starmap(self._kneighbors, args)
        return {ds: res for ds, res in zip(self.ds.keys(), results)}

    def fetch(self, name, indices):
        with h5py.File(self.h5_file, "r") as f:
            estimator = pickle.loads(f[name][()])
            return estimator._fit_X[indices]
