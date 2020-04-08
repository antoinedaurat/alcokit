import h5py
import os
import pickle
import re
from copy import deepcopy
from functools import wraps
from sklearn.neighbors import KNeighborsTransformer
from cafca.util import is_audio_file
from cafca.preprocessing import stft
from multiprocessing import cpu_count, Pool
import numpy as np
from torch.utils.data import DataLoader
from torch import from_numpy, cuda, device as torch_device
import warnings

warnings.filterwarnings("ignore")

'''
@todo : never forget : "TypeError: h5py objects cannot be pickled"
'''


def audio_fs_dict(root):
    root_name = os.path.split(root.strip("/"))[-1]
    items = [(d, list(filter(is_audio_file, f))) for d, _, f in os.walk(root)]
    return root_name, dict(item for item in items if len(item[1]) > 0)


def with_db_read(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(args[0])
        args[0].f = h5py.File(args[0].h5_file, "r")
        result = f(*args, **kwargs)
        print(result)
        args[0].f.close()
        return result
    return wrapper


class HdBase(object):

    compression_args = dict(compression="gzip", compression_opts=4)

    @staticmethod
    def data_transform(abs_path):
        return stft(abs_path)

    def pre_process(self, root_dir, rel_file_path):
        print("processing", os.path.split(rel_file_path)[-1])
        abs_path = root_dir + rel_file_path
        data = self.data_transform(abs_path)
        tmp_db = ".".join(abs_path.split(".")[:-1] + ["h5"])
        with h5py.File(tmp_db, "w") as f:
            f.create_dataset(rel_file_path, shape=data.shape if data.shape else (1,), data=data, **self.compression_args)
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
        root_group = f.create_group(root_name)
        for db in tmp_dbs:
            self.post_process(root_group, db)
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

    def __getitem__(self, ds):
        """
        returns the Datasets referenced or matched by `ds` in the h5py file
        @type ds: `str` or `re.Pattern`
        @return: the found Datasets if `ds` is a `str` and the list of the Datasets whose name matched `ds`
        if `ds` is a regex
        N.B. THE FILE HANDLE TO THE DB MUST BE OPENED PRIOR TO CALLING THIS METHOD!
        """
        with h5py.File(self.h5_file, "r") as f:
            if isinstance(ds, str):
                return f[ds]
            elif isinstance(ds, re.Pattern):
                return [f[k] for k in self.ds.keys() if ds.match(k) is not None]
            else:
                raise ValueError("method __getitem__ not implemented for the type `%s`" % str(type(ds)))

    def __iter__(self):
        """
        @yields: the Datasets stored in the h5py file
        """
        for ds in self.ds.keys():
            yield self[ds]


class NeighborsForest(HdBase):

    compression_args = dict()

    estimator = KNeighborsTransformer()

    @staticmethod
    def data_transform(abs_path):
        fft = stft(abs_path)
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


class HdfDB(HdBase):

    def __init__(self, h5_file, batch_size, cache_size, device=None):
        super(HdfDB, self).__init__(h5_file)
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.ds_gen = None
        self.cache = None
        self.device = device if device is not None else torch_device('cuda' if cuda.is_available() else 'cpu')
        self.init_sampling()
        self.bpe = self.N // batch_size
        print("ready to yield", self.bpe, "batches for", self.N, "datapoints")

    def size(self):
        return self.cache.size()

    def init_sampling(self):

        def ds_gen():
            keys = list(self.ds.keys())
            for i in np.random.permutation(np.arange(len(keys))):
                yield keys[i]

        self.ds_gen = ds_gen()
        return self._refill_cache()

    def _refill_cache(self):
        N, i = 0, 0
        to_cache = []
        while N < self.cache_size:
            try:
                new = self[next(self.ds_gen)].value
                to_cache += [new]
                N += new.shape[0]
            except StopIteration:
                break
        if N > 0:
            self.cache = from_numpy(np.concatenate(to_cache, axis=0)).to(self.device)
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



