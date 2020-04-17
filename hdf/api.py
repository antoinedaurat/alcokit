import h5py
import numpy as np
import pandas as pd


class FeatureProxy(object):
    def __init__(self, db_path, ds_name, segments=None):
        self.h5_file = db_path
        self.name = ds_name
        self.segs = segments
        self.meta = self._get_metadata()

    def __getitem__(self, item):
        if type(item) in (int, tuple, list, slice, np.ndarray):
            with h5py.File(self.h5_file) as f:
                rv = f[self.name + "/data"][item]
            return rv
        elif isinstance(item, pd.DataFrame):
            items = self._to_slice(item)
            with h5py.File(self.h5_file) as f:
                rv = [f[self.name + "/data"][item] for item in items]
            return rv
        elif isinstance(item, pd.Series):
            if item.dtype == np.bool:
                if item.index[0] == self.segs.index[0]:
                    items = self._to_slice(self.segs[item])
                else:
                    items = self._to_slice(self.meta[item])
                with h5py.File(self.h5_file) as f:
                    rv = [f[self.name + "/data"][item] for item in items]
                return rv
            else:
                raise ValueError("pd.Series passed to __getitem__ should be of dtype=np.bool")
        else:
            raise ValueError("type of item passed to `__getitem__` not recognised: {}".format(type(item)))

    def _get_metadata(self):
        return pd.read_hdf(self.h5_file, key=self.name + "/metadata")

    @staticmethod
    def _to_slice(item):
        slices = []
        for _, loc in item.iterrows():
            slices += [slice(loc["start"], loc["stop"])]
        return slices


class Database(object):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        try:
            self.segs = pd.read_hdf(self.h5_file, key="segments/metadata")
        except KeyError:
            self.segs = None
        f = h5py.File(h5_file, "r")
        f.visit(self._register_features())

    def _register_features(self):
        def register(name):
            if "/data" in name:
                name = name.strip("data")
                setattr(self, name.strip("/"), FeatureProxy(self.h5_file, name, self.segs))
                return None
            return None

        return register