import h5py
import numpy as np
import pandas as pd
from alcokit.score import Score
from alcokit.sources import ScoredSourceMixin, NPSource
from functools import partial


class FeatureProxy(ScoredSourceMixin):

    time_axis = 0

    @property
    def _constructor(self):
        return NPSource

    @property
    def cat(self):
        # getitem transposes !!
        return lambda x: self._constructor(np.concatenate(x, axis=1))

    @property
    def stack(self):
        return np.stack

    def __init__(self, h5_file, ds_name, transposed=True):
        self.h5_file = h5_file
        self.name = ds_name
        self.T = transposed
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name]
            self.N = ds.shape[0]
            self.attrs = {k: v for k, v in ds.attrs.items()}

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        with h5py.File(self.h5_file, "r") as f:
            rv = f[self.name][item]
        return rv.T if self.T else rv


class Database(object):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.info = self._get_dataframe("/info")
        self.score = Score(self._get_dataframe("/score"))
        with h5py.File(h5_file, "r") as f:
            # add found features as self.feature_name = FeatureProxy(self, feature_name, self.segs)
            self._register_features(self.features)

    @property
    def features(self):
        names = self.info.iloc[:, 2:].T.index.get_level_values(0)
        return list(set(names))

    def visit(self, func=print):
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _get_dataframe(self, key):
        try:
            return pd.read_hdf(self.h5_file, key=key)
        except KeyError:
            return pd.DataFrame()

    def save_dataframe(self, key, df):
        with h5py.File(self.h5_file, "r+") as f:
            if key in f:
                f.pop(key)
        df.to_hdf(self.h5_file, key=key, mode="r+")
        return self._get_dataframe(key)

    def layout_for(self, feature):
        with h5py.File(self.h5_file, "r") as f:
            if "layouts" not in f.keys():
                return pd.DataFrame()
        return self._get_dataframe("layouts/" + feature)

    def _register_features(self, names):
        for name in names:
            setattr(self, name, FeatureProxy(self.h5_file, name))
        return None



