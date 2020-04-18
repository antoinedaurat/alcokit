import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from cafca.extract.similarity import cos_sim_graph
from cafca.hdf.iter_utils import ssts, dts, reduce_2d


class FeatureProxy(object):
    def __init__(self, h5_file, ds_name, segments=None):
        self.h5_file = h5_file
        self.name = ds_name
        self.segs = segments
        self.meta = self._get_metadata()
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name + "/data"]
            attrs = {k: v for k, v in ds.attrs.items()}
        self.attrs = attrs
        self.N = self.meta["duration"].sum()

    def __len__(self):
        return self.N

    def _get_metadata(self):
        return pd.read_hdf(self.h5_file, key=self.name + "/metadata")

    def save_metadata(self):
        # TODO
        pass

    @staticmethod
    def _gen_slices(item):
        for _, loc in item.iterrows():
            yield slice(loc["start"], loc["stop"])

    def gen_item(self, item):
        if type(item) in (int, tuple, list, slice, np.ndarray):
            with h5py.File(self.h5_file, "r") as f:
                yield f[self.name + "/data"][item]
        elif isinstance(item, pd.DataFrame):
            items = ssts(item)
            with h5py.File(self.h5_file, "r") as f:
                for item in items:
                    yield f[self.name + "/data"][item]
        elif isinstance(item, pd.Series):
            if item.dtype == np.bool:
                if item.index[0] == self.segs.index[0]:  # let's hope this is safe enough and not too much limiting...
                    items = ssts(self.segs[item])
                else:
                    items = ssts(self.meta[item])
                with h5py.File(self.h5_file, "r") as f:
                    for item in items:
                        yield f[self.name + "/data"][item]
            else:  # TODO : this could be the .index of self.meta or self.segs !
                raise ValueError("pd.Series passed to `gen_item` should be of dtype=np.bool")
        else:
            raise ValueError("type of item passed to `gen_item` not recognised: {}".format(type(item)))

    def __getitem__(self, item):
        return [res for res in self.gen_item(item)]

    def neighbors(self, item_x, item_y=None, param=None, mode="best",
                  n_jobs=cpu_count()):

        if item_y is None:
            # use the whole feature as y
            item_y = item_x

        splits_x, splits_y = dts(item_x), dts(item_y)
        X, Y = np.concatenate(self[item_x]), np.concatenate(self[item_y])

        G = cos_sim_graph(X, Y, param, mode, n_jobs)
        G = reduce_2d(G, splits_x, splits_y, np.mean, n_jobs)

        locs = None
        if mode == "best":
            idx = np.argsort(G, axis=1)[:, :param]
            locs = [item_y.iloc[neighbor] for neighbor in idx]
        elif mode == "radius":
            locs = []
            for row in G:
                idx = np.argsort(row)
                less_than = idx[row[idx] <= param]
                locs += [item_y.iloc[less_than]]
        return locs


class Database(object):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        try:
            self.segs = pd.read_hdf(self.h5_file, key="segments/metadata")
        except KeyError:
            self.segs = None
        with h5py.File(h5_file, "r") as f:
            # add found features as self.feature_name = FeatureProxy(self, feature_name, self.segs)
            f.visit(self._register_features())
        self.meta = self._get_metadata()

    def _get_metadata(self):
        return pd.read_hdf(self.h5_file, key="/meta")

    def _register_features(self):
        def register(name):
            if "/data" in name:
                name = name.strip("data")
                setattr(self, name.strip("/"), FeatureProxy(self.h5_file, name, self.segs))
                return None
            return None

        return register

    def add_feature(self, name, transform, own_feature):
        # TODO
        pass
