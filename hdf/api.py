import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from cafca.extract.similarity import cos_sim_graph
from cafca.hdf.iter_utils import ssts, dts, reduce_2d, irbod, ibod
from torch.utils.data import DataLoader, Dataset
import re


class FeatureProxy(Dataset):
    def __init__(self, h5_file, ds_name, segments=None):
        self.h5_file = h5_file
        self.name = ds_name
        self.segs = segments
        self.meta = self._get_metadata()
        self.iter_meta = self.meta
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name + "/data"]
            attrs = {k: v for k, v in ds.attrs.items()}
        self.attrs = attrs
        self.N = self.meta["duration"].sum()
        self.level = "file"

    def __len__(self):
        return self.N

    def __call__(self, level):
        if level == "files":
            self.iter_meta = self.meta
        elif level == "segs":
            self.iter_meta = self.segs
        elif level == "frames":
            rg = np.arange(self.N)
            self.iter_meta = pd.DataFrame(np.stack((rg, rg + 1)).T, columns=["start", "stop"])
        self.level = level
        return self

    def _get_metadata(self):
        return pd.read_hdf(self.h5_file, key=self.name + "/metadata")

    def save_metadata(self):
        with h5py.File(self.h5_file, "r+") as f:
            del f[self.name + "/metadata"]
        self.meta.to_hdf(self.h5_file, key=self.name + "/metadata", mode="r+")
        return self._get_metadata()

    def match(self, item):
        item_ = re.compile(item) if type(item) is str else item
        has_name = self.iter_meta["name"].str.contains(item_)
        has_dir = self.iter_meta["directory"].str.contains(item_)
        return self.iter_meta[has_dir | has_name]

    @staticmethod
    def _gen_slices(item):
        for _, loc in item.iterrows():
            yield slice(loc["start"], loc["stop"])

    def gen_item(self, item):
        """
        """
        if type(item) is int:
            # get the data for this iloc
            with h5py.File(self.h5_file, "r") as f:
                series = self.iter_meta.iloc[item]
                yield f[self.name + "/data"][series["start"]:series["stop"]]
        elif type(item) in (str, re.Pattern):
            # find matching locs in the columns "directory' and 'name'
            matches = self.match(item)
            if not matches.empty:
                items = ssts(matches)
                with h5py.File(self.h5_file, "r") as f:
                    for item_ in items:
                        yield f[self.name + "/data"][item_]
            else:
                raise ValueError("no match found in the columns 'directory' and 'name' for '%s'" % item)
        elif type(item) in (tuple, list, slice, np.ndarray):
            # straight to dataset
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
                    items = ssts(self.iter_meta[item])
                with h5py.File(self.h5_file, "r") as f:
                    for item in items:
                        yield f[self.name + "/data"][item]
            else:  # TODO : this could be the .index of self.meta or self.segs !
                raise ValueError("pd.Series passed to `gen_item` should be of dtype=np.bool")
        else:
            raise ValueError("type of item passed to `gen_item` not recognised: {}".format(type(item)))

    def __getitem__(self, item):
        rv = [res for res in self.gen_item(item)]
        if len(rv) <= 1:
            return rv[0]
        return rv

    def iterate(self, level, mode="length", **kwargs):
        if mode == "duration":
            self(level)
            batches = irbod(self.iter_meta,
                            kwargs.get("batch_size", 1),
                            kwargs.get("shuffle", False),
                            kwargs.get("drop_last", True))
            kwargs["collate_fn"] = kwargs.get("collate_fn", list)
            for idx_batch in batches:
                print(idx_batch)
                yield kwargs["collate_fn"](self[b] for b in idx_batch)
        else:
            return DataLoader(self(level), **kwargs)

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
