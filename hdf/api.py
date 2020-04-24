import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from cafca.extract.similarity import segments_sim
from cafca.hdf.iter_utils import ssts, dts, irbod
from cafca.extract.utils import reduce_2d
from torch.utils.data import DataLoader, Dataset
import re


class FeatureProxy(Dataset):
    def __init__(self, h5_file, ds_name, parent=None):
        self.h5_file = h5_file
        self.name = ds_name
        for attr in parent.__dict__:
            if "_m" in attr:
                setattr(self, attr, getattr(parent, attr))
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
        """
        set the time-level of reference from which to get or iterate data and metadata
        @param level: "files", "segs", or "frames"
        @return: self
        this makes lines such as :
        `for x in db.feature("frames").iterate()`
        quite handy...
        """
        if level == "files":
            self.iter_meta = self.meta
        elif level in "segs":
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

    @staticmethod
    def _gen_slices(item):
        for _, loc in item.iterrows():
            yield slice(loc["start"], loc["stop"])

    def gen_item(self, item):
        """
        core of __getitem__ that yields a generator where each next element is the data corresponding
        to (the elements of) `item`.
        @param item: different types correspond to different behaviors :
            - int : get the data for the metadata element at this global index
                    (roughly equivalent to data[metadata.iloc[int]] )
            - str and re.Pattern : get all the data corresponding to metadata rows where
                    "file" or "directory" contains the string (or regex)
            - iterables of type tuple, list, slice and np.ndarray : pass it directly to the h5py.Dataset of the feature
                    i.e. dataset[iterable]
            - boolean pd.Series are understood as implicitly be meant for the metadata dataframe,
                    and gen_item returns the data of the rows where the Series is True.
                    i.e. it returns `dataset[metadata[bool_series]]` and allows to shorten the query
                    db.feature[db.feature.meta[db.feature.meta["start"] > 100]]
                    to
                    db.feature[db.feature.meta["start"] > 100].
            - any pd.DataFrame containing the columns "start" and "stop" will also work and be interpreted as slices.
                    the returned value is then : `(df.iloc[i]["start"]:df.iloc[i]["stop"] for i in range(len(df)))`
        @return: generator yielding the slice of data corresponding to (the elements of) `item`
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
        """
        c.f. `gen_item(item)`
        @param item:
        @return:
        """
        rv = [res for res in self.gen_item(item)]
        if len(rv) <= 1:
            return rv[0]
        return rv

    def iterate(self, level, mode="length", **kwargs):
        """

        @param level:
        @param mode:
        @param kwargs:
        @return: data
        """
        if mode == "duration":
            self(level)
            batches = irbod(self.iter_meta,
                            kwargs.get("batch_size", 1),
                            kwargs.get("shuffle", False),
                            kwargs.get("drop_last", True))
            kwargs["collate_fn"] = kwargs.get("collate_fn", list)
            for idx_batch in batches:
                yield kwargs["collate_fn"](self[b] for b in idx_batch)
        else:
            return DataLoader(self(level), **kwargs)

    def match(self, item):
        """

        @param item:
        @return: metadata
        """
        item_ = re.compile(item) if type(item) is str else item
        has_name = self.iter_meta["name"].str.contains(item_)
        has_dir = self.iter_meta["directory"].str.contains(item_)
        return self.iter_meta[has_dir | has_name]

    def data_and_splits(self, item):
        return np.concatenate(self[item]), dts(item)

    def neighbors(self, item_x, item_y=None, param=1, mode="best",
                  n_jobs=cpu_count()):
        """

        @param item_x:
        @param item_y:
        @param param:
        @param mode:
        @param n_jobs:
        @return: metadata
        """
        if item_y is None:
            item_y = item_x

        splits_x, splits_y = dts(item_x), dts(item_y)
        X, Y = np.concatenate(self[item_x]), np.concatenate(self[item_y])

        if self.level != "frames":
            segments_sim(X, splits_x, Y, splits_y, param, mode, n_jobs=n_jobs)


class Database(object):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, "r") as f:
            # add found features as self.feature_name = FeatureProxy(self, feature_name, self.segs)
            f.visit(self._register_features())
        self.meta = self._get_metadata()

    def _get_metadata(self):
        return pd.read_hdf(self.h5_file, key="/meta")

    def save_metadata(self):
        with h5py.File(self.h5_file, "r+") as f:
            del f["/meta"]
        self.meta.to_hdf(self.h5_file, key="/meta", mode="r+")
        return self._get_metadata()

    def _register_features(self):
        file = self.h5_file

        def register(name):
            if "_m" in name and "/" not in name:
                setattr(self, name, pd.read_hdf(file, key=name, mode="r"))
                return None
            if "/data" in name:
                name = name.strip("data")
                setattr(self, name.strip("/"), FeatureProxy(self.h5_file, name, self))
                return None
            return None

        return register

    def add_feature(self, name, transform, own_feature):
        # TODO
        pass
