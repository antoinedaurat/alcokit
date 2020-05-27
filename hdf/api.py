import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from cafca.extract.similarity import segments_sim, sort_graph
from cafca.hdf.iter_utils import ssts, dts, irbod
from torch.utils.data import DataLoader, Dataset
import re


def is_m_frame(name):
    return re.search(re.compile(r"_m$"), name) is not None


class Axis0Meta(object):
    """
    fake m_frame to pass ints, slices etc directly to the .h5 Datasets when setting the reference m_frame to "frames"
    """

    def __init__(self, h5_file, ds_name, transposed=True):
        self.h5_file = h5_file
        self.name = ds_name
        self.T = transposed
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name + "/data"]
            attrs = {k: v for k, v in ds.attrs.items()}
            self.N = ds.shape[0]
        self.attrs = attrs

    def __getitem__(self, item):
        with h5py.File(self.h5_file, "r") as f:
            rv = f[self.name + "/data"][item]
        return rv.T if self.T else rv

    def __len__(self):
        return self.N


class FeatureProxy(Dataset):
    def __init__(self, h5_file, ds_name, parent=None, transposed=True):
        self.h5_file = h5_file
        self.name = ds_name
        self.axis0_m = Axis0Meta(self.h5_file, self.name, transposed)
        self.meta_m = None
        self.T = transposed
        if parent is not None:
            for attr in parent.__dict__:
                if is_m_frame(attr):
                    setattr(self, attr, getattr(parent, attr))
        self.iter_meta = self.meta_m
        self.level = "meta_m"
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name + "/data"]
            self.N = ds.shape[0]
            self.attrs = {k: v for k, v in ds.attrs.items()}
            self.has_graph = "graph" in f[self.name].keys()

    @property
    def graph(self):
        if self.has_graph:
            with h5py.File(self.h5_file, "r") as f:
                G = f[self.name + "/graph"][()]
            return G
        return None

    def __len__(self):
        return self.N

    def __call__(self, level):
        """
        set the time-level of reference from which to get or iterate data and metadata
        @param level: string of the name of the m_frame to set as default
        @return: self
        this makes lines such as :
        `for x in db.feature("frames").iterate()`
        quite handy...
        """
        meta = getattr(self, level, None)
        if meta is None:
            raise ValueError("no m_frame found for level name '%s'" % level)
        self.iter_meta = meta
        self.level = level
        return self

    def _get_metadata(self):
        return pd.read_hdf(self.h5_file, key=self.name + "/meta_m")

    def save_metadata(self):
        with h5py.File(self.h5_file, "r+") as f:
            del f[self.name + "/metadata"]
        self.meta_m.to_hdf(self.h5_file, key=self.name + "/metadata", mode="r+")
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
            - boolean pd.Series are understood as implicitly be meant for the current m_frame
                    (which is set by calling db.feature("m_frame")
                    `gen_item` returns the data of the rows where the Series is True.
                    i.e., once you called `db.feature("meta"), it returns `dataset[meta[bool_series]]`
                    and allows to shorten the query
                    `db.feature[db.feature.meta[db.feature.meta["start"] > 100]]`
                    to
                    `db.feature[db.feature.meta["start"] > 100]`.
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
            slices = ssts(self.iter_meta.iloc[item])
            with h5py.File(self.h5_file, "r") as f:
                for slice_i in slices:
                    yield f[self.name + "/data"][slice_i]
        elif isinstance(item, pd.DataFrame):
            items = ssts(item)
            with h5py.File(self.h5_file, "r") as f:
                for item in items:
                    yield f[self.name + "/data"][item]
        elif isinstance(item, pd.Series):
            if item.dtype == np.bool:
                items = ssts(self.iter_meta[item])
                with h5py.File(self.h5_file, "r") as f:
                    for item in items:
                        yield f[self.name + "/data"][item]
            elif "start" in item.index and "stop" in item.index:  # SINGLE DF ROW
                with h5py.File(self.h5_file, "r") as f:
                    yield f[self.name + "/data"][item.start:item.stop]
            else:  # TODO : this could be the .index of self.meta or self.segs !
                raise ValueError("pd.Series passed to `gen_item` should be of dtype=np.bool"
                                 " or have 'start' and 'stop' in its index")
        else:
            raise ValueError("type of item passed to `gen_item` not recognised: {}".format(type(item)))

    def __getitem__(self, item):
        """
        c.f. `gen_item(item)`
        @param item:
        @return:
        """
        if self.T:
            rv = [res.T for res in self.gen_item(item)]
        else:
            rv = [res for res in self.gen_item(item)]
        if len(rv) <= 1 and isinstance(item, int):
            # strip the list for DataLoader and co
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
                  metric="cosine", reduce_func=np.mean,
                  batch_size=500,
                  n_cores=cpu_count(), return_graph=False):

        if item_y is None:
            item_y = item_x

        def check_args(item_):
            if isinstance(item_, tuple) and isinstance(item_[0], FeatureProxy) and isinstance(item_[1], pd.DataFrame):
                feat, item = item_
            elif isinstance(item_, pd.DataFrame):
                feat, item = self, item_
            else:
                raise ValueError
            return feat, item

        feat_x, item_x = check_args(item_x)
        feat_y, item_y = check_args(item_y)

        if self.has_graph and metric == "cosine":
            G = self.graph[item_x.index.values]
            G = G[:, item_y.index.values]
            locs = sort_graph(G, mode, param)
            rv = (locs, G) if return_graph else locs
        else:
            rv = segments_sim(feat_x, item_x, feat_y, item_y,
                              param, mode, metric, reduce_func,
                              batch_size, return_graph, n_cores)
        if return_graph:
            locs, graph = rv
            return [item_y.iloc[loc] for loc in locs], graph
        locs = rv
        return [item_y.iloc[loc] for loc in locs]


class Database(object):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.info = self._get_metadata("/info")
        self.meta_m = None
        with h5py.File(h5_file, "r") as f:
            # add found features as self.feature_name = FeatureProxy(self, feature_name, self.segs)
            f.visit(self._register_m_frames())
            f.visit(self._register_features())

    def visit(self, func=print):
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _get_metadata(self, key):
        return pd.read_hdf(self.h5_file, key=key)

    def save_metadata(self, key, meta):
        with h5py.File(self.h5_file, "r+") as f:
            if key in f:
                f.pop(key)
        meta.to_hdf(self.h5_file, key=key, mode="r+")
        return self._get_metadata(key)

    def save_m_frame(self, frame, key):
        frame.to_hdf(self.h5_file, key=key, mode="r+")
        return pd.read_hdf(self.h5_file, key=key)

    def _register_features(self):
        def register(name):
            if "/data" in name:
                name = name.strip("data")
                setattr(self, name.strip("/"), FeatureProxy(self.h5_file, name.strip("/"), self))
                return None
            return None

        return register

    def _register_m_frames(self):
        def register(name):
            if is_m_frame(name) and getattr(self, name.split("/")[-1], None) is None:
                setattr(self, name.split("/")[-1], pd.read_hdf(self.h5_file, key=name, mode="r"))
                return None
            return None

        return register

    def add_feature(self, name, transform, own_feature):
        # TODO
        pass
