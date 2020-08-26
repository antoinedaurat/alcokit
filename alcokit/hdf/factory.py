import h5py
import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count, Pool
from alcokit.util import fs_dict, is_audio_file
from alcokit.hdf.api import Database
from alcokit.score import Score
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# TODO : add handling of sparse matrices ?

def default_extract_func(abs_path):
    from alcokit.fft import FFT
    fft = abs(FFT.stft(abs_path))
    score = Score.from_recurrence_matrix(fft)
    return dict(fft=({}, fft.T), score=({}, score))


def sizeof_fmt(num, suffix='b'):
    """
    straight from https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    """
    for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def _empty_info(features_names):
    tuples = [("directory", ""), ("name", ""),
              *[t for feat in features_names for t in [(feat, "dtype"), (feat, "shape"), (feat, "size")]
                if feat != "score"]
              ]
    idx = pd.MultiIndex.from_tuples(tuples)
    return pd.DataFrame([], columns=idx)


def split_path(path):
    parts = path.split("/")
    prefix, file_name = "/".join(parts[:-1]), parts[-1]
    return prefix, file_name


def file_to_db(abs_path, extract_func=default_extract_func, mode="w"):
    """
    if mode == "r+" this will either:
        - raise an Exception if the feature already exists
        - concatenate data along the "feature_axis", assuming that each feature correspond to the same file
          or file collections.
          If you want to concatenate dbs along the "file_axis" consider using `concatenate_dbs(..)`
    @param abs_path:
    @param extract_func:
    @param mode:
    @return:
    """
    logger.info("making db for %s" % abs_path)
    tmp_db = ".".join(abs_path.split(".")[:-1] + ["h5"])
    rv = extract_func(abs_path)
    info = _empty_info(rv.keys())
    info.loc[0, [("directory", ""), ("name", "")]] = split_path(abs_path)
    with h5py.File(tmp_db, mode) as f:
        for name, (attrs, data) in rv.items():
            if issubclass(type(data), np.ndarray):
                ds = f.create_dataset(name=name, shape=data.shape, data=data)
                ds.attrs.update(attrs)
                info.loc[0, name] = ds.dtype, ds.shape, sizeof_fmt(data.nbytes)
            elif issubclass(type(data), pd.DataFrame):
                pd.DataFrame(data).to_hdf(tmp_db, name, "r+")
            f.flush()
        if "info" in f.keys():
            prior = pd.read_hdf(tmp_db, "info", "r")
            info = pd.concat((prior, info.iloc[:, 2:]), axis=1)
        info.to_hdf(tmp_db, "info", "r+")
    f.close()
    return tmp_db


def make_db_for_each_file(root_directory,
                          extract_func=default_extract_func,
                          extension_filter=is_audio_file,
                          n_cores=cpu_count()):
    root_name, tree = fs_dict(root_directory, extension_filter)
    args = [(os.path.join(dir, file), extract_func)
            for dir, files in tree.items() for file in files]
    with Pool(n_cores) as p:
        tmp_dbs = p.starmap(file_to_db, args)
    return tmp_dbs


def collect_infos(tmp_dbs):
    infos = []
    for db in tmp_dbs:
        infos += [Database(db).info]
    return pd.concat(infos, ignore_index=True)


def collect_scores(tmp_dbs):
    scores = []
    offset = 0
    for db in tmp_dbs:
        scr = Database(db).score
        scr.loc[:, ("start", "stop")] = scr.loc[:, ("start", "stop")].values + offset
        scr.loc[:, "name"] = ".".join(db.split(".")[:-1])
        scores += [scr]
        offset = scr.last_stop
    return pd.DataFrame(pd.concat(scores, ignore_index=True))


def zip_prev_next(iterable):
    return zip(iterable[:-1], iterable[1:])


def ds_definitions_from_infos(infos):
    tb = infos.iloc[:, 2:].T
    paths = ["/".join(parts) for parts in infos.iloc[:, :2].values]
    # change the paths' extensions
    paths = [".".join(path.split(".")[:-1]) + ".h5" for path in paths]
    features = set(tb.index.get_level_values(0))
    ds_definitions = {}
    for f in features:
        dtype = tb.loc[(f, "dtype"), :].unique().item()
        shapes = tb.loc[(f, "shape"), :].values
        dims = shapes[0][1:]
        assert all(shp[1:] == dims for shp in
                   shapes[1:]), "all features should have the same dimensions but for the first axis"
        layout = Score.from_duration([s[0] for s in shapes])
        ds_shape = (layout.last_stop, *dims)
        layout.index = paths
        ds_definitions[f] = {"shape": ds_shape, "dtype": dtype, "layout": layout}
    return ds_definitions


def create_datasets_from_defs(target, defs, mode="w"):
    with h5py.File(target, mode) as f:
        for name, params in defs.items():
            f.create_dataset(name, shape=params["shape"], dtype=params["dtype"])
            layout = params["layout"]
            layout.reset_index(drop=False, inplace=True)
            layout = layout.rename(columns={"index": "name"})
            pd.DataFrame(layout).to_hdf(target, "layouts/" + name, "r+")
        f.close()
    return


def make_integration_args(target):
    args = []
    with h5py.File(target, "r") as f:
        for feature in f["layouts"].keys():
            df = Score(pd.read_hdf(target, "layouts/" + feature))
            args += [(target, source, feature, indices) for source, indices in
                     zip(df.name, df.slices(time_axis=0))]
    return args


def integrate(target, source, key, indices):
    with h5py.File(source, "r") as src:
        data = src[key][()]
    with h5py.File(target, "r+") as trgt:
        trgt[key][indices] = data
    return


def aggregate_dbs(target, dbs, mode="w", remove_sources=False):
    infos = collect_infos(dbs)
    score = collect_scores(dbs)
    definitions = ds_definitions_from_infos(infos)
    create_datasets_from_defs(target, definitions, mode)
    args = make_integration_args(target)
    for arg in args: integrate(*arg)
    if remove_sources:
        for src in dbs: os.remove(src)
    infos.to_hdf(target, "info", "r+")
    score.to_hdf(target, "score", "r+")


def make_root_db(db_name, root_directory, extension_filter, extract_func=default_extract_func, n_cores=cpu_count(), remove_sources=True):
    dbs = make_db_for_each_file(root_directory, extract_func, extension_filter, n_cores)
    aggregate_dbs(db_name, dbs, "w", remove_sources)