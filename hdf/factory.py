import h5py
import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count, Pool
from cafca.util import audio_fs_dict
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# TODO : add handling of sparse features ?


def _file_to_db(abs_path, extract_func):
    logger.info("making temp db for %s" % abs_path)
    features = {}
    tmp_db = ".".join(abs_path.split(".")[:-1] + ["h5"])
    with h5py.File(tmp_db, "w") as f:
        f.attrs["path"] = abs_path
        rv = extract_func(abs_path)
        for name, (attrs, data) in rv.items():
            ds = f.create_dataset(name=name, shape=data.shape, data=data)
            ds.attrs.update(attrs)
            features[name] = (ds.dtype, *ds.shape)
            f.flush()
    f.close()
    return tmp_db, features


def _make_temp_dbs(root_directory,
                   extract_func,
                   n_cores=cpu_count()):
    root_name, tree = audio_fs_dict(root_directory)
    args = [(os.path.join(dir, file), extract_func)
            for dir, files in tree.items() for file in files]
    with Pool(n_cores) as p:
        tmp_dbs = p.starmap(_file_to_db, args)
    return pd.DataFrame.from_dict(dict(tmp_dbs), orient="index")


def _collect_segments_metadata(metadata):
    logger.info("collecting segments' metadata")
    files = list(metadata.index)
    frames = []
    for file in files:
        with h5py.File(file, "r") as f:
            segments = f["segments"][()]
            # last elements in segments should always be the length of the whole segmented data
            durations = np.diff(segments)
            index = pd.MultiIndex.from_product([[file], range(len(durations))], names=["file", "index"])
            # (start, stop, duration)
            data = list(zip(range(len(durations)), segments[:-1], np.cumsum(durations), durations))
            frames += [pd.DataFrame(data, index=index, columns=["index", "start", "stop", "duration"])]
            f.close()
    return pd.concat(frames, sort=False)


def _aggregate_from_metadata(target_file, metadata, exclude=None, **kwargs):
    if exclude is None:
        exclude = set()
    features = [col for col in metadata.columns if col not in exclude]
    metas = {feature: (list(set([meta[0] for meta in metadata[feature]]))[0],
                       np.array([meta[1:] for meta in metadata[feature]]))
             for feature in features}
    with h5py.File(target_file, "w") as f:
        f.attrs["features"] = list(metas.keys())
        for feature, (dtype, shapes) in metas.items():
            feature_meta = {}
            logger.info("copying %s" % feature)
            assert np.all(shapes[:, 1:] == shapes[0, 1:])
            offsets = np.cumsum(shapes[:, 0])
            total_shape = (offsets[-1], *shapes[0, 1:])
            offsets = np.r_[0, offsets[:-1]]
            feature_ds = f.create_dataset(feature + "/data", dtype=dtype, shape=total_shape, **kwargs)
            attrs = set()  # all the tmp_dbs should have the same attrs, that's why we only keep track of unique items
            for i, file in zip(range(len(offsets)), metadata.index):
                sub_db = h5py.File(file, "r")
                start, stop = offsets[i], offsets[i] + shapes[i, 0]
                feature_ds[start:stop] = sub_db[feature][()]
                attrs = attrs.union(set(sub_db[feature].attrs.items()))
                sub_db.close()
                f.flush()
                # {file: {feature: (start, stop, dur)}}
                file = file.strip(".h5")
                feature_meta[file] = dict(index=i, start=start, stop=stop, duration=shapes[i, 0])
            feature_meta = pd.DataFrame.from_dict(feature_meta, orient="index")
            feature_meta.to_hdf(target_file, feature+"/metadata", "r+")
        for file in metadata.index:
            os.remove(file)

    f.close()
    return None


def _add_metadata(db_path, key, metadata):
    metadata.to_hdf(db_path, key=key + "/metadata", mode="r+")
    return None


def db_factory(root_directory, target_file, extract_func, n_cores=cpu_count(), segmented=True, **kwargs):
    logger.info("storing features in temp dbs")
    tmp_metadata = _make_temp_dbs(root_directory, extract_func, n_cores)

    segments_metadata = _collect_segments_metadata(tmp_metadata) if segmented else None

    logger.info("copying temp dbs to target file")
    _aggregate_from_metadata(target_file, tmp_metadata, ["segments"] if segmented else None, **kwargs)

    if segments_metadata is not None:
        _add_metadata(target_file, "segments", segments_metadata)

    logger.info("done! following Groups and Datasets are now stored in '{}':".format(target_file))
    h5py.File(target_file, "r").visititems(lambda x, obj: logger.info("name={} ; object={}".format(x, str(obj))))
    return target_file


