import h5py
import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count, Pool
from cafca.util import audio_fs_dict
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# TODO : add handling of sparse matrices ?


def sizeof_fmt(num, suffix='b'):
    """
    straight from https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    """
    for unit in ['','k','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


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
            features[name] = {"dtype": ds.dtype, "shape": ds.shape, "size": sizeof_fmt(data.nbytes)}
            f.flush()
    f.close()
    return tmp_db, features


def split_path(path):
    parts = path.split("/")
    prefix, file_name = "/".join(parts[:-1]), parts[-1]
    return prefix, file_name


def _make_temp_dbs(root_directory,
                   extract_func,
                   n_cores=cpu_count()):
    root_name, tree = audio_fs_dict(root_directory)
    args = [(os.path.join(dir, file), extract_func)
            for dir, files in tree.items() for file in files]
    with Pool(n_cores) as p:
        tmp_dbs = p.starmap(_file_to_db, args)
    df = pd.DataFrame.from_dict({split_path(file): {(f, k): val for f, d in features.items()
                                        for k, val in d.items()}
                                 for file, features in tmp_dbs},
                                orient="index")
    df = df.rename_axis(index=["directory", "name"])
    return df


def _collect_segments_metadata(metadata):
    logger.info("collecting segments' metadata")
    files = list(metadata.index)
    frames = []
    offset = 0
    for file in files:
        # join directory and file_name
        file = "/".join(file)
        with h5py.File(file, "r") as f:
            segments = f["segments"][()]
            # last elements in segments should always be the length of the whole segmented data
            durations = np.diff(segments)
            index = pd.MultiIndex.from_product([[file.strip(".h5")], range(len(durations))],
                                               names=["file", "index"])
            # get the metadata (file, index, start, stop, duration)
            data = list(zip(offset + segments[:-1], offset + np.cumsum(durations), durations))
            df = pd.DataFrame(data, index=index, columns=["start", "stop", "duration"])
            # add (file, index) as columns
            df = df.reset_index()
            frames += [df]
            # increment the start-index for the next file
            offset += segments[-1]
            f.close()
    return pd.concat(frames, ignore_index=True)


def _aggregate_from_metadata(target_file, metadata, mode="w", exclude=None, **kwargs):
    if exclude is None:
        exclude = set()
    features = set([col for col in metadata.T.index.get_level_values(0) if col not in exclude])
    with h5py.File(target_file, mode) as f:
        for feature in features:
            dtype = metadata[feature, "dtype"].unique().item()
            shapes = np.array([shape for shape in metadata[feature]["shape"]])
            meta = {}
            logger.info("copying %s" % feature)
            assert np.all(shapes[:, 1:] == shapes[0, 1:])
            # get the shape of the aggregated DS
            offsets = np.cumsum(shapes[:, 0])
            total_shape = (offsets[-1], *shapes[0, 1:])
            offsets = np.r_[0, offsets[:-1]]
            # create the ds
            feature_ds = f.create_dataset(feature + "/data", dtype=dtype, shape=total_shape, **kwargs)
            # all the tmp_dbs should have the same attrs so we only keep track of unique items
            attrs = set()
            # copy each file
            for i, file in zip(range(len(offsets)), metadata.index):
                # join directory and file_name
                file = "/".join(file)
                # get the source
                sub_db = h5py.File(file, "r+")
                # get the slice
                start, stop = offsets[i], offsets[i] + shapes[i, 0]
                # copy
                feature_ds[start:stop] = sub_db[feature][()]
                # intersect the attrs
                attrs = attrs.union(set(sub_db[feature].attrs.items()))
                # clean up
                del sub_db[feature]
                sub_db.flush()
                sub_db.close()
                f.flush()
                # store the metadata of this file
                file = split_path(file)
                meta[file] = dict(index=i, start=start, stop=stop, duration=shapes[i, 0])
            # add attrs to the DS
            for key, value in attrs:
                feature_ds.attrs[key] = value
            # this will be handy to figure out batch_size when iterating/querying etc.
            feature_ds.attrs["axis0_nbytes"] = feature_ds[0].nbytes
            f.flush()
            # store the metadata for the whole feature
            meta = pd.DataFrame.from_dict(meta, orient="index")
            meta = meta.rename_axis(index=["directory", "name"])
            meta.reset_index().to_hdf(target_file, feature+"/metadata", "r+")
        # clean up
        for file in metadata.index:
            # join directory and file_name
            file = "/".join(file)
            os.remove(file)
        metadata = metadata.reset_index()
        metadata.to_hdf(target_file, "meta", "r+")
    f.close()
    return None


def _add_metadata(db_path, key, metadata):
    metadata.to_hdf(db_path, key=key + "/metadata", mode="r+")
    return None


def db_factory(root_directory, target_file, mode, extract_func, n_cores=cpu_count(), segmented=True, **kwargs):
    logger.info("storing features in temp dbs")
    tmp_metadata = _make_temp_dbs(root_directory, extract_func, n_cores)

    segments_metadata = _collect_segments_metadata(tmp_metadata) if segmented else None

    logger.info("copying temp dbs to target file")
    _aggregate_from_metadata(target_file, tmp_metadata, mode, ["segments"] if segmented else None, **kwargs)

    if segments_metadata is not None:
        _add_metadata(target_file, "segments", segments_metadata)

    logger.info("done! following Groups and Datasets are now stored in '{}':".format(target_file))
    h5py.File(target_file, "r").visititems(lambda x, obj: logger.info("name={} ; object={}".format(x, str(obj))))
    return target_file


