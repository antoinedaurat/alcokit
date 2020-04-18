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
    df = pd.DataFrame.from_dict(dict(tmp_dbs), orient="index")
    df.index.name = "file"
    return df


def _collect_segments_metadata(metadata):
    logger.info("collecting segments' metadata")
    files = list(metadata.index)
    frames = []
    offset = 0
    for file in files:
        with h5py.File(file, "r") as f:
            segments = f["segments"][()]
            # last elements in segments should always be the length of the whole segmented data
            durations = np.diff(segments)
            index = pd.MultiIndex.from_product([[file.strip(".h5")], range(len(durations))], names=["file", "index"])
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
    features = [col for col in metadata.columns if col not in exclude]
    # for each feature, collect the dtype and the shapes of all the files
    metas = {feature: (list(set([meta[0] for meta in metadata[feature]]))[0],
                       np.array([meta[1:] for meta in metadata[feature]]))
             for feature in features}
    with h5py.File(target_file, mode) as f:
        for feature, (dtype, shapes) in metas.items():
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
                file = file.strip(".h5")
                meta[file] = dict(index=i, start=start, stop=stop, duration=shapes[i, 0])
            # add attrs to the DS
            for key, value in attrs:
                feature_ds.attrs[key] = value
            # this will be handy to figure out batch_size when iterating/querying etc.
            feature_ds.attrs["axis0_nbytes"] = feature_ds[0].nbytes
            f.flush()
            # store the metadata for the whole feature
            meta = pd.DataFrame.from_dict(meta, orient="index")
            meta.index.name = "file"
            meta.reset_index().to_hdf(target_file, feature+"/metadata", "r+")
        # clean up
        for file in metadata.index:
            os.remove(file)
        metadata.reset_index().to_hdf(target_file, "meta", "r+")
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


