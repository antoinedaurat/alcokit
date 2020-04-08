import librosa
from multiprocessing import Pool
import numpy as np
from cafca.segment import SegmentMap, expected_len
from cafca.util import is_audio_file
import os
import h5py
import warnings
warnings.filterwarnings("ignore")


def segment_file(file_path,
                 n_fft=2048, hop_length=512,  # ffts params
                 L=6, k=200, sym=True, bandwidth=5, thresh=.2, min_dur=5):  # segmentation params
    fft = librosa.stft(librosa.load(file_path)[0], n_fft=n_fft, hop_length=hop_length)
    return SegmentMap(fft, L=L, k=k, sym=sym, bandwidth=bandwidth, thresh=thresh, min_dur=min_dur, plot=False)


def segment_directory(files,
                      n_cores=4,
                      n_fft=2048, hop_length=512,  # ffts params
                      L=6, k=200, sym=True, bandwidth=5, thresh=.2, min_dur=5  # segmentation params
                      ):
    params = n_fft, hop_length, L, k, sym, bandwidth, thresh, min_dur
    with Pool(n_cores) as p:
        maps = p.starmap(segment_file, [(f, *params) for f in files])
    return maps


def standardize_directory(maps, n_cores=4, seg_len="all", mode="modulo"):
    if seg_len == "all":
        all_lens = [n for m in maps for n in m.o_lens_list]
        N = int(np.ceil(expected_len(all_lens)))
        args = [(m, N) for m in maps]
    else:
        args = [(m, m.expected_length()) for m in maps]
    func = SegmentMap.mod_standardize if mode == "modulo" else SegmentMap.standardize
    with Pool(n_cores) as p:
        maps = p.starmap(func, args)
    return maps, [arg[1] for arg in args]


def to_hdf5(directory, target,
            n_cores=4, seg_len="all", mode="modulo",
            n_fft=2048, hop_length=512,  # ffts params
            L=6, k=200, sym=True, bandwidth=5, thresh=.2, min_dur=5  # segmentation params
            ):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if is_audio_file(f)]
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(d)]
    maps = segment_directory(files, n_cores, n_fft, hop_length, L, k, sym, bandwidth, thresh, min_dur)
    maps, lens = standardize_directory(maps, n_cores, seg_len, mode)
    with h5py.File(target, "w") as f:
        grp = f.create_group(directory)
        for name, m, n in zip(files, maps, lens):
            ds_name = os.path.split(name)[-1]
            ds = m.as_dataset(n)
            grp.create_dataset(ds_name, ds.shape, dtype="f", data=ds, compression="gzip", compression_opts=4)


