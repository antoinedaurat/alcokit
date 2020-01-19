import os

import librosa
import mido
import numpy as np

from cafca.util import flat_dir


def _as_dataset(iterable, ndim=2, max_t=-1, t_axis=1):
    """
    helper functions to format iterables of [C x] N x ti arrays
    to a single [C x] T (:= sum_ti) x N array
    """
    if ndim == 2:  # T x F
        return np.hstack(tuple(iterable())).T

    elif ndim == 3:  # T x max_t x F
        if max_t <= 0:  # max_t := max(Ti)

            max_t = max([tm.shape[t_axis] for tm in iterable()])

            def pad_and_transpose(x):
                if len(x.shape) == 2:  # F x T
                    x = np.pad(x, ((0, 0), (0, max_t - x.shape[t_axis])))
                    return x.T
                else:  # F x T x (Real, Imag)
                    x = np.pad(x, ((0, 0), (0, max_t - x.shape[t_axis]), (0, 0)))
                    return np.transpose(x, (1, 0, 2))

            X = [pad_and_transpose(tm) for tm in iterable()]

            return np.stack(X)

        else:  # T := sum_i Ti % max_t

            def pad_and_transpose(x):
                if len(x.shape) == 2:  # F x T
                    x = np.pad(x, ((0, 0), (0, max_t - (x.shape[t_axis] % max_t))))
                    return x.T
                else:  # F x T x (Real, Imag)
                    x = np.pad(x, ((0, 0), (0, max_t - (x.shape[t_axis] % max_t)), (0, 0)))
                    return np.transpose(x, (1, 0, 2))

            X = [pad_and_transpose(tm) for tm in iterable()]

            # split each by max_t ( TODO : frame + hop instead of split )
            X = [np.split(tm,
                          np.arange(tm.shape[0] + tm.shape[0] % max_t + 1, step=max_t))
                 for tm in X]

            # flatten the splits
            X = [x for split in X for x in split if x.size != 0]

            return np.stack(X)


# **************************************#
# AUDIOSET
# **************************************#

def first_and_last_nonzero(y):
    n = y.size
    # first and last cut points
    a, b = 0, -1
    while y[a] == 0:
        a += 1
    while y[b] == 0:
        b -= 1
    return a, n + b + 1


class AudioSet(object):
    def __init__(self,
                 directory,
                 frepr=librosa.stft,
                 max_n_samples=-1,
                 channels=1,
                 sr=22050,
                 hop_length=1024,
                 F=2048,
                 recursive=False
                 ):
        """
        get names and wavs from a directory
        and compute their representations with frepr
        """
        self.directory = directory
        self.frepr = frepr
        self.files = {}
        self.wav = {}
        self.repr = {}
        self.channels = channels
        self.sr = sr
        self.hop_length = hop_length
        self.F = F
        self._cache_wav_and_repr(channels=channels,
                                 max_n_samples=max_n_samples,
                                 recursive=recursive)

    def _cache_wav_and_repr(self, channels=1, max_n_samples=-1, recursive=False):
        directory = self.directory
        # make a generator to loop through the files
        gen = iter([])
        if directory:
            if not recursive:
                gen = [os.path.join(directory, f)
                       for f in os.listdir(directory)
                       if f.split(".")[-1] in ("mp3", "wav")]
            else:
                gen = flat_dir(directory)
        # loop through the files and compute frepr
        i = 0
        for file in gen:
            if not file.split(".")[-1] in ("mp3", "wav", "aif", "aiff"):
                print("skipping ", file)
                continue
            if 0 < max_n_samples <= i:
                break
            print("loading audiofile :", file, "at index", i)
            self.files[i] = file
            self.wav[i], _ = librosa.load(file, sr=self.sr,
                                          mono=False if channels != 1 else True)
            # trim the wav file
            start, stop = first_and_last_nonzero(self.wav[i])
            self.wav[i] = self.wav[i][start:stop]
            # ## fix its length
            # n = self.wav[i].size
            # pad = n + self.F // (self.F // self.hop_length)
            # self.wav[i] = librosa.util.fix_length(self.wav[i], pad)
            self.repr[i] = self.frepr(self.wav[i])
            i += 1

    def as_dataset(self, ndim=2, max_t=-1, t_axis=1):
        return _as_dataset(self.repr.values, ndim=ndim, max_t=max_t, t_axis=t_axis)


# **************************************#
# MIDISET
# **************************************#


def midifile2tm(midifile,
                sr=22050, F=2048, hop_length=1024,
                quant_cut_off=.75,
                P=88,
                channels="merge"
                ):
    """
    transform a midifile to a [C channels x] P pitches x T timesteps array
    if the channels are not merged, they are returned flattened
    """
    # DURATIONS
    glob_pos = np.cumsum(np.array([msg.time for msg in midifile]))
    frames_per_sec = 1. * (sr / hop_length)
    # compute quantized global positions 
    glob_pos = (glob_pos * frames_per_sec + (1 - quant_cut_off))
    glob_pos = np.rint(glob_pos).astype(np.int32)

    # glob_pos = ( glob_pos * frames_per_sec ).astype(np.int32)

    # TYPES
    def encode_msg_type(msg):
        if msg.type == "note_off":
            return -1
        elif msg.type == "note_on":
            return 1
        else:
            return 0

    types = np.array([encode_msg_type(m) for m in midifile])
    onsets = (types == 1).nonzero()[0]
    offsets = (types == -1).nonzero()[0]

    # # remove leading silence :
    if glob_pos[onsets[0]] != 0:
        glob_pos[onsets[0]:] -= glob_pos[onsets[0]]

    # PITCHES
    pitches = np.array([msg.note
                        if msg.type in ("note_on", "note_off")
                        else -1
                        for msg in midifile])
    # CHANNELS
    mchannels = np.array([msg.channel
                          if msg.type in ("note_on", "note_off")
                          else -1
                          for msg in midifile])

    C = len(set([m.channel for m in midifile if m.type in ("note_on", "note_off")]))
    T = glob_pos.max() + 1
    encoded = np.zeros((C, P, T), dtype=np.int32)

    # setting encoded in the order 1/ offs 2/ ons ensure, that we override note offs connecting repeating pitches
    encoded[mchannels[offsets], pitches[offsets], glob_pos[offsets]] = -1
    # ensure we don't get zeros summing on & off in separate channels
    encoded[mchannels[onsets], pitches[onsets], glob_pos[onsets]] = C

    def write_durations(p_row):
        """
        set the row to 1 for the duration of the onsets (index of offsets are excluded)
        """
        idx = p_row.nonzero()[0]
        slices = [slice(i, j) for i, j in zip(idx[:-1], idx[1:])]
        vals = p_row[idx]
        for t, v in enumerate(vals):
            if v >= 1:
                p_row[slices[t]] = 2
            #             p_row[slices[t]] = np.linspace(1., .2, slices[t].stop - slices[t].start)
            else:
                # reset not_offs to 0 (would cause exception if last note is a note_on...)
                p_row[idx[t]] = 0
        return p_row

    # get the unique pairs (channel, pitch) where there is nonzeros
    active_cr = np.unique(np.asarray(encoded.nonzero()[0:2]).T, axis=0)
    for r in active_cr:
        encoded[r[0], r[1]] = write_durations(encoded[r[0], r[1]])

    if channels == "merge":
        encoded = encoded.sum(axis=0)
        return np.clip(encoded, 0, 1)
    elif channels == "flatten":
        encoded = encoded.reshape(P, -1)
        return np.clip(encoded, 0, 1)
    return np.clip(encoded, 0, 1)


class MidiSet(object):
    def __init__(self,
                 directory=".",
                 max_n_samples=-1,
                 num_pitches=88,
                 channels="merge",
                 sr=22500,
                 F=2048,
                 hop_length=1024,
                 quant_cut_off=.75
                 ):
        self.directory = directory
        self.num_pitches = num_pitches
        self.sr = sr
        self.F = F
        self.hop_length = hop_length
        self.quant_cut_off = quant_cut_off
        self.channels = channels
        self.files = {}
        self.tm = {}
        self.midi = {}
        self._cache_midi_and_tm(max_n_samples=max_n_samples,
                                channels=channels)

    def _cache_midi_and_tm(self, max_n_samples=-1, channels="merge"):
        i = 0
        for file in os.listdir(self.directory):
            if ".mid" not in file:
                print("skipping file", file)
                continue
            if 0 < max_n_samples <= i:
                break
            print("loading midifile :", file, "at index", i)
            midifile = mido.MidiFile(os.path.join(self.directory, file))
            tm = midifile2tm(midifile, sr=self.sr,
                             F=self.F, hop_length=self.hop_length,
                             quant_cut_off=self.quant_cut_off,
                             P=self.num_pitches, channels=channels)
            self.files[i] = file
            self.midi[i] = midifile
            self.tm[i] = tm
            i += 1

    def as_dataset(self, ndim=2, max_t=-1, t_axis=1):
        return _as_dataset(self.tm.values, ndim=ndim, max_t=max_t, t_axis=t_axis)

    def sync_with_audioset(self, audioset, t_axis=1):
        """
        equalize the lengths of the tm to those of the repr of an AudioSet.
        longer tm get trimmed, shorter tm get padded with zeros at the end.
        """
        for (i, rep), (j, tm) in zip(audioset.repr.items(), self.tm.items()):
            if rep.shape[t_axis] < tm.shape[t_axis]:
                new_shape = tuple(slice(x if i == t_axis else tm.shape[i])
                                  for i, x in enumerate(rep.shape)
                                  if i < len(tm.shape))
                self.tm[j] = tm[new_shape]
            elif rep.shape[t_axis] > tm.shape[t_axis]:
                pad_shape = tuple((0, rep.shape[i] - tm.shape[i]) if i == t_axis
                                  else (0, 0)
                                  for i, x in enumerate(rep.shape)
                                  if i < len(tm.shape))
                self.tm[j] = np.pad(tm, pad_shape)
        return None


# Experimental ( to be integrated in midifile2tm ? )

def encode_durations(tm, glob_pos, types, pitches, channels):
    P = 88
    C = tm.shape[0]
    for c in range(C):
        for p in range(P):
            subset = (pitches == p) & (channels == c)
            pos = glob_pos[subset]
            typ = types[subset]
            ons = pos[typ == 1]
            offs = pos[typ == -1]
            if ons.size > 0:
                for n, f in zip(ons, offs):
                    tm[c, p, n:f] = np.linspace(1., 0., f - n + 1)[:-1]
    return tm
