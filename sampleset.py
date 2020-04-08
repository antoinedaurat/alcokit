import os
from itertools import accumulate as accum

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa.core.spectrum import griffinlim
from librosa.display import specshow, waveplot
from sklearn.preprocessing import MinMaxScaler
from cafca.util import flat_dir, is_audio_file


def scale(X: list):
    indices = list(accum([x.shape[0] for x in X]))[:-1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(np.vstack(X))
    return np.split(X, indices), scaler


class SampleSet(object):
    """
     CONSTRUCTORS
    """

    def from_ffts(self, ffts, n_fft=1024, hop_length=512, sample_rate=22050, wavs=[], names=[]):
        """ """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.wav = {}
        self.ffts = {}
        self.names = {}
        self.indices = {}
        self.N = len(ffts)
        for i, fft in enumerate(ffts):
            if len(fft.shape) == 3:  #
                if fft.shape[-1] == 1:
                    fft = np.squeeze(fft)
                else:
                    fft = np.squeeze(fft.view(dtype=np.complex64))
            elif len(fft.shape) == 2 and fft.shape[-1] == 2:  # flat samples with imag
                fft = fft.view(dtype=np.complex64).reshape(-1, n_fft, 2)
            elif len(fft.shape) == 1:
                fft = fft.reshape(-1, n_fft, 2)
            self.ffts[i] = fft
            if not wavs:
                if fft.shape[-1] == 2:
                    self.wav[i] = librosa.istft(self.ffts[i].T)
                else:
                    self.wav[i] = griffinlim(self.ffts[i].T)
            else:
                self.wav[i] = wavs[i]
            if not names:
                self.names[i] = "sample_" + str(i) if not names else names[i]
            else:
                self.names[i] = names[i]
            self.indices[self.names[i]] = i
        return self

    def from_directory(self, directory, max_n_samples=-1, recursive=False, with_imag=True):
        """get names and wavs from files and compute the ffts"""
        gen = iter([])
        if directory:
            if not recursive:
                gen = enumerate(
                    [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if is_audio_file(f)])
            else:
                gen = enumerate(flat_dir(directory))

        self.N = 0
        for _, file in gen:
            if not is_audio_file(file):
                print("skipping unsupported file", file)
                continue
            i = self.N
            if 0 < max_n_samples <= i:
                break
            print("loading :", file, "at index", i)
            self.names[i] = file if file not in self.names.keys() else "_" + file
            self.indices[file] = i
            self.wav[i], _ = librosa.load(file)
            self.ffts[i] = librosa.stft(self.wav[i], **self.fft_params)
            self.N += 1
            if not with_imag:
                self.ffts[i] = abs(self.ffts[i])
            n_frames = self.ffts[i].shape[1]
            if n_frames > self.max_n_frames:
                self.max_n_frames = n_frames
        if self.N == 0:
            print("WARNING : no files were found in directory...")
        return self

    def __init__(self,
                 directory="",
                 wavs=[],
                 ffts=[],
                 names=[],
                 n_fft=1024,
                 hop_length=512,
                 sample_rate=22050,
                 with_imag=True,
                 max_n_samples=-1,
                 recursive=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fft_params = dict(n_fft=n_fft, hop_length=hop_length)
        self.sample_rate = sample_rate
        self.wav = {}
        self.ffts = {}
        self.names = {}
        self.indices = {}
        self.N = 0
        self.max_n_frames = 0

        if directory:
            self.from_directory(directory, max_n_samples=max_n_samples, recursive=recursive, with_imag=with_imag)
        elif ffts:
            self.from_ffts(ffts, **self.fft_params, sample_rate=self.sample_rate, wavs=wavs, names=names)

    '''
     UTILS
    '''

    @property
    def all_names(self):
        return list(self.names.items())

    def from_name(self, name):
        i = self.indices[name]
        return self.wav[i], self.ffts[i]

    def as_dataset(self, with_imag=False, scaled_r=False, scaled_i=False, padding=True, ndim=3):
        """ returns an array of shape (n_samples, k_frames, n_fft // 2 + 1, [1 if whit_channel])"""
        X = [x.T for x in self.ffts.values()]  # each sample has shape T x F
        if not with_imag:
            X = [abs(x) for x in X]
            if scaled_r:
                X, scaler_r = scale(X)
        else:
            X_real = [abs(x) for x in X]
            X_imag = [np.imag(x) for x in X]
            if scaled_r:
                X_real, scaler_r = scale(X_real)
            if scaled_i:
                X_imag, scaler_i = scale(X_imag)
            X = [np.dstack((xr, xi)) for xr, xi in zip(X_real, X_imag)]
        if padding:
            if with_imag:
                for i, fft in enumerate(X):
                    X[i] = np.pad(fft, ((0, self.max_n_frames - fft.shape[0]), (0, 0), (0, 0)), constant_values=0)
            else:
                for i, fft in enumerate(X):
                    X[i] = np.pad(fft, ((0, self.max_n_frames - fft.shape[0]), (0, 0)), constant_values=0)
        if ndim == 2:  # example x (flatten T x F) [x 2 if with_imag]
            if with_imag:
                # split real and imag parts
                return np.vstack([x[:, :, 0].reshape(-1) for x in X]), \
                       np.vstack([x[:, :, 1].reshape(-1) for x in X])
            else:
                return np.vstack([x.reshape(-1) for x in X])
        elif ndim == 3:  # example x T x F
            return np.stack([x for x in X])
        elif ndim == 4:  # example x T x F x channel
            if with_imag:
                return np.stack([x for x in X])
            else:
                return np.stack([x.reshape(*x.shape, 1) for x in X])

    def as_batches_generator(self, b_size=10, with_imag=False, scaled_r=False, scaled_i=False, padding=True, ndim=3):
        pass
        # this should allow to feed samples of different sizes to a recurrent model...

    def play(self, i):
        return ipd.Audio(self.wav[i], rate=self.sample_rate)

    def plot_spectrum(self, i, min_t=0, min_f=0, max_t=-1, max_f=-1, figsize=(15, 5), segments=None):
        # TODO : accept ranges as argument
        spectrum = self.ffts[i].T
        if max_t < 0 and max_f < 0:
            spectrum = spectrum[min_f:, min_t:]
        elif max_t < 0 and max_f > 0:
            spectrum = spectrum[min_f:max_f, min_t:]
        elif max_t > 0 and max_f < 0:
            spectrum = spectrum[min_f:, min_t:max_t]
        else:
            spectrum = spectrum[min_f:max_f, min_t:max_t]
        plt.figure(figsize=figsize)
        specshow(librosa.amplitude_to_db(spectrum), sr=self.sample_rate,
                 hop_length=self.hop_length, x_axis='time', y_axis='linear')
        if np.all(segments != None):
            plt.vlines(librosa.frames_to_time(segments, sr=self.sample_rate, hop_length=self.hop_length),
                       0, 4000, color="w")
        plt.title(self.names[i])
        return None

    def plot_wave(self, i, min_t=0, max_t=-1):
        wav = self.wav[i]
        if max_t < 0:
            wav = wav[min_t:max_t]
        else:
            wav = wav[min_t:]
        return waveplot(wav)

    def export_audio(self, directory):
        directory = directory if directory[-1] == '/' else directory + '/'
        for name, i in self.names.items():
            scipy.io.wavfile.write(directory + name + '.wav', rate=self.sample_rate, data=self.wav[i])

    def plot_all(self, min_t=0, min_f=0, max_t=-1, max_f=-1, figsize=(10, 5), segments=[]):
        for i in range(len(self.ffts)):
            self.plot_spectrum(i, min_t=min_t, min_f=min_f, max_t=max_t, max_f=max_f,
                               figsize=figsize, segments=segments[i] if segments else [])

    def play_all(self):
        for i in range(len(self.wav)):
            print(self.names[i])
            ipd.display(self.play(i))

    # SPECTRALS
    def hpss(self):
        H, P = zip(*map(librosa.decompose.hpss, self.ffts.values()))
        return SampleSet(ffts=H, names=list(self.names.keys()), **self.fft_params), \
               SampleSet(ffts=P, names=list(self.names.keys()), **self.fft_params)

    def chroma_cens(self, n_chroma=12, n_octaves=7):
        return [librosa.feature.chroma_cens(x, n_chroma=n_chroma, n_octaves=n_octaves) for x in self.wav.values()]

    def pitch_track(self):
        return [librosa.piptrack(S=fft.T) for fft in self.ffts.values()]

    def melspectrogram(self, n_mels=128, fmin=32, fmax=8000):
        return [librosa.feature.melspectrogram(S=abs(fft.T), n_mels=n_mels, fmin=fmin, fmax=fmax) \
                for fft in self.ffts.values()]

    def mfcc(self):
        pass

    # DECOMPOSITIONS
    def segment_by_plp(self):
        pulse = map(librosa.beat.plp, self.wav.values())
        beats_plp = map(lambda p: np.flatnonzero(librosa.util.localmax(p)), pulse)
        segments = [np.split(wav, seg) for wav, seg in zip(self.wav.values(), beats_plp)]

        # wavs = [w for wavs in segments for w in wavs]
        # ffts = [librosa.stft(w, **self.fft_params).T for w in wavs]
        # names = [self.names[i] + "_" + str(j) for i, seg in enumerate(segments) for j, _ in enumerate(seg)]
        # new = SampleSet(ffts=ffts, wavs=wavs, names=names, **self.fft_params, sample_rate=self.sample_rate)
        return beats_plp

    def segment_by_beat_track(self):
        pass

    def segment_by_onset(self):
        pass
        oenv = map(librosa.onset.onset_strength, self.wav.values())
        onset_raw = map(librosa.onset.onset_detect, self.wav.values())
        onset_bt = [librosa.onset.onset_backtrack(r, e) for r, e in zip(onset_raw, oenv)]
        # onset_wav = [librosa.frames_to_samples(o) for o in onset_bt]
        # ffts = [np.split(x, o) for x, o in zip(self.ffts.values(), onset_bt)]
        # wavs = [np.split(x, o) for x, o in zip(self.wav.values(), onset_wav)]
        # names = [self.names[i] + "_" + str(j) for i, seg in enumerate(ffts) for j, _ in enumerate(seg)]
        # ffts = [s for seg in ffts for s in seg]
        # wavs = [s for seg in wavs for s in seg]
        # new = SampleSet(ffts=ffts, wavs=wavs, names=names, **self.fft_params, sample_rate=self.sample_rate)
        return onset_bt

    def segment_by_agglo(self, k=2, clusterer=None):
        segments = [librosa.segment.agglomerative(x, k=k, axis=0, clusterer=clusterer) for x in self.ffts.values()]
        return segments

    def make_subsegment(self, method="beat_track", n_segments=2):
        pass


def segment_by_agglo(X: list, k=2, clusterer=None):
    return [librosa.segment.agglomerative(x, k=k, axis=0, clusterer=clusterer) for x in X]
