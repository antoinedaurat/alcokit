import os
from cafca import HOP_LENGTH, N_FFT, SR
from cafca.util import flat_dir, is_audio_file
from cafca.fft import FFT


class FFTSet(dict):

    def from_directory(self, directory, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       sr=SR, max_n_samples=-1, recursive=False):
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
            self[i] = FFT.stft(file, n_fft, hop_length, sr)
            self.N += 1
        if self.N == 0:
            print("WARNING : no files were found in directory...")
        return self

    def __init__(self,
                 directory="",
                 n_fft=N_FFT,
                 hop_length=HOP_LENGTH,
                 sr=SR,
                 max_n_samples=-1,
                 recursive=True):
        super(FFTSet, self).__init__()
        self.N = 0
        self.from_directory(directory, n_fft=n_fft, hop_length=hop_length, sr=sr,
                            max_n_samples=max_n_samples, recursive=recursive)

    @property
    def names(self):
        return {i: fft.file for i, fft in self.items()}

