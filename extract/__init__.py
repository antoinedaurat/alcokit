from alcokit.fft import FFT
from alcokit.score import Score


def default_extract_func(abs_path):
    fft = abs(FFT.stft(abs_path))
    score = Score.from_recurrence_matrix(fft)
    return dict(fft=({}, fft.T), score=({}, score))