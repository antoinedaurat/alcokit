import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
import torch

# Constants
# those are shared and imported everywhere. Having them here makes it easier
# to set them once globally and forget about them, e.g.
# import cafca
#
# cafca.HOP_LENGTH = 1234
#
# from cafca import algo -> every stft, audio display etc. will have a default hop_length of 1234

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cafca initialized with device:", DEVICE)