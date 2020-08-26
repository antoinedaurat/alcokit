import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", message="Did not find hyperparameters at model hparams. Saving checkpoint without hyperparameters.")
warnings.filterwarnings("ignore", message="The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` in the `DataLoader` init to improve performance.")

# Constants
# those are shared and imported everywhere. Having them here makes it easier
# to set them once globally and forget about them, e.g.
# import alcokit
#
# alcokit.HOP_LENGTH = 1234
#
# from alcokit import algo -> every stft, audio display etc. will have a default hop_length of 1234

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050

