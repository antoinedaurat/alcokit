import torch.nn.functional as F
import numpy as np

from cafca.learn.modules import *
from cafca.learn.model_zoo.vae.vae import VariationalAutoEncoder


class FcVAE(VariationalAutoEncoder):
    def __init__(self, input_shape, sizes, batch_norm, activation, dropout,
                 reconstruction_loss=F.smooth_l1_loss,
                 pre_sampler_act=nn.Tanh):
        input_dim = input_shape[-1] if len(input_shape) == 2 else np.prod(input_shape[1:])
        z_dim = sizes[-1]
        sizes = [input_dim, *sizes[:-1]]

        encoder = nn.Sequential(
            # Flatten(),
            FcStack(sizes, batch_norm, activation, dropout),
            ParamedSampler(sizes[-1], z_dim, pre_sampler_act)
        )
        decoder = nn.Sequential(
            FcStack([z_dim, *sizes[::-1]], batch_norm, activation, dropout),
            # UnFlatten(input_shape[1:])
        )
        super(FcVAE, self).__init__(encoder, decoder, reconstruction_loss)