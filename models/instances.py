import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cafca.models.parts import *
from cafca.models.vae import AutoEncoder, VariationalAutoEncoder


class FcVAE(VariationalAutoEncoder):
    def __init__(self, input_shape, sizes, batch_norm, activation, dropout,
                 reconstruction_loss=F.smooth_l1_loss,
                 pre_sampler_act=nn.Tanh):
        input_dim = input_shape[-1] if len(input_shape) == 2 else np.prod(input_shape[1:])
        z_dim = sizes[-1]
        sizes = [input_dim, *sizes[:-1]]

        encoder = nn.Sequential(
            Flatten(),
            FcStack(sizes, batch_norm, activation, dropout),
            ParamedSampler(sizes[-1], z_dim, pre_sampler_act)
        )
        decoder = nn.Sequential(
            FcStack([z_dim, *sizes[::-1]], batch_norm, activation, dropout),
            UnFlatten(input_shape[1:])
        )
        super(FcVAE, self).__init__((1, input_dim), encoder, decoder, reconstruction_loss)


def check_pair(down, up, input_):
    print("input", input_.shape)
    mid = down(input_)
    print("mid", mid.shape)
    final = up.forward(mid)
    print("final", final.shape)


class ConvAutoEncoder(AutoEncoder):
    def __init__(self,
                 input_shape,
                 filters,
                 kernels,
                 strides,
                 pads,
                 dilations,
                 poolargs,
                 batch_norm,
                 activation,
                 dropout,
                 check_shapes=False):
        convs_down, convs_up = zip(*[convpair(f, k, s, p, d, batch_norm, activation, pool, dropout)
                                     for f, k, s, p, d, pool in
                                     zip(filters, kernels, strides, pads, dilations, poolargs)])
        if check_shapes:
            input_ = torch.randn(input_shape)
            for down, up in zip(convs_down, convs_up):
                check_pair(down, up, input_)
                print()
                input_ = down(input_)

        module = nn.Sequential(
            *convs_down,
            *convs_up[::-1]
        )
        super(ConvAutoEncoder, self).__init__(module, F.smooth_l1_loss, input_shape)


class ConvVAE(VariationalAutoEncoder):
    def __init__(self,
                 input_shape,
                 filters,
                 kernels,
                 strides,
                 pads,
                 dilations,
                 poolargs,
                 batch_norm,
                 activation,
                 dropout,
                 fc_sizes,
                 fc_bn,
                 fc_act,
                 fc_dropout,
                 pre_sampler_act,
                 recon_loss=F.smooth_l1_loss,
                 check_shapes=False
                 ):

        convs_down, convs_up = zip(*[convpair(f, k, s, p, d, batch_norm, activation, pool, dropout)
                                     for f, k, s, p, d, pool in
                                     zip(filters, kernels, strides, pads, dilations, poolargs)])
        if check_shapes:
            input_ = torch.randn(input_shape)
            for down, up in zip(convs_down, convs_up):
                check_pair(down, up, input_)
                print()
                input_ = down(input_)

        input_ = torch.randn(input_shape)
        for down in convs_down:
            input_ = down(input_)

        # fcvae will take care of flattening and reshaping input and output
        fcvae = FcVAE(tuple(input_.size()), fc_sizes, fc_bn, fc_act, fc_dropout,
                      pre_sampler_act=pre_sampler_act)
        fc_enc, fc_dec = fcvae.encoder, fcvae.decoder

        encoder = nn.Sequential(*convs_down, fc_enc)
        decoder = nn.Sequential(fc_dec, *convs_up[::-1])

        super(ConvVAE, self).__init__(input_shape, encoder, decoder, reconstruction_loss=recon_loss)
