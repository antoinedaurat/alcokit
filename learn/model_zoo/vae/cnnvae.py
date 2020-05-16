import torch
from torch import nn
from torch.nn import functional as F

from cafca.learn.modules import convpair
from cafca.learn.model_zoo.vae.fcvae import FcVAE
from cafca.learn.model_zoo.vae import VariationalAutoEncoder


def check_pair(down, up, input_):
    print("input", input_.shape)
    mid = down(input_)
    print("mid", mid.shape)
    final = up.forward(mid)
    print("final", final.shape)


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

        super(ConvVAE, self).__init__(encoder, decoder, reconstruction_loss=recon_loss)