import torch
from torch.nn import functional as F

from cafca.learn.model_base import ModelBase


class VariationalAutoEncoder(ModelBase):

    def __init__(self,
                 encoder,
                 decoder,
                 reconstruction_loss=F.smooth_l1_loss):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss

    def forward(self, input: torch.Tensor):
        z, mu, logvar = self.encoder(input)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        return self.encoder(self.preprocess(x))

    def decode(self, x):
        return self.decoder(self.preprocess(x))
