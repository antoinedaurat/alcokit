import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cafca.models.base import ModelBase


class AutoEncoder(ModelBase):
    def __init__(self, module, reconstruction_loss, input_shape=None):
        super(AutoEncoder, self).__init__(module, input_shape)
        self.reconstruction_loss = reconstruction_loss

    def get_loss(self, x):
        x_pred = self(x)
        return self.reconstruction_loss(x_pred, x)


class VariationalAutoEncoder(AutoEncoder):

    def __init__(self,
                 input_shape,
                 encoder,
                 decoder,
                 reconstruction_loss=F.smooth_l1_loss):
        super(VariationalAutoEncoder, self).__init__(None, reconstruction_loss, input_shape)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = None

    def forward(self, input: torch.Tensor):
        z, mu, logvar = self.encoder(input)
        recon = self.decoder(z)
        return recon, mu, logvar

    def get_loss(self, x_true):
        x_pred, mu, logvar = self.forward(x_true)
        RE = self.reconstruction_loss(x_pred, x_true)
        # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return RE

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(self.as_input(x))[0]

    def decode(self, x):
        with torch.no_grad():
            return self.decoder(self.as_input(x))

    def visualize(self, x, tags=None):
        x = self.as_input(x)
        if x.size(-1) != 2:
            print("reducing encoding with PCA...")
            encoded = self.encode(x)
            encoded = PCA(n_components=2).fit_transform(self.as_output(encoded))
        else:
            encoded = self.as_output(self.encode(x))
        if tags is None:
            tags = [str(i) for i in range(len(x))]
        plt.figure(figsize=(12, 12))
        plt.scatter(encoded[:, 0], encoded[:, 1])
        for n, txt in enumerate(tags):
            plt.text(encoded[n, 0], encoded[n, 1], txt)
        plt.show()
