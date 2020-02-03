import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import time


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, output_shape: tuple):
        super(UnFlatten, self).__init__()
        self.output_shape = output_shape

    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), *self.output_shape)


class CheckedSequential(nn.Sequential):
    """
    wrapper class that makes a forward pass through a list of Modules
    to check the compatibility of their input/output shapes and store those shapes in a dict
    before passing the modules to nn.Sequential
    """

    @staticmethod
    def _check(a, input_, i):
        try:
            input_ = a(input_)
        except RuntimeError as e:
            msg = "previous shape %s could not go through module %s at index %i. " % \
                  (str(tuple(input_.size())), str(a), i) + \
                  "Exception was : 'RuntimeError: %s'" % str(e)
            raise RuntimeError(msg)

        if isinstance(input_, tuple):
            input_ = input_[0]
        return input_

    def __init__(self, input_shape, *args):
        input_ = torch.randn(*input_shape)
        shapes = {-1: tuple(input_.size())}
        for i, a in enumerate(args):
            input_ = self._check(a, input_, i)
            shapes[i] = tuple(input_.size())

        self.shapes = shapes
        self.output_shape = shapes[i]
        super(CheckedSequential, self).__init__(*args)

    def extend_(self, x):
        out = self._check(x, torch.randn(*self.output_shape), len(self))
        self.shapes[max(self.shapes.keys()) + 1] = tuple(out.size())
        self.output_shape = tuple(out.size())
        self._modules[str(len(self))] = x


class BottleNeck(nn.Module):
    def __init__(self, input_dim: int, z_dim: int):
        super(BottleNeck, self).__init__()
        self.fc1 = nn.Linear(input_dim, z_dim)
        self.fc2 = nn.Linear(input_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size(), device=self.get_device())
        z = mu + std * eps
        return z, mu, logvar

    def get_device(self):
        return next(self.parameters()).device.type


class FCStack(CheckedSequential):
    @staticmethod
    def fc_unit(d_in, d_out,
                dropout=0.,
                activation=None,
                batch_norm=False):
        """
        build a Sequential from a nn.Linear layer and the specified options
        """
        unit = [nn.Linear(d_in, d_out)]
        if batch_norm:
            unit += [nn.BatchNorm1d(d_out)]
        if dropout > 0.:
            unit += [nn.Dropout(p=dropout)]
        if activation is not None:
            unit += [activation()]
        return nn.Sequential(*unit)

    def __init__(self, input_dim, sizes=(), activation=None, dropout=0., batch_norm=False, tail=[]):
        last_out = input_dim
        options = dict(activation=activation, dropout=dropout)
        units = []
        N = len(sizes)
        for n, out in enumerate(sizes):
            # apply batchnorm only at the last layer and always omit the last activation
            if n == N - 1:
                options["activation"] = None
                units += [self.fc_unit(last_out, out, batch_norm=batch_norm, **options)]
            else:
                units += [self.fc_unit(last_out, out, **options)]
            last_out = out

        super(FCStack, self).__init__((1, input_dim), *(units + tail))


class CNNUnit(nn.Module):
    def __init__(self, f, k, s, d, p,
                 n_conv, transposed=False,
                 batch_norm=False, activation=nn.ReLU, pooling=None, dropout=0.):
        super(CNNUnit, self).__init__()
        base = nn.Conv2d if not transposed else nn.ConvTranspose2d
        pool = nn.MaxPool2d if not transposed else nn.MaxUnpool2d
        self.conv = nn.Sequential(*[base(f[0], f[1], k, s, d, p) for _ in range(n_conv)])
        self.post = nn.Sequential(*[
            nn.BatchNorm2d(f[1]) if batch_norm else None,
            activation() if activation is not None else None,
        ])


class CNNStack(CheckedSequential):
    @staticmethod
    def cnn_unit(base, f, k, s, d, p,
                 dropout=0.,
                 activation=nn.ReLU,
                 pooling=None,
                 batch_norm=False):
        """
        build a Sequential from a Conv2d layer and the specified options
        """
        params = dict(kernel_size=k, stride=s, dilation=d, padding=p)
        unit = [base(f[0], f[1], **params).float()]
        if batch_norm:
            unit += [nn.BatchNorm2d(f[1])]
        if dropout > 0.:
            unit += [nn.Dropout(p=dropout)]
        if activation is not None:
            unit += [activation()]
        if pooling is not None:
            unit += [pooling]
        return nn.Sequential(*unit)

    def __init__(self,
                 base=None,
                 input_shape=None,
                 dropout=0.,
                 activation=None,
                 pooling=None,
                 batch_norm=False,
                 filters=(),
                 kernels=(),
                 strides=(),
                 dilations=(),
                 paddings=(),
                 tail=[]):
        N = len(filters)
        params = zip(range(N), filters, kernels, strides, dilations, paddings)
        options = dict(dropout=dropout, activation=activation, pooling=pooling)
        layers = []
        for n, f, k, s, d, p in params:
            # apply batchnorm only at the last layer and always omit the last activation
            if n == N - 1:
                options["activation"] = None
                layers += [self.cnn_unit(base, f, k, s, d, p, batch_norm=batch_norm, **options)]
            else:
                layers += [self.cnn_unit(base, f, k, s, d, p, **options)]

        super(CNNStack, self).__init__(input_shape, *(layers + tail))


class Encoder(CheckedSequential):
    def __init__(self, stack: CheckedSequential, z_dim: int):
        bottleneck = BottleNeck(stack.output_shape[-1], z_dim)
        super(Encoder, self).__init__(stack.shapes[-1], stack, bottleneck)
        self.z_dim = z_dim


class VariationalAutoEncoder(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: CheckedSequential,
                 reconstruction_loss=F.smooth_l1_loss,
                 non_linear_tail=False):
        assert encoder.shapes[-1] == decoder.output_shape
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = self.output_shape = encoder.shapes[-1]
        self.z_dim = self.encoder.z_dim
        self.reconstruction_loss = reconstruction_loss
        self.non_linear_tail = non_linear_tail
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = None

    def forward(self, input: torch.Tensor):
        z, mu, logvar = self.encoder(input)
        raw = self.decoder(z)
        if self.non_linear_tail:
            recon = torch.stack((nn.ReLU()(raw[:, 0]), nn.Tanh()(raw[:, 1])), dim=1)
            return recon, mu, logvar
        else:
            return raw, mu, logvar

    def loss(self, x_true, x_pred, mu, logvar):
        BCE = self.reconstruction_loss(x_true, x_pred, reduction="sum")
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def one_epoch(self, train_iterator, test_iterator):
        """
        combine the train step and the evaluation step for one epoch and return both losses.
        """
        self.train()
        train_loss = []
        start = time.time()
        for i, x in enumerate(train_iterator):
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self(x)
            loss = self.loss(x_recon, x, mu, logvar)
            loss.backward()
            train_loss += [loss.item()]
            self.optimizer.step()
        if test_iterator is None:
            duration = time.time() - start
            return train_loss, [0.], (duration, duration / (i+1))
        self.eval()
        test_loss = []
        with torch.no_grad():
            for i, x in enumerate(test_iterator):
                x_recon, z_mu, z_var = self(x)
                loss = self.loss_fn(x_recon, x, z_mu, z_var)
                test_loss += [loss.item()]
        duration = time.time() - start
        return train_loss, test_loss, (duration, duration / (i+1))

    def as_input(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, torch.Tensor):
            if x.device == self.device:
                return x.float()
            else:
                return x.float().to(self.device)

    @staticmethod
    def as_output(y):
        return y.detach().cpu().numpy()

    def fit(self, X, X_cv, lr, batch_size, n_epochs):
        # move the input to the right device and type and make iterators
        train_iterator = DataLoader(self.as_input(X), batch_size=batch_size, shuffle=True)
        if X_cv is not None:
            test_iterator = DataLoader(self.as_input(X_cv), batch_size=batch_size)
        else:
            test_iterator = None
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loss, test_loss = [], []
        print("Starting training")
        for e in range(n_epochs):
            tr_loss, ts_loss, (dur, avg_dur) = self.one_epoch(train_iterator, test_iterator)
            train_loss += tr_loss
            test_loss += ts_loss
            tr_loss = sum(tr_loss) / len(tr_loss)
            ts_loss = sum(ts_loss) / len(ts_loss)
            print(f'Epoch {e+1} took {dur:.3f}sec [{1000*(avg_dur/batch_size):.3f}ms/sample]. Train Loss: {tr_loss:.3f}, Test Loss: {ts_loss:.3f}')
        return train_loss, test_loss

    def encode(self, x, numpy=False):
        with torch.no_grad():
            return self.encoder(self.as_input(x))[0]

    def decode(self, x, numpy=False):
        with torch.no_grad():
            return self.decoder(self.as_input(x))

    def predict(self, x, numpy=False):
        with torch.no_grad():
            return self.forward(self.as_input(x))[0]

    def summary(self):
        with torch.no_grad():
            return summary(self, input_size=self.input_shape[1:], device=self.device.type)

    def visualize(self, x, visu_dim=2, tags=None):
        x = self.as_input(x)
        if visu_dim != self.z_dim:
            print("reducing encoding with PCA...")
            encoded = self.encode(x)
            encoded = PCA(n_components=visu_dim).fit_transform(self.as_output(encoded))
        else:
            encoded = self.as_output(self.encode(x))
        if tags is None:
            tags = [str(i) for i in range(len(x))]
        plt.figure(figsize=(8, 8))
        plt.scatter(encoded[:, 0], encoded[:, 1])
        for n, txt in enumerate(tags):
            plt.text(encoded[n, 0], encoded[n, 1], txt)
        plt.show()


def cnn_encdec_pair(input_shape=None,
                    z_dim=2,
                    fc_sizes=(),
                    dropout=0.,
                    activation=None,
                    pooling=None,
                    batch_norm=False,
                    filters=(),
                    kernels=(),
                    strides=(),
                    dilations=(),
                    paddings=()):
    enc_stack = CNNStack(nn.Conv2d, input_shape, dropout, activation, pooling,
                         batch_norm, filters, kernels, strides, dilations, paddings)
    last_conv_shape = enc_stack.output_shape[1:]
    enc_stack.extend_(Flatten())
    enc_fc_stack = FCStack(np.prod(enc_stack.output_shape), fc_sizes, nn.ReLU)
    rev_filters = list(reversed([shape[::-1] for shape in filters]))
    dec_fc_stack = FCStack(z_dim, list(fc_sizes[::-1]) + [np.prod(enc_stack.output_shape)], nn.ReLU)
    dec_fc_stack.extend_(UnFlatten(last_conv_shape))
    # no pooling for decoder net
    dec_stack = CNNStack(nn.ConvTranspose2d, dec_fc_stack.output_shape, dropout, activation, None,
                         batch_norm, rev_filters, kernels[::-1], strides[::-1],
                         dilations[::-1], paddings[::-1])
    encoder = CheckedSequential(input_shape, enc_stack, enc_fc_stack)
    decoder = CheckedSequential((1, 1, 1, z_dim), dec_fc_stack, dec_stack)
    return Encoder(encoder, z_dim), decoder
