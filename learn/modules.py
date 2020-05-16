import torch
import torch.nn as nn
import torch.distributions as D


class Pass(nn.Module):
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, output_shape: tuple):
        super(UnFlatten, self).__init__()
        self.output_shape = output_shape

    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), *self.output_shape)


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class FcUnit(nn.Module):
    def __init__(self, n_in, n_out, batch_norm, activation, dropout):
        super(FcUnit, self).__init__()
        modules = [x for x in [
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out) if batch_norm else None,
            activation() if activation else None,
            nn.Dropout(dropout) if dropout else None]
                   if x is not None]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class FcStack(nn.Module):
    def __init__(self, sizes, batch_norm, activation, dropout):
        super(FcStack, self).__init__()
        sizes = zip(sizes[:-1], sizes[1:])
        self.module = nn.Sequential(*[FcUnit(n_in, n_out, batch_norm, activation, dropout)
                                      for n_in, n_out in sizes])

    def forward(self, x):
        return self.module(x)


class ConvUnit(nn.Module):
    def __init__(self, f, k, s, p, d, batch_norm, activation, pool, dropout):
        super(ConvUnit, self).__init__()
        modules = [x for x in [
            nn.Conv2d(f[0], f[1], k, s, p, d),
            nn.BatchNorm2d(f[1]) if batch_norm else None,
            activation() if activation else None,
            pool if pool else None,
            nn.Dropout(dropout) if dropout else None]
                   if x is not None]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class TConvUnit(nn.Module):
    def __init__(self, f, k, s, p, d, batch_norm, activation, pool, dropout):
        super(TConvUnit, self).__init__()
        modules = [x for x in [
            pool if pool else None,
            nn.ConvTranspose2d(f[1], f[0], k, s, p, dilation=d),
            nn.BatchNorm2d(f[0]) if batch_norm else None,
            activation() if activation else None,
            nn.Dropout(dropout) if dropout else None]
                   if x is not None]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class PoolDown(nn.Module):
    def __init__(self, *args):
        super(PoolDown, self).__init__()
        self.pool = nn.MaxPool2d(*args, return_indices=True)
        self.idx = None

    def forward(self, x):
        maxs, self.idx = self.pool(x)
        return maxs


class PoolUp(nn.Module):
    def __init__(self, pooldown, *args):
        super(PoolUp, self).__init__()
        self.pool = nn.MaxUnpool2d(*args)
        self.pooldown = pooldown

    def forward(self, x):
        return self.pool(x, self.pooldown.idx)


def poolpair(*args):
    down = PoolDown(*args)
    return down, PoolUp(down, *args)


def convpair(f, k, s, p, d, batch_norm, activation, poolargs, dropout):
    poolp = poolpair(*poolargs) if poolargs else (False, False)
    conv = ConvUnit(f, k, s, p, d, batch_norm, activation, poolp[0], dropout)
    conv_t = TConvUnit(f, k, s, p, d, batch_norm, activation, poolp[1], dropout)
    return conv, conv_t


class ParamedSampler(nn.Module):
    """
    Parametrized Sampler for Variational Auto-Encoders
    """

    def __init__(self, input_dim: int, z_dim: int, pre_activation=nn.Tanh):
        super(ParamedSampler, self).__init__()
        self.fc1 = nn.Linear(input_dim, z_dim)
        self.fc2 = nn.Linear(input_dim, z_dim)
        self.z_dim = z_dim
        self.pre_act = pre_activation() if pre_activation is not None else lambda x: x

    def forward(self, h):
        mu, logvar = self.fc1(self.pre_act(h)), self.fc2(self.pre_act(h))
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size(), device=self.get_device())
        z = mu + std * eps
        return z, mu, logvar

    def get_device(self):
        return next(self.parameters()).device.type


class MultivariateImaginaryInput(D.MultivariateNormal):
    def __init__(self, n):
        self.mu = nn.Parameter(torch.rand(n), requires_grad=True)
        sigma = torch.rand(n, n)
        with torch.no_grad():
            sigma *= sigma.T
            sigma += n * torch.eye(n)
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        super(MultivariateImaginaryInput, self).__init__(self.mu, self.sigma)

    def parameters(self):
        return iter([self.mu, self.sigma])


class ImaginaryInputMixture(nn.Module):
    def __init__(self, s, n):
        super(ImaginaryInputMixture, self).__init__()
        self.s, self.event_shape = s, n if isinstance(n, tuple) else (n,)
        self.alpha = nn.Parameter(torch.rand(s), requires_grad=True)
        self.planes = {i: MultivariateImaginaryInput(n) for i in range(s)}

    def forward(self, shape):
        # change to torch.mm for efficiency...
        output = torch.zeros(*shape, *self.event_shape)
        for s in range(self.s):
            output += self.alpha[s] * self.planes[s].rsample(shape)
        return output

    def rsample(self, shape):
        return self.forward(shape)

    def parameters(self):
        for imin in self.planes.values():
            for param in imin.parameters():
                yield param
        yield self.alpha


class ImaginaryPopulation(nn.Module):
    def __init__(self, k, input_class, **kwargs):
        super(ImaginaryPopulation, self).__init__()
        self.k = k
        self.planes = {i: input_class(**kwargs) for i in range(k)}

    def forward(self, k):
        output = torch.zeros(*k.size(), *self.planes[0].event_shape)
        for i, imin in self.planes.items():
            where = k == i
            output[where] = imin.rsample((where.sum(),))
        return output

    def parameters(self):
        for imin in self.planes.values():
            for param in imin.parameters():
                yield param
