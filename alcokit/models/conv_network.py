import torch
import torch.nn as nn
from alcokit.models.modules import ParamedSampler


class GatedConv(nn.Module):
    def __init__(self, dim, kernel, groups, stride=1, dilation=1, padding=0, down=True):
        super(GatedConv, self).__init__()
        mod = nn.Conv1d if down else nn.ConvTranspose1d
        self.conv_f = mod(dim, dim, kernel, stride=stride, dilation=dilation, padding=padding, groups=groups)
        self.conv_g = mod(dim, dim, kernel, stride=stride, dilation=dilation, padding=padding, groups=groups)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        f = self.tanh(self.conv_f(x))
        g = self.sig(self.conv_g(x))
        return f * g


class TimeResidualBlock(nn.ModuleList):
    def __init__(self, *modules):
        super(TimeResidualBlock, self).__init__(modules)

    def forward(self, x):
        for conv in self:
            h = conv(x[:, :, :-conv.dilation * conv.stride])
            h = torch.repeat_interleave(h, torch.tensor([conv.stride] * h.size(-1)), dim=-1)
            x = torch.cat((x[:, :, :conv.kernel * conv.dilation],
                           h + x[:, :, conv.kernel * conv.dilation:]), dim=-1)
        return x


def permute_time_freq(x):
    return x.permute(0, 2, 1).contiguous()


class ConvBlock(nn.Module):
    def __init__(self, dim, n_layers=1, kernel=2, groups=1, stride=1, dilated=False, down=True, forward="sequential"):
        super(ConvBlock, self).__init__()
        self.forward_method = forward
        if self.forward_method == "sequential":
            self.block = nn.Sequential(
                *[GatedConv(dim, kernel, groups, stride=stride,
                            dilation=2 ** i if dilated else 1, down=down) for i in range(n_layers)]
            )
        elif self.forward_method == "residual":
            self.block = TimeResidualBlock(
                *[GatedConv(dim, kernel, groups, stride=stride,
                            dilation=2 ** i if dilated else 1, down=down) for i in range(n_layers)]
            )
        else:
            raise ValueError("`forward` argument not recognized. Must be one of 'sequential', 'residual'")

    def forward(self, x):
        """
        input shape : (batch_size x T x model_dim)
        output shape : (batch_size x T' x model_dim)
        """
        x = permute_time_freq(x)
        x = self.block(x)
        x = permute_time_freq(x)
        return x

    def receptive_field(self):
        modules = self.block if isinstance(self.block, TimeResidualBlock) else list(self.block.children())
        rf = modules[0].kernel
        for cv in modules[1:]:
            if cv.stride == cv.kernel:
                rf *= (cv.kernel - 1) * cv.stride
            else:
                rf += (cv.kernel - 1) * cv.dilation
        return rf


class ConvolutionNetwork(nn.Sequential):
    def __init__(self, dim, symmetrical=False, core=ParamedSampler, **conv_blocks_kwargs):
        if symmetrical:
            if "down" in conv_blocks_kwargs:
                conv_blocks_kwargs.pop("down")
            super(ConvolutionNetwork, self).__init__(
                ConvBlock(dim, **conv_blocks_kwargs, down=True),
                ParamedSampler(dim, dim, return_params=False),
                ConvBlock(dim, **conv_blocks_kwargs, down=False)
            )
        else:
            super(ConvolutionNetwork, self).__init__(
                ConvBlock(dim, **conv_blocks_kwargs)
            )

    def receptive_field(self):
        return next(self.children()).receptive_field()

    def future_length(self):
        n_children = len(list(self.children()))
        if n_children > 1:
            return self.receptive_field()
        return 1