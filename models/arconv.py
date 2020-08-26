from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim
import torch
from alcokit.models.modules import ParamedSampler, Pass, Abs
from alcokit.models import Model


class GatedConv(nn.Module):
    def __init__(self, dim, kernel, groups, dilation=1, down=True):
        super(GatedConv, self).__init__()
        mod = nn.Conv1d if down else nn.ConvTranspose1d
        self.conv_f = mod(dim, dim, kernel, stride=1, dilation=dilation, groups=groups)
        self.conv_g = mod(dim, dim, kernel, stride=1, dilation=dilation, groups=groups)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.kernel = kernel
        self.dilation = dilation

    def forward(self, x):
        f = self.tanh(self.conv_f(x))
        g = self.sig(self.conv_g(x))
        return f * g


def cond_conv(conv, x):
    h = conv(x[:, :, :-conv.dilation])
    return torch.cat((x[:, :, :conv.kernel * conv.dilation], h + x[:, :, conv.kernel * conv.dilation:]), dim=-1)


class ARConv(Model):
    def __init__(self, **kwargs):
        super(ARConv, self).__init__(**kwargs)

        self.inpt_conv = nn.Linear(self.ninpt, self.dim)

        self.net_down = nn.Sequential(
            GatedConv(self.dim, 2, self.dheads),
        )

        if self.cond_layers > 0:
            self.conds = nn.ModuleList(
                [
                    GatedConv(self.dim, 2, self.dheads, dilation=2**i)
                    for i in range(1, self.cond_layers+1)
                ]
            )
        else:
            self.conds = []

        self.samp = ParamedSampler(self.dim, self.dim, pre_activation=Pass)

        self.net_up = nn.Sequential(
            GatedConv(self.dim, 2, self.uheads, down=False, dilation=1),
        )

        self.fcout = nn.Sequential(nn.Linear(self.dim, self.ninpt), Abs())

        self.sched_first_epoch = getattr(self, "sched_first_epoch", 0)
        self.sched_last_epoch = getattr(self, "sched_last_epoch", self.max_epochs)

    def forward(self, x):
        x = self.inpt_conv(x)
        x = x.permute(0, 2, 1)
        x = self.net_down(x)
        for i, cv in enumerate(self.conds):
            x = cond_conv(cv, x)
        x = x.permute(0, 2, 1)
        x = self.samp(x)[0]
        x = x.permute(0, 2, 1)
        x = self.net_up(x)
        x = x.permute(0, 2, 1)
        x = self.fcout(x)
        return x

    def training_step(self, batch, batch_idx):
        batch, target = batch
        out = self.forward(batch)
        L = self.loss_fn(out, target)
        self.ep_losses += [L.item()]
        return {"loss": L}

    def configure_optimizers(self):
        super(ARConv, self).configure_optimizers()
        opt = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        if getattr(self, "sched_factor", False) and getattr(self, "sched_step_size", False):
            self.sched = StepLR(opt, gamma=self.sched_factor, step_size=self.sched_step_size)
        return opt

    def on_epoch_end(self):
        super(ARConv, self).on_epoch_end()
        if getattr(self, "sched", False) and self.sched_last_epoch > self.current_epoch >= self.sched_first_epoch:
            self.sched.step()
        print(self.losses[-1])