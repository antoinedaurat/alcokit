import torch
import torch.nn as nn
from alcokit.models import Model
from alcokit.models.data import load
from alcokit.models.seq2seq_lstm import Seq2SeqLSTM
from alcokit.models.conv_network import ConvolutionNetwork
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


class CooledAdam(optim.Adam):

    def zero_avg(self, step=10000):
        for p_group in self.param_groups:
            for p in p_group["params"]:
                self.state[p]["exp_avg"].zero_()
                self.state[p]["exp_avg_sq"].zero_()
                self.state[p]["step"] = step


class QuietOneCycleLR(OneCycleLR):

    def step(self, epoch=None):
        try:
            super(QuietOneCycleLR, self).step(epoch=epoch)
        except ValueError:
            pass


def mean_L1_prop(output, target):
    L = nn.L1Loss(reduction="none")(output, target).sum(dim=(0, -1), keepdim=True)
    return 100 * (L / target.sum(dim=(0, -1), keepdim=True)).mean()


class Head(nn.Module):
    def __init__(self, input_dim, output_dim,
                 scaled=False, layer=nn.Linear, non_linearity="exp",
                 **layer_kwargs):
        super(Head, self).__init__()
        self.layer = layer(input_dim, output_dim + int(scaled), **layer_kwargs)
        self.scaled = scaled
        nl = {"abs": torch.abs, "exp": torch.exp, "softmax": nn.Softmax(dim=-1)}
        self.non_linearity = nl[non_linearity]

    def forward(self, x):
        x = self.layer(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.scaled:
            return self.non_linearity(x[:, :, :-1]) * x[:, :, -1:].abs()
        return self.non_linearity(x)


class AutoRegressiveModel(Model):
    def __init__(self,
                 input_dim,
                 model_dim,
                 architecture,
                 net_kwargs,
                 head_kwargs=dict(scaled=True, layer=nn.Linear, non_linearity=torch.exp),
                 database=None,
                 train_set=None,
                 optim_kwargs=dict(lr=1e-3, betas=(.9, .9)),
                 epoch_restart=False,
                 scheduler_kwargs=dict(max_lr=1e-3, div_factor=3., final_div_factor=1, pct_start=.25,
                                       cycle_momentum=False, epochs=100),
                 loss_fn=mean_L1_prop,
                 max_epochs=100,
                 batch_length=64,
                 **kwargs):
        super(AutoRegressiveModel, self).__init__(input_dim=input_dim, model_dim=model_dim,
                                                  architecture=architecture, net_kwargs=net_kwargs,
                                                  head_kwargs=head_kwargs,
                                                  database=database, train_set=train_set,
                                                  loss_fn=loss_fn, max_epochs=max_epochs,
                                                  optim_kwargs=optim_kwargs, epoch_restart=epoch_restart,
                                                  scheduler_kwargs=scheduler_kwargs,
                                                  batch_length=batch_length,
                                                  **kwargs)
        if architecture == "conv":
            self.input = nn.Linear(input_dim, model_dim)
            self.net = ConvolutionNetwork(model_dim, **net_kwargs)
            self.receptive_field = self.net.receptive_field()
            self.future_length = self.net.future_length()
        elif architecture == "lstm":
            self.input = nn.Identity()
            self.net = Seq2SeqLSTM(input_dim, model_dim, **net_kwargs)
            self.receptive_field = batch_length
            self.future_length = batch_length
        self.head = Head(model_dim, input_dim, **head_kwargs)
        self.dl, self.sched, self.opt = None, None, None

    def forward(self, x):
        x = self.input(x)
        x = self.net(x)
        x = self.head(x)
        return x

    def prepare_data(self):
        # if not os
        db = self.database
        self.dl = load((db.fft, db.fft), self.train_set.copy(), "frame",
                       k=self.batch_length, stride=1, shifts=(0, self.receptive_field),
                       batch_size=self.batch_size, shuffle=True,
                       pre_cat=True, device="cuda")

    def train_dataloader(self):
        return self.dl

    def configure_optimizers(self):
        super(AutoRegressiveModel, self).configure_optimizers()
        kwargs = self.optim_kwargs
        self.opt = CooledAdam(self.parameters(), **kwargs)
        kwargs = self.scheduler_kwargs
        self.sched = QuietOneCycleLR(self.opt, steps_per_epoch=len(self.dl), **kwargs)
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        batch, target = batch
        out = self.forward(batch)
        L = self.loss_fn(out, target[:, :out.size(1)])
        self.ep_losses += [L.item()]
        return {"loss": L}

    def on_epoch_end(self):
        super(AutoRegressiveModel, self).on_epoch_end()
        if self.epoch_restart:
            self.opt.zero_avg()

    def sample(self, x, sample_length, input_length=None, output_length=None, reverse=False):
        pass

    @staticmethod
    def load(version_dir, epoch=None):
        return Model.load(AutoRegressiveModel, version_dir, epoch)
