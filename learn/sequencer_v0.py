import torch
import torch.nn as nn
import torch.optim as optim
from cafca.learn.modules import ParamedSampler, Pass
from cafca.learn.losses import weighted_L1
from cafca.learn import Model, DefaultHP


class EncoderRNN(nn.Module):
    def __init__(self, input_d, H, num_layers, bottleneck="add", n_fc=1):
        super(EncoderRNN, self).__init__()
        self.bottleneck = bottleneck
        self.h = H

        self.rnn = nn.LSTM(input_d, self.h if bottleneck == "add" else self.h // 2,
                           num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(self.h, self.h), nn.Tanh()) for _ in range(n_fc - 1)],
            nn.Linear(self.h, self.h),  # NO ACTIVATION !
        )

    def forward(self, x, hiddens=None, cells=None):
        if hiddens is None or cells is None:
            states, (hiddens, cells) = self.rnn(x)
        else:
            states, (hiddens, cells) = self.rnn(x, (hiddens, cells))
        states = self.first_and_last_states(states)
        return self.fc(states), (hiddens, cells)

    def first_and_last_states(self, sequence):
        sequence = sequence.view(*sequence.size()[:-1], self.h, 2).sum(dim=-1)
        first_states = sequence[:, 0, :]
        last_states = sequence[:, -1, :]
        if self.bottleneck == "add":
            return first_states + last_states
        else:
            return torch.cat((first_states, last_states), dim=-1)


class DecoderRNN(nn.Module):
    def __init__(self, H, num_layers, bottleneck="add"):
        super(DecoderRNN, self).__init__()
        self.h = H
        self.rnn1 = nn.LSTM(self.h, self.h if bottleneck == "add" else self.h // 2,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(self.h, self.h if bottleneck == "add" else self.h // 2,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, hiddens, cells):
        if hiddens is None or cells is None:
            output, (_, _) = self.rnn1(x)
        else:
            output, (_, _) = self.rnn1(x, (hiddens, cells))
        output = output.view(*output.size()[:-1], self.h, 2).sum(dim=-1)

        if hiddens is None or cells is None:
            output2, (hiddens, cells) = self.rnn2(x)
        else:
            output2, (hiddens, cells) = self.rnn2(x, (hiddens, cells))
        output2 = output2.view(*output2.size()[:-1], self.h, 2).sum(dim=-1)

        return output + output2, (hiddens, cells)


class Sequencer(Model):

    def __init__(self,
                 # instance
                 root_dir="test_model/",
                 name="sequencer",
                 version="v0",
                 overwrite=False,

                 # module
                 input_d=1025,
                 h=512,
                 num_layers=1,
                 bottleneck="add",
                 n_fc=1,
                 loss_fn=weighted_L1,

                 # training
                 batch_size=16,
                 inpt_len=8,
                 trgt_len=8,
                 lr=1e-4,
                 max_epochs=8192,
                 **kwargs):
        args_d = dict(root_dir=root_dir, name=name, version=version, overwrite=overwrite,
                      input_d=input_d, h=h, num_layers=num_layers, bottleneck=bottleneck, n_fc=n_fc, loss_fn=loss_fn,
                      batch_size=batch_size, inpt_len=inpt_len, trgt_len=trgt_len, lr=lr, max_epochs=max_epochs)
        # Model creates object's attributes for each keyword param
        super(Sequencer, self).__init__(**args_d, **kwargs)

        # modules
        self.enc = EncoderRNN(self.input_d, self.h, self.num_layers, self.bottleneck, self.n_fc)
        self.dec = DecoderRNN(self.h, self.num_layers, self.bottleneck)
        self.sampler = ParamedSampler(self.h, self.h, pre_activation=Pass)
        self.sampler_out = ParamedSampler(self.h, self.input_d, pre_activation=Pass)

    def forward(self, x):
        coded, (h_enc, c_enc) = self.enc(x)
        coded = coded.unsqueeze(1).repeat(1, self.trgt_len, 1)
        residuals, mu, logvar = self.sampler(coded)
        coded = coded + residuals
        output, (h_dec, c_dec) = self.dec(coded, h_enc, c_enc)
        output = self.sampler_out(output)[0]
        output = output.abs()
        return output, (h_enc, c_enc), (h_dec, c_dec)

    def training_step(self, batch, batch_idx):
        inpt, trgt = batch
        output, _, _ = self.forward(inpt)
        L = sum(self.loss_fn(x.T, y.T) / x.size(0) for x, y in zip(trgt, output))
        L /= len(batch)
        self.ep_losses += [L.item()]
        return {"loss": L}

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=self.lr)
