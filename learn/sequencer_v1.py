import torch
import torch.nn as nn
import torch.optim as optim
from cafca.learn.modules import ParamedSampler, Pass
from cafca.learn.losses import weighted_L1
from cafca.learn import Model


class EncoderLSTM(nn.Module):
    def __init__(self, input_d, model_dim, num_layers, bottleneck="add", n_fc=1):
        super(EncoderLSTM, self).__init__()
        self.bottleneck = bottleneck
        self.dim = model_dim
        self.lstm = nn.LSTM(input_d, self.dim if bottleneck == "add" else self.dim // 2,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(self.dim, self.dim), nn.Tanh()) for _ in range(n_fc - 1)],
            nn.Linear(self.dim, self.dim),  # NO ACTIVATION !
        )

    def forward(self, x, hiddens=None, cells=None):
        if hiddens is None or cells is None:
            states, (hiddens, cells) = self.lstm(x)
        else:
            states, (hiddens, cells) = self.lstm(x, (hiddens, cells))
        states = self.first_and_last_states(states)
        return self.fc(states), (hiddens, cells)

    def first_and_last_states(self, sequence):
        sequence = sequence.view(*sequence.size()[:-1], self.dim, 2).sum(dim=-1)
        first_states = sequence[:, 0, :]
        last_states = sequence[:, -1, :]
        if self.bottleneck == "add":
            return first_states + last_states
        else:
            return torch.cat((first_states, last_states), dim=-1)


class DecoderLSTM(nn.Module):
    def __init__(self, model_dim, num_layers, bottleneck="add"):
        super(DecoderLSTM, self).__init__()
        self.dim = model_dim
        self.lstm1 = nn.LSTM(self.dim, self.dim if bottleneck == "add" else self.dim // 2,
                             num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.dim, self.dim if bottleneck == "add" else self.dim // 2,
                             num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, hiddens, cells):
        if hiddens is None or cells is None:
            output, (_, _) = self.lstm1(x)
        else:
            output, (_, _) = self.lstm1(x, (hiddens, cells))  # V1 decoder DOES GET hidden states from enc !
        output = output.view(*output.size()[:-1], self.dim, 2).sum(dim=-1)

        if hiddens is None or cells is None:
            output2, (hiddens, cells) = self.lstm2(output)
        else:
            output2, (hiddens, cells) = self.lstm2(output, (hiddens, cells))  # V1 residual DOES GET hidden states from first lstm !
        output2 = output2.view(*output2.size()[:-1], self.dim, 2).sum(dim=-1)

        return output + output2, (hiddens, cells)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, model_dim,
                 num_layers=1,
                 bottleneck="add",
                 n_fc=1):
        super(Seq2SeqLSTM, self).__init__()
        self.enc = EncoderLSTM(input_dim, model_dim, num_layers, bottleneck, n_fc)
        self.dec = DecoderLSTM(model_dim, num_layers, bottleneck)
        self.sampler = ParamedSampler(model_dim, model_dim, pre_activation=Pass)

    def forward(self, x, output_length=None):
        coded, (h_enc, c_enc) = self.enc(x)
        if output_length is None:
            output_length = x.size(1)
        coded = coded.unsqueeze(1).repeat(1, output_length, 1)
        residuals, _, _ = self.sampler(coded)
        coded = coded + residuals
        output, (_, _) = self.dec(coded, h_enc, c_enc)
        return output


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
        self.enc = EncoderLSTM(self.input_d, self.h, self.num_layers, self.bottleneck, self.n_fc)
        self.dec = DecoderLSTM(self.h, self.num_layers, self.bottleneck)
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
