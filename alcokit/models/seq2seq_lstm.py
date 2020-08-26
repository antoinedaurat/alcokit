import torch
import torch.nn as nn
import torch.optim as optim
from alcokit.models.modules import ParamedSampler, Pass
from alcokit.models.losses import weighted_L1
from alcokit.models import Model


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
