from cafca.models import device
from cafca.models.parts import ParamedSampler
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np




H = Z = 128


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.h = H
        self.rnn = nn.RNN(1025, self.h, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.h * 2, Z)

    def first_and_last_states(self, packed_sequence):
        unpacked, lengths = pad_packed_sequence(packed_sequence, batch_first=True)
        indices = torch.LongTensor(np.array(lengths) - 1) \
            .view(-1, 1) \
            .expand(unpacked.size(0), unpacked.size(2)) \
            .unsqueeze(1).to(device)
        last_states = unpacked.gather(dim=1, index=indices).squeeze(dim=1)[:, :self.h]
        zeros = torch.zeros(unpacked.size(0), 1, unpacked.size(2)).long().to(device)
        first_states = unpacked.gather(dim=1, index=zeros).squeeze(dim=1)[:, self.h:]
        return torch.cat((first_states, last_states), dim=-1), lengths

    def forward(self, x):
        packed = pack_sequence(x, enforce_sorted=False)
        states, _ = self.rnn(packed)
        states, lengths = self.first_and_last_states(states)
        return nn.Tanh()(self.fc(states)), lengths


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.h = H
        self.rnn = nn.RNN(Z, self.h, num_layers=2, batch_first=True, bidirectional=True, nonlinearity='tanh')
        self.fc = nn.Linear(self.h, 1025)

    def forward(self, x, lengths):
        states = pack_sequence([state.repeat(n).view(n, -1) for state, n in zip(x, lengths)],
                               enforce_sorted=False)
        output, _ = self.rnn(states)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        output = [x[:n].view(n, -1, 2).sum(dim=-1) for x, n in zip(output, lengths)]
        return [nn.ReLU()(self.fc(x)) for x in output]


class RNNVAE(nn.Module):
    def __init__(self):
        super(RNNVAE, self).__init__()
        self.enc = EncoderRNN().to(device)
        self.dec = DecoderRNN().to(device)
        self.sampler = ParamedSampler(Z, Z).to(device)

    def forward(self, data):
        coded, lengths = self.enc(data)
        coded, mu, logvar = self.sampler(coded)
        output = self.dec(coded, lengths)
        return output, mu, logvar