from cafca.learn import DEVICE as DEVICE
from cafca.learn.utils import numcpu, to_torch
import torch
import torch.nn as nn
from torchsummary import summary


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.device = None
        self.to_device()

    def to_device(self, device=None):
        if device is None:
            self.device = DEVICE
            self.to(self.device)
        else:
            self.device = device
            self.to(self.device)

    def preprocess(self, x):
        return to_torch(x)

    @staticmethod
    def numcpu(y):
        return numcpu(y)

    def predict(self, x):
        self.eval()
        return self.numcpu(self.forward(self.preprocess(x)))

    def summary(self, input_shape=None):
        with torch.no_grad():
            return summary(self, input_size=input_shape, device=self.device.type)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save({
            "model_state_dict": state_dict,
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=DEVICE)
        self.load_state_dict(state_dict["model_state_dict"])
        self.to_device(DEVICE)


