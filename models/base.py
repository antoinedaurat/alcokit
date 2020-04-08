import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary


class ModuleBase(nn.Module):
    def __init__(self, module, input_shape=None):
        super(ModuleBase, self).__init__()
        self.module = module
        self.input_shape = input_shape
        if self.input_shape is not None and self.module is not None:
            # make a forward pass to get the output shape
            self.output_shape = tuple(self.module(torch.randn(*input_shape)).size())
        else:
            self.output_shape = None

    def forward(self, x):
        return self.module(x)


class ModelBase(ModuleBase):
    def __init__(self, module, input_shape=None):
        super(ModelBase, self).__init__(module, input_shape)
        self.device = None
        self.to_device()
        self.optimizer = None

    def to_device(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
        else:
            self.device = device
            self.to(self.device)

    def get_loss(self, x):
        """stump to be overriden"""
        return self.module(x)

    def one_epoch(self, train_iterator, test_iterator):
        """
        combine the train step and the evaluation step for one epoch and return both losses.
        """
        self.train()
        train_loss = []
        start = time.time()
        for i, x in enumerate(train_iterator):
            self.optimizer.zero_grad()
            loss = self.get_loss(x)
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
                loss = self.get_loss(x)
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
        else:
            return x

    @staticmethod
    def as_output(y):
        if isinstance(y, torch.Tensor):
            with torch.no_grad():
                return y.detach().cpu().numpy()
        else:  # tuple
            with torch.no_grad():
                return y[0].detach().cpu().numpy()

    def fit(self, X, X_cv, batch_size, n_epochs, optimizer, checkpoints_dir=None, n_checkpoints=0):
        # move the input to the right device and type and make iterators
        # train_iterator = DataLoader(self.as_input(X), batch_size=batch_size, shuffle=True)
        train_iterator = X
        if X_cv is not None:
            test_iterator = DataLoader(self.as_input(X_cv), batch_size=batch_size)
        else:
            test_iterator = None
        if checkpoints_dir is not None and n_checkpoints > 0:
            check_freq = round(n_epochs / n_checkpoints)
            get_path = lambda i: checkpoints_dir + "_epoch_" + str(i) + ".pt"
        else:
            check_freq, get_path = False, None
        self.optimizer = optimizer if self.optimizer is None else self.optimizer
        train_loss, test_loss = [], []
        print("Starting training")
        for e in range(n_epochs):
            tr_loss, ts_loss, (dur, avg_dur) = self.one_epoch(train_iterator, test_iterator)
            train_loss += tr_loss
            test_loss += ts_loss
            tr_loss = sum(tr_loss) / len(tr_loss)
            ts_loss = sum(ts_loss) / len(ts_loss)
            print(f'Epoch {e+1} took {dur:.3f}sec [{1000*(avg_dur/batch_size):.3f}ms/sample]. Train Loss: {tr_loss:.3f}, Test Loss: {ts_loss:.3f}')
            if check_freq and e > 0 and ((e % check_freq) == 0 or (e == n_epochs-1)):
                path = get_path(e+1)
                print("Saving model and optimizer in", path)
                self.save(path, save_optimizer=(e == n_epochs - 1))
        return train_loss, test_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.as_output(self.forward(self.as_input(x)))

    def summary(self):
        with torch.no_grad():
            return summary(self, input_size=self.input_shape[1:], device=self.device.type)

    def save(self, path, save_optimizer=False):
        state_dict = self.state_dict()
        optim_dict = self.optimizer.state_dict()
        torch.save({
            "model_state_dict": state_dict,
            "optimizer_state_dict": optim_dict if save_optimizer else {}
        }, path)

    def load(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict["model_state_dict"])
        if state_dict["optimizer_state_dict"]:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.to_device(device)


