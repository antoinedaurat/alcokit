import torch
from functools import partial
from cafca.learn.train import TrainingLoop, default_batch_step
import h5py
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("loading models module with device:", DEVICE)


class Model(object):

    name = "model"

    def __init__(self,
                 hp,
                 module,
                 loss_fn,
                 optimizer,
                 db_path=None):
        self.hp = hp
        self.module = module
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.loop = TrainingLoop(self.batch_step,
                                 None, None,
                                 None,
                                 self.save_checkpoint)

        self.db_path = db_path
        if db_path is not None:
            self.group = self.name + "/"
            with h5py.File(db_path, "r+") as f:
                f.create_group(self.group)
                f[self.group].attrs.update({k: v if type(v) in (float, int, str) else str(v)
                                            for k, v in self.hp.items()})

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def batch_step(self, batch):
        return default_batch_step(self.module, self.loss_fn, self.optimizer, batch)

    def fit(self, train_gen, n_epochs):
        return self.loop.run(train_gen, n_epochs)

    def save_checkpoint(self, e):
        if e > 0 and (e % 1024 == 0 or e == self.hp["n_epochs"]-1):
            with h5py.File(self.db_path, "r+") as f:
                name = self.group + "epoch_" + str(e) + "/"
                if name in f:
                    del f[name]
                dict_ = self.module.state_dict()
                grp = f.create_group(name)
                for k, v in dict_.items():
                    grp.create_dataset(k, data=v.detach().cpu().numpy())

    def save_loss(self):
        with h5py.File(self.db_path, "r+") as f:
            name = self.group + "tr_loss"
            f.create_dataset(name, data=np.array(self.loop.tr_loss))
        return