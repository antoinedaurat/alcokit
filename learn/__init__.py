import torch
from IPython import get_ipython
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from time import time, gmtime
import os
import shutil


def is_notebook():
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        return True
    elif shell == 'TerminalInteractiveShell':
        return False
    else:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class DefaultHP(dict):
    def __init__(self, **kwargs):
        defaults = dict(
            lr=1e-2,
            max_epochs=1024,
            batch_size=32,
            loss_fn=nn.L1Loss(reduction="none"),
            root_dir="test_model/",
            name="model",
            version="v0",
            overwrite=False,
            era_duration=30.
        )
        super(DefaultHP, self).__init__(**kwargs)
        self.update(defaults)


class Model(pl.LightningModule):
    hp = DefaultHP()

    @property
    def path(self):
        return self.root_dir + self.name + "/" + self.version

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.hp.update(kwargs)
        for k, v in self.hp.items():
            setattr(self, k, v)

    @staticmethod
    def load(clazz, version_dir, epoch=None):
        hp = torch.load(version_dir + "hparams.pt")
        hp["overwrite"] = False
        instance = clazz(**hp)
        sd = torch.load(version_dir + ("epoch=%i.ckpt" % epoch if epoch is not None else "final.ckpt"))
        instance.load_state_dict(sd)
        return instance

    def on_train_start(self):
        self.init_directories()
        self.ep_bar = tqdm(range(1, 1 + self.max_epochs), unit="epoch",
                           position=0, leave=False, dynamic_ncols=True)
        self.ep_losses = []
        self.losses = []
        self.start_time = time()
        self.last_era_time = time()
        self.ep_since_era = 0

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss_fn(batch, output)
        self.ep_losses += [loss.item()]
        return {"loss": loss}

    def on_era_end(self):
        pass

    def on_epoch_end(self):
        ep_loss = sum(self.ep_losses) / len(self.ep_losses)
        self.losses += [ep_loss]
        self.ep_losses = []
        self.ep_bar.update()
        self.ep_since_era += 1
        now = time()
        if now - self.last_era_time >= self.era_duration:
            torch.save(self.state_dict(), self.path + "/epoch=%i.ckpt" % self.current_epoch)
            self.on_era_end()
            self.last_era_time = now

    def on_train_end(self):
        torch.save(self.state_dict(), self.path + "/final.ckpt")
        self.save_loss()
        total_time = gmtime(time() - self.start_time)
        print("Training finished after {0} days {1} hours {2} mins {3} seconds".format(total_time[2] - 1, total_time[3],
                                                                                       total_time[4], total_time[5]))

    def save_loss(self):
        return np.save(self.path + "/tr_losses", np.array(self.losses))

    def get_trainer(self, **kwargs):
        defaults = dict(gpus=1,
                        min_epochs=1,
                        max_epochs=2048,
                        reload_dataloaders_every_epoch=True,
                        checkpoint_callback=False,
                        progress_bar_refresh_rate=5,
                        logger=False,
                        process_position=1,
                        )
        defaults.update(kwargs)
        return pl.Trainer(**defaults)

    def init_directories(self):
        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)
        if not os.path.isdir(self.root_dir + self.name):
            os.mkdir(self.root_dir + self.name)
        if self.version in os.listdir(self.root_dir + self.name) and not self.overwrite:
            while self.version in os.listdir(self.root_dir + self.name):
                self.version = "v" + str(int(self.version[1:]) + 1)
            os.mkdir(self.path)
        else:
            if os.path.isdir(self.path):
                shutil.rmtree(self.path, ignore_errors=True)
            os.mkdir(self.path)
        print("current version:", self.version)
        torch.save(self.hp, self.path + "/hparams.pt")


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cafca initialized with device:", DEVICE)
