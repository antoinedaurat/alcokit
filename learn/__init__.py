import torch
from IPython import get_ipython
import h5py
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from time import time, gmtime


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
        defaults = {
            "lr": 1e-2,
            "max_epochs": 1024,
            "batch_size": 32,
            "loss_fn": nn.L1Loss(reduction="none"),
        }
        super(DefaultHP, self).__init__(**defaults)
        self.update(kwargs)


class Model(pl.LightningModule):
    db_path = "model_db.h5"
    name = "model"
    overwrite = False
    hp = DefaultHP()
    ckpt_period = 64

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        pass

    def on_train_start(self):
        self.ep_bar = tqdm(range(1, 1+self.trainer.max_epochs), unit="epoch",
                           position=0, leave=False, dynamic_ncols=True)
        self.ep_losses = []
        self.losses = []
        self.init_db()
        self.start_time = time()

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss_fn(batch, output)
        self.ep_losses += [loss.item()]
        return {"loss": loss}

    def debug_callback(self, e):
        pass

    def on_epoch_end(self):
        ep_loss = sum(self.ep_losses) / len(self.ep_losses)
        self.losses += [ep_loss]
        self.ep_losses = []
        self.ep_bar.update()
        e = self.current_epoch
        if e > 0 and self.ckpt_period > 0 and e % self.ckpt_period == 0:
            self.save_checkpoint(e)
        self.debug_callback(e)

    def on_train_end(self):
        self.save_checkpoint(self.current_epoch)
        self.save_loss()
        total_time = gmtime(time() - self.start_time)
        print("Training finished after {0} days {1} hours {2} mins {3} seconds".format(total_time[2]-1, total_time[3],
                                                                                       total_time[4], total_time[5]))

    def init_db(self):
        db_path = self.db_path
        if db_path is not None:
            self.name += "/v0/"
            group = self.name
            try:
                f = h5py.File(db_path, "r")
                f.close()
            except OSError:
                with h5py.File(db_path, "w") as f:
                    print("created model Database", db_path)
            with h5py.File(db_path, "r+") as f:
                if group in f:
                    if self.overwrite:
                        del f[group]
                    else:
                        v = 0
                        while group in f:
                            v += 1
                            group = group.split("/")[0] + "/v" + str(v)
                        self.name = group
                f.create_group(group)
                f[group].attrs.update({k: v if type(v) in (float, int, str) else str(v)
                                       for k, v in self.hp.items()})

    def save_checkpoint(self, e):
        with h5py.File(self.db_path, "r+") as f:
            name = self.name
            name += "/epoch_" + str(e) + "/"
            dict_ = self.state_dict()
            grp = f.create_group(name)
            for k, v in dict_.items():
                grp.create_dataset(k, data=v.detach().cpu().numpy())

    def save_loss(self):
        with h5py.File(self.db_path, "r+") as f:
            name = self.name + "/tr_loss"
            f.create_dataset(name, data=np.array(self.losses))
        return


def default_trainer():
    return pl.Trainer(gpus=1,
                      min_epochs=512,
                      max_epochs=2048,
                      reload_dataloaders_every_epoch=True,
                      checkpoint_callback=False,
                      progress_bar_refresh_rate=5,
                      logger=False,
                      )


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cafca initialized with device:", DEVICE)