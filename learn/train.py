from cafca.learn.utils import is_notebook
from time import time
import sys
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
import numpy as np


class DefaultHP(dict):
    def __init__(self, **kwargs):
        defaults = {
            "lr": 1e-2,
            "n_epochs": 1024,
            "batch_size": 64,
            "weight_decay": 0.05,
            "loss_fn": "mse",
        }
        super(DefaultHP, self).__init__(**defaults)
        self.update(kwargs)


def default_batch_step(model, loss_fn, optimizer, batch):
    model.train()
    optimizer.zero_grad()
    predictions = model(batch)
    loss = loss_fn(batch, predictions)
    loss.backward()
    optimizer.step()
    return loss.item()


def is_decreasing(e, tr_losses, ts_losses, n_lasts=50, thresh=1e-1):
    if len(tr_losses) >= 10 and all(np.isnan(x) for x in tr_losses[-10:]):
        return False
    if e < n_lasts:
        return True
    n = min(n_lasts // 2, len(tr_losses) // 2)
    before = tr_losses[-n*2:-n]
    now = tr_losses[-n:]
    diff = (sum(before) / len(before)) - (sum(now) / len(now))
    return diff >= thresh


class LossObserver(ReduceLROnPlateau):
    """
    Hack to to use ReduceLROnPlateau as a simple progress tester
    """
    dummy_optim = optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    def __init__(self, *args, **kwargs):
        super(LossObserver, self).__init__(self.dummy_optim, *args, **kwargs)

    def all_good(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        return self.num_bad_epochs > self.patience


class TrainingLoop(object):
    def __init__(self,
                 batch_step,
                 test=None,
                 cv_step=None,
                 validate=None,
                 epoch_callback=None,
                 ):
        self.batch_step = batch_step
        self.test = test
        self.cv_step = cv_step
        self.validate = validate
        self.epoch_callback = epoch_callback
        self.tr_loss = []
        self.ts_loss = []

    def log(self, e, dur, L):
        pass
        # print("Epoch {} : duration = {:.2f} sec. -- loss = {:.5f}".format(e, dur, L))

    def run(self, train, n_epochs):
        total_dur = time()
        for e in tqdm(range(n_epochs), unit="epoch", file=sys.stdout):
            start = time()
            ep_loss = [self.batch_step(batch=batch) for batch in train]
            self.tr_loss += [sum(ep_loss) / len(ep_loss)]

            if self.test is not None and self.cv_step is not None:
                loss = [self.cv_step(batch) for batch in self.test]
                self.ts_loss += [sum(loss) / len(loss)]

            self.log(e,  time()-start, self.tr_loss[-1])

            if e > 0 and self.validate is not None:
                if not self.validate(e, self.tr_loss, self.ts_loss):
                    print("Epoch {}: Loss not decreasing enough! BREAK!".format(e))
                    print(self.tr_loss[-10:])
                    self.epoch_callback(n_epochs)
                    break

            if self.epoch_callback is not None:
                self.epoch_callback(e)

        print()
        total_dur = time() - total_dur
        print("Done {} epochs in {:.2f} sec. Start Loss : {:.2f} -- Final Loss : {:2f}".format(
            len(self.tr_loss), total_dur, self.tr_loss[0], self.tr_loss[-1]))
        return self.tr_loss, self.ts_loss
