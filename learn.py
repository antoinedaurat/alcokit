from time import time


class TrainingLoop(object):
    def __init__(self,
                 train,
                 batch_step,
                 test=None,
                 cv_step=None,
                 break_test=None,
                 epoch_callback=None,
                 ):
        self.train = train
        self.batch_step = batch_step
        self.test = test
        self.cv_step = cv_step
        self.break_test = break_test
        self.epoch_callback = epoch_callback
        self.tr_loss = []
        self.ts_loss = []

    def log(self, e, n_batches, dur):
        pass

    def run(self, n_epochs):
        print("Starting Training")
        total_dur = time()
        for e in range(n_epochs):
            start = time()
            ep_loss = [self.batch_step(batch) for batch in self.train]
            self.tr_loss += [sum(ep_loss) / len(ep_loss)]

            if self.test is not None and self.cv_step is not None:
                loss = [self.cv_step(batch) for batch in self.test]
                self.ts_loss += [sum(loss) / len(loss)]

            if self.break_test is not None:
                if self.break_test(e, self.tr_loss, self.ts_loss):
                    break

            if self.epoch_callback is not None:
                self.epoch_callback(e)

            self.log(e, len(ep_loss), time()-start)

        total_dur = time() - total_dur

        return self.tr_loss, self.ts_loss
