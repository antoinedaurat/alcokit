from alcokit.models.data import *
from alcokit.hdf.factory import file_to_db
from alcokit.hdf.api import Database
from alcokit.score import Score
import unittest
import os

# Test data

X_np = np.random.rand(333, 111)
Y_np = np.random.rand(333, 3)
X_torch = torch.from_numpy(X_np)
Y_torch = torch.from_numpy(Y_np)

score = Score.from_duration([111, 111, 111])
subscore = score.iloc[[0, 2]].copy()


# Test DB

def extract_func(path):
    return {"X": ({}, X_np), "Y": ({}, Y_np), "score": ({}, score)}


db_path = file_to_db("testdb", extract_func, "w")
db = Database(db_path)
X_h5 = db.X
Y_h5 = db.Y


class TestData(unittest.TestCase):

    def test_flat_sampler(self):
        smp = FlatSampler(score)

        # contiguous score
        assert len(smp) == 333
        assert len([i for i in smp]) == 333

        # discontinuous score
        smp = FlatSampler(subscore)
        assert len(smp) == 222
        assert len([i for i in smp]) == 222

    def test_event_sampler(self):
        smp = EventSampler(score)
        idx = [i for i in smp]

        # contiguous score
        assert len(smp) == 3
        assert len(idx) == 3
        assert all("Event" in str(type(i)) for i in smp)

        # discontinuous score
        smp = EventSampler(subscore)
        idx = [i for i in smp]

        assert len(smp) == 2
        assert len(idx) == 2
        assert all("Event" in str(type(i)) for i in smp)

    def test_frame_sampler(self):
        smp = FrameSampler(N=333, k=5, stride=1, shifts=(0, 4))
        idx = [i for i in smp]

        assert len(smp) == 333-5-4+1
        assert len(idx) == 333-5-4+1
        assert all(type(i) is slice for i in smp)

    def test_numpy_set(self):
        pass
        # Single Feature

        # Multi Feature

    def test_torch_set(self):
        pass

    def test_opened_ds(self):
        pass

    def test_prepare(self):
        pass

    def test_load(self):
        # Single feature np
        dl = load(X_np, score, "flat",
                  batch_size=32,
                  pre_cat=False, collate_fn=zip_stack)

        N = sum([b[0].size(0) for b in dl])
        assert N == 333, N

        # Single feature HDF
        dl = load(X_h5, score, "flat",
                  batch_size=32,
                  pre_cat=False, collate_fn=zip_stack)

        N = sum([b[0].size(0) for b in dl])
        assert N == 333, N

        # Single feature HDF PRECAT SUBSCORE
        dl = load(X_h5, subscore, "flat",
                  batch_size=32,
                  pre_cat=True, collate_fn=zip_stack)

        N = sum([b[0].size(0) for b in dl])
        assert N == 222, N

        # MULTI feature HDF
        dl = load((X_h5, Y_h5), score, "flat",
                  batch_size=32,
                  pre_cat=False, collate_fn=zip_stack)
        xx = [b for b in dl][0]
        N = sum([b[0].size(0) for b in dl])
        assert N == 333, N

        # Single feature HDF
        dl = load(X_h5, score, "event",
                  batch_size=32,
                  pre_cat=False, collate_fn=zip_stack)

        N = sum([b[0].size(0) for b in dl])
        assert N == 3, N

        # MULTI feature HDF
        dl = load((X_h5, Y_h5), score,
                  "frame", k=11, stride=1, shifts=(0, 3),
                  batch_size=32,
                  pre_cat=False, collate_fn=zip_stack)
        xx = [b for b in dl][0]
        N = sum([b[0].size(0) for b in dl])
        assert N == 333-11-3+1, N


if __name__ == '__main__':
    unittest.main()

    # clean up
    os.remove(db_path)