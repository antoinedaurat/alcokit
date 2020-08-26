import unittest
from alcokit.sources import NPSource, TorchSource
import numpy as np
import torch
from alcokit.score import Score


class TestSources(unittest.TestCase):

    def test_np_persistent_type(self):
        arr = np.arange(36).reshape(6, 6)
        nps = NPSource(arr)

        assert isinstance(nps.flat[:], NPSource)

        assert isinstance(nps[0] + 1, NPSource)

        assert isinstance(nps[:, -1] * 8, NPSource)

    def test_torch_persistent_type(self):
        tsr = torch.arange(36).view(6, 6).float()
        ts = TorchSource(tsr)

        # TODO: Unfortunately not supported in pytorch yet...
        # assert isinstance(ts.view(-1), TorchSource)
        #
        # assert isinstance(ts[0] + 1, TorchSource)
        #
        # assert isinstance(ts[:, -1] * 8, TorchSource)

    def test_algos_on_arrays(self):
        from alcokit.alcokit.algos import harmonic_percussive, REP_SIM
        nps = NPSource(np.random.rand(256, 100))
        h, p = harmonic_percussive(nps)

        assert isinstance(h, np.ndarray)
        assert isinstance(p, np.ndarray)

        assert isinstance(h, NPSource)
        assert isinstance(p, NPSource)

        bg, fg = REP_SIM(nps)

        assert isinstance(bg, NPSource)
        assert isinstance(fg, NPSource)

    def test_scored_source_methods(self):
        ds = NPSource(np.arange(36).reshape(6, 6))

        assert isinstance(ds[0], NPSource)

        assert isinstance(ds[slice(0, 4)], NPSource)

        score = Score.from_duration([2, 2, 2])

        colcat = ds.colcat(score)
        assert isinstance(colcat, NPSource)
        assert colcat.shape[1] == 6

        uncat = ds.uncat(score)
        assert len(uncat) == 3

        colstack = ds.colstack(score)
        assert colstack.shape[0] == 3

        score["stop"] += 2
        patched = ds.patch(ds[:, -4:], (3, 3))
        assert patched.shape[1] == 7


if __name__ == '__main__':
    unittest.main()