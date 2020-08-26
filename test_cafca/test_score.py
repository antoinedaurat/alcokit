import numpy as np
from alcokit.score import Score
import unittest


class TestScore(unittest.TestCase):
    def test_constructors(self):
        sc = Score.from_stop([3, 6, 9, 12])

        assert sc.start[0] == 0
        assert sc.stop[3] == 12
        assert sc.stop[0] == 3

        ######################

        sc = Score.from_duration([3, 6, 9, 12])

        assert sc.start[0] == 0
        assert sc.stop[3] == 30
        assert sc.stop[0] == 3

        ######################

        sc = Score.from_frame_definition(20, 4, 2)
        assert sc.start[8] == 16
        assert sc.stop[8] == 20
        assert np.all(sc.duration.values == 4)
        assert sc.duration.sum() == 36

        ######################

        X = np.c_[np.random.rand(64, 100), np.ones((64, 50)), np.random.rand(64, 50)]
        sc = Score.from_recurrence_matrix(X)
        assert sc.start[0] == 0
        assert sc.stop.iloc[-1] == 200
        assert sc.duration.sum() == 200

    def test_methods(self):
        ######################

        sc = Score.from_stop([3, 6, 9, 12])

        assert sc.is_consistent()
        assert sc.is_contiguous()
        sc.stop += 2
        assert not sc.is_consistent()
        assert not sc.is_contiguous()

        sc.update_duration()
        assert sc.is_consistent()
        sc.make_contiguous()
        assert sc.is_contiguous()

        ######################

        sc = Score.from_stop([3, 6, 9, 12])
        sc.add_params(stretch=[1., 2., 1., 2.],
                      shift=[1., -1., 1., -1.])
        assert "stretch" in sc.columns
        assert "shift" in sc.columns

        ######################

    def test_join(self):
        sc = Score.from_stop([3, 6, 9, 12])
        prior_N = sc.span
        sc = sc.join_events((0, 1), (2, 3))
        assert sc.span == prior_N
        assert sc.is_contiguous()
        assert sc.is_consistent()
        assert len(sc) == 2

    def test_split(self):
        sc = Score.from_stop([3, 6, 9, 12])
        prior_span = sc.span
        prior_length = len(sc)
        sc = sc.split_events((0, 2), (1, .3333), (2, [1, 3]))

        assert len(sc) > prior_length
        assert type(sc) is Score
        assert sc.span == prior_span
        assert sc.is_contiguous()
        assert sc.is_consistent()

    def test_shift(self):
        # SPAN DESTRUCTIVE / CHAIN CONSERVING SHIFTS

        sc = Score.from_stop([3, 6, 9, 12])
        prior_span = sc.span
        sc = sc.shift_events(slice(0, None), 1)
        assert sc.span == prior_span - 1
        assert sc.is_contiguous()
        assert sc.is_consistent()

        sc = Score.from_stop([3, 6, 9, 12])
        sc = sc.shift_events(slice(0, None), -2)

        assert sc.span == prior_span - 2
        assert sc.is_contiguous()
        assert sc.is_consistent()

        # SPAN CONSERVING / CHAIN DESTRUCTIVE SHIFTS (with x_ and _x)

        sc = Score.from_stop([3, 6, 9, 12])
        sc = sc.shift_x(slice(0, None), 2, False)

        assert sc.span == prior_span
        assert not sc.is_contiguous()
        assert sc.is_consistent()

        sc = Score.from_stop([3, 6, 9, 12])
        sc = sc.x_shift(slice(0, None), -2, False)

        assert sc.span == prior_span
        assert not sc.is_contiguous()
        assert sc.is_consistent()

        # SPAN CONSERVING / CHAIN CONSERVING SHIFTS (with x_ and _x)

        sc = Score.from_stop([3, 6, 9, 12])
        sc = sc.shift_x(slice(0, None), 2, True)

        assert sc.span == prior_span
        assert sc.is_contiguous()
        assert sc.is_consistent()

        sc = Score.from_stop([3, 6, 9, 12])
        sc = sc.x_shift(slice(0, None), -2, True)

        assert sc.span == prior_span
        assert sc.is_contiguous()
        assert sc.is_consistent()

    def test_extend(self):
        sc1 = Score.from_stop([3, 6, 9, 12])
        span1 = sc1.span
        sc2 = Score.from_stop([3, 6, 9, 12])
        span2 = sc2.span

        sc = sc1.extend(sc2)

        assert sc.span == span1 + span2
        assert len(sc) == len(sc1) + len(sc2)

    def test_queries(self):
        sc = Score.from_stop([3, 6, 9, 12])
        sc.add_params(stretch=[1., 2., 1., 2.],
                      shift=[1., -1., 1., -1.])

        subset = sc.hard_q.or_(stretch=lambda s: s > 1.,
                               shift=lambda s: s > 0)
        assert (subset.index == sc.index).all()

        subset = sc.hard_q.and_(stretch=lambda s: s > 1.,
                                shift=lambda s: s > 0)
        assert len(subset) == 0


if __name__ == '__main__':
    unittest.main()
