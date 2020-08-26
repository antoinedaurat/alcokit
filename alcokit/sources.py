from functools import partial
import torch
import numpy as np
import pandas as pd
from alcokit.score import Score


class ScoredSourceMixin(object):
    """
    Instanceless class that should be mixed/composed with other objects implementing __getitem__.
    This allow to extend, for instance, numpy arrays, with a collection of methods which take
    scores as arguments and facilitate dealing with data in different contexts (e.g. generation, extraction, modelling...)
    The core idea being : scores define ways of iterating through pieces of data and the result of this iteration
    (in most case, you'll want to map some functional to each piece) should either be returned as a sequence (the collection
    of the resulting pieces), as a concatenated sequence, or as a stack.
    Simply put: this is the interface between the Score class and whatever class you'll have that takes slices as argument in getitem
    """

    # OVERRIDE THOSE TO MIX WITH AN OTHER CLASS:

    time_axis = 0

    @property
    def _constructor(self):
        return object

    @property
    def cat(self):
        return partial(np.concatenate, axis=self.time_axis)

    @property
    def stack(self):
        return np.stack

    def __getitem__(self, item):
        pass

    #################################

    def _format_score(self, score):
        if isinstance(score, pd.DataFrame):
            return score.slices(self.time_axis)
        else:
            return score

    @property
    def _pad_slice(self):
        return [slice(0, None)] * self.time_axis

    def colcat(self, score):
        score = self._format_score(score)
        result = self.cat(tuple(self[i] for i in score))
        return self._constructor(result)

    def uncat(self, score):
        score = self._format_score(score)
        return tuple(self[i] for i in score)

    def colstack(self, score):
        score = self._format_score(score)
        result = self.stack(tuple(self[i] for i in score))
        return self._constructor(result)

    def for_each(self, score: pd.DataFrame, func,
                 collate_fn=None,
                 return_score=False):
        """
        this works only if `score` is a valid DataFrame score
        """
        out = []
        for slice_i, event in zip(score.slices(self.time_axis), score.events):
            out += [func(self[slice_i], event)]
        rv = out if collate_fn is None else collate_fn(out)
        if return_score:
            return rv, Score.from_data(out)
        return rv

    def patch(self, other, overlap):
        start, n = overlap
        a, b, c, d = self[(*self._pad_slice, slice(0, start))], \
                     self[(*self._pad_slice, slice(start, None))], \
                     other[(*self._pad_slice, slice(0, n))], \
                     other[(*self._pad_slice, slice(n, None))]
        m = (b + c,) if b.shape[0] > 0 and c.shape[0] > 0 else tuple()
        return self._constructor(self.cat((a, *m, d)))

    def colpatch(self, score: pd.DataFrame):
        """
        this works only if `score` is a valid DataFrame score
        """
        slices = score.slices(self.time_axis)
        ovls = score.ovl_defs
        result = self[slices[0]]
        for i, ovl in zip(slices[1:], ovls[:-1]):
            result = result.patch(self[i], ovl)
        return self._constructor(result)


class NPSource(np.ndarray, ScoredSourceMixin):

    def __new__(cls, array):
        obj = array.view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    time_axis = 1

    @property
    def _constructor(self):
        return NPSource

    @property
    def cat(self):
        return partial(np.concatenate, axis=self.time_axis)

    @property
    def stack(self):
        return np.stack