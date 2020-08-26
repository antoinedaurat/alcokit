import pandas as pd
import numpy as np
import torch
from alcokit.extract.segmentation import from_recurrence_matrix


def ssts(item, axis=0):
    """
    Start-Stop To Slices
    @param item: a pd.DataFrame with a "start" and a "stop" column (typically the metadata of a db)
    @param axis:
    @return: 1d array of slices for retrieving each element in `item`
    """
    arr = np.atleast_2d(item[['start', 'stop']].values)
    slices = map(lambda a: (*[slice(0, None)] * axis, slice(a[0], a[1]), ), arr)
    return slices


class Score(pd.DataFrame):

    # OVERRIDES

    @property
    def _constructor(self):
        return Score

    @staticmethod
    def _validate(obj):
        if 'start' not in obj.columns or "stop" not in obj.columns:
            raise ValueError("Must have 'start' and 'stop' columns.")

    # PROPERTIES

    @property
    def first_start(self):
        return self.start.min()

    @property
    def last_stop(self):
        return self.stop.max()

    @property
    def span(self):
        return self.last_stop - self.first_start

    @property
    def cumdur(self):
        return np.cumsum(self["duration"].values)

    @property
    def flat(self):
        return np.array([i for ev in self.events for i in range(ev.start, ev.stop)])

    def slices(self, time_axis=0):
        """
        This is the back-end core of a score. This method efficiently returns the indexing objects
        necessary to communicate with n-dimensional data.

        @return: an array of slices where slice_i corresponds to the row/Event_i in the DataFrame
        """
        return ssts(self, time_axis)

    @property
    def durations_(self):
        return self.stop.values - self.start.values

    @property
    def ovl_defs(self):
        last = self.stop.values[-1]
        ovl = np.array([(a, b - a)
                        for a, b in zip(self.start.values[1:], self.stop.values[:-1])]
                       + [(last, 0)])
        return ovl

    @property
    def events(self):
        return self.itertuples(name="Event", index=True)

    # UPDATING METHOD

    def make_contiguous(self):
        self.reset_index(drop=True, inplace=True)
        cumdur = self.cumdur
        self.loc[:, "start"] = np.r_[0, cumdur[:-1]] if cumdur[0] != 0 else cumdur[:-1]
        self.loc[:, "stop"] = cumdur
        return self

    def update_duration(self):
        self.reset_index(drop=True, inplace=True)
        self.loc[:, "duration"] = self.stop.values - self.start.values
        return self

    def clean_up(self):
        """remove Events with 0 duration"""
        self.drop(index=(self.duration.values == 0).nonzero()[0], inplace=True)
        self.reset_index(drop=True, inplace=True)
        return self

    # Sanity Checks

    def is_consistent(self):
        return (self["duration"].values == (self.stop.values - self.start.values)).all()

    def is_contiguous(self):
        return (self.start.values[1:] == self.stop.values[:-1]).all()

    # MUTATING METHODS

    def add_params(self, **kwargs):
        for name, val in kwargs.items():
            self.loc[:, name] = val
        return self

    def add_duration(self):
        return self.update_duration()

    def add_stretch(self, default=1.):
        return self.add_params(stretch=default)

    def add_stretch_from_target_duration(self, targets):
        targets = np.asarray(targets)
        return self.add_stretch(default=self.durations_ / targets)

    def add_shift(self, default=0.):
        return self.add_params(shift=default)

    def add_target_duration(self):
        if "duration" not in self.columns or "stretch" not in self.columns:
            raise ValueError("Must contain the columns 'duration' and 'stretch' to compute 'target_duration'")
        self.loc[:, "target_duration"] = self.durations_ / self.stretch.values
        return self

    def cut_at(self, *frame_i):
        """
        adds events whose stop are at each frame_i and starts at the first stop < frame_i
        @param frame_i:
        @return:
        """
        pass

    def uncut_at(self, *event_i):
        """
        remove the events with indices `events_i` by merging with them with the next one
        @param event_i:
        @return:
        """
        pass

    def join_events(self, *event_i):
        for i in event_i:
            first = i.start if type(i) is slice else min(i)
            others = list(range(i.start + 1, i.stop)) if type(i) is slice else list(set(i) - {first})
            self.loc[first, "stop"] = self.loc[others, "stop"].max()
            if "duration" in self.columns:
                self.loc[first, "duration"] = self.loc[first, "stop"] - self.loc[first, "start"]
            self.drop(index=others, inplace=True)
        self.reset_index(drop=True, inplace=True)
        return self

    def split_events(self, *splits_def):
        """
        @param sp_def: some iterable of iterables where :
                - the first element is the index of the event to split,
                - the rest of the iterable can be:
                    - an int: number of splits of equal length to perform (any reminder will be added to the last split)
                    - a float: floor(1 / x) splits of equal length will be performed (the reminder will be added to the last split)
                    - a list/iterable of stops: will split the event at those specific locations
        @return:
        """

        def _as_stops(sp_def, duration):
            if getattr(sp_def, "__iter__", False):
                return np.asarray(sorted(list(set(sp_def) | {duration})))
            elif type(sp_def) is int:
                eq_n = [duration // sp_def] * sp_def
                eq_n[-1] += duration % sp_def
                return np.cumsum(eq_n)
            elif type(sp_def) is float and sp_def < 1.:
                n = int(np.floor(1 / sp_def))
                eq_n = [duration // n] * n
                eq_n[-1] += duration % n
                return np.cumsum(eq_n)
            else:
                raise ValueError("split_definition '{}' not recognized".format(sp_def))

        splits_def = sorted(list(splits_def), key=lambda s: s[0])
        i_offset = 0
        for s_def in splits_def:
            index = i_offset + s_def[0]
            # build the new score to be inserted
            to_split = self.loc[index, :]
            dur = to_split.stop - to_split.start
            stops = _as_stops(s_def[1], dur)
            new = Score.from_stop(stops + to_split.start)
            new["stop"] = stops + to_split["start"]
            new["start"] = np.r_[0, stops[:-1]] + to_split["start"]
            if "duration" in self.columns:
                new = new.add_duration()
            # concat everything and reset indices
            prior = pd.concat((self.iloc[:index], new))
            self = pd.concat((prior, self.iloc[index + 1:]),
                             axis=0,
                             ignore_index=True)
            i_offset += len(stops) - 1
        return self

    def shift_events(self, event_i, n, adjust_others=True):
        self.shift_x(event_i, n, False)
        self.x_shift(event_i, n, False)
        if adjust_others:
            diffs = np.r_[self.start.values[1:] - self.stop.values[:-1], 0]
            if n > 0:
                self.loc[diffs > 0, "stop"] = self.loc[diffs > 0, "stop"].values + diffs[diffs > 0]
                diffs = np.roll(diffs, 1, axis=0)
                self.loc[diffs < 0, "start"] = self.loc[diffs < 0, "start"].values - diffs[diffs < 0]
            else:
                self.loc[diffs < 0, "stop"] = self.loc[diffs < 0, "stop"].values + diffs[diffs < 0]
                diffs = np.roll(diffs, 1, axis=0)
                self.loc[diffs > 0, "start"] = self.loc[diffs > 0, "start"].values - diffs[diffs > 0]
        if "duration" in self.columns:
            self.update_duration()
            self.clean_up()
        return self

    def x_shift(self, event_i, n, adjust_x=True):
        event_i = np.asarray(event_i) if type(event_i) is not slice \
            else np.arange(len(self))[event_i]
        starts = self.loc[event_i, "start"].values
        self.loc[event_i, "start"] = np.maximum(n + starts, self.first_start)
        if adjust_x:
            self.shift_x(np.maximum(event_i - 1, 0), n, x_adjust=False)
        if "duration" in self.columns:
            self.update_duration()
            self.clean_up()
        return self

    def shift_x(self, event_i, n, x_adjust=True):
        event_i = np.asarray(event_i) if type(event_i) is not slice \
            else np.arange(len(self))[event_i]
        stops = self.loc[event_i, "stop"].values
        self.loc[event_i, "stop"] = np.minimum(n + stops, self.last_stop)
        if x_adjust:
            self.x_shift(np.minimum(event_i + 1, len(self) - 1), n, adjust_x=False)
        if "duration" in self.columns:
            self.update_duration()
            self.clean_up()
        return self

    def overlap(self, event_i, n):
        """
        combines shift_x(event_i, n) and x_shift(event_i, -n)
        @param event_i:
        @param n:
        @return:
        """
        self.shift_x(event_i, n, x_adjust=False)
        self.x_shift(event_i, -n, adjust_x=False)
        return self

    def overlap_x(self, event_i, n):
        return self.shift_x(event_i, n, x_adjust=False)

    def x_overlap(self, event_i, n):
        return self.x_shift(event_i, -n, adjust_x=False)

    def extend(self, other):
        Score._validate(other)
        other.loc[slice(0, None), ("start", "stop")] = self.last_stop + other.loc[
            slice(0, None), ("start", "stop")].values
        return pd.concat([self, other], ignore_index=True, axis=0)

    # CONSTRUCTORS

    @staticmethod
    def from_start_stop(starts, stops, durations):
        return Score((starts, stops, durations), index=["start", "stop", "duration"]).T

    @staticmethod
    def from_stop(stop):
        """
        integers in `stops` correspond to the prev[stop] and next[start] values.
        `stops` must contain the last index ! and it can begin with 0, but doesn't have to...
        """
        stop = np.asarray(stop)
        if stop[0] == 0:
            starts = np.r_[0, stop[:-1]]
            stop = stop[1:]
        else:
            starts = np.r_[0, stop[:-1]]
        durations = stop - starts
        return Score.from_start_stop(starts, stop, durations)

    @staticmethod
    def from_duration(duration):
        stops = np.cumsum(duration)
        starts = np.r_[0, stops[:-1]]
        return Score.from_start_stop(starts, stops, duration)

    @staticmethod
    def from_data(sequence):
        if issubclass(type(sequence[0]), np.ndarray):
            axis = 1
        elif issubclass(type(sequence[0]), torch.Tensor):
            axis = 0
        else:
            raise TypeError(
                "sequence's elements' type not recognized. Must be a subclass of np.ndarray or torch.Tensor")
        duration = np.array([x.shape[axis] for x in sequence])
        return Score.from_duration(duration)

    @staticmethod
    def from_frame_definition(total_duration, frame_length, stride=1, butlasts=0):
        starts = np.arange(total_duration - frame_length - butlasts + 1, step=stride)
        durations = frame_length + np.zeros_like(starts, dtype=np.int)
        stops = starts + durations
        return Score.from_start_stop(starts, stops, durations)

    @staticmethod
    def from_recurrence_matrix(X, **kwargs):
        stops = from_recurrence_matrix(X, **kwargs)
        return Score.from_stop(stops)


@pd.api.extensions.register_dataframe_accessor("soft_q")
class SoftQueryAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def or_(self, **kwargs):
        series = False
        for col_name, func in kwargs.items():
            series = series | func(self._df[col_name])
        return series

    def and_(self, **kwargs):
        series = True
        for col_name, func in kwargs.items():
            series = series & func(self._df[col_name])
        return series


@pd.api.extensions.register_dataframe_accessor("hard_q")
class HardQueryAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def or_(self, **kwargs):
        series = self._df.soft_q.or_(**kwargs)
        return self._df[series].reset_index(drop=True)

    def and_(self, **kwargs):
        series = self._df.soft_q.and_(**kwargs)
        return self._df[series].reset_index(drop=True)
