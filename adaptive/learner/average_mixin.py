# -*- coding: utf-8 -*-

from collections import Mapping
from math import sqrt
import sys

import numpy as np
import scipy.stats

from .learner1D import Learner1D


inf = sys.float_info.max


class AverageMixin:
    @property
    def data(self):
        return {k: v.mean for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: v.standard_error if v.n >= self.min_values_per_point else inf
            for k, v in self._data.items()}

    def mean_values_per_point(self):
        return np.mean([x.n for x in self._data.values()])

    def _next_seed(self, point):
        _data = self._data.get(point, {})
        pending_seeds = self.pending_points.get(point, set())
        seed = len(_data) + len(pending_seeds)
        if seed in _data or seed in pending_seeds:
            # Means that the seed already exists, for example
            # when '_data[point].keys() | pending_points[point] == {0, 2}'.
            # Only happens when starting the learner after cancelling/loading.
            return (set(range(seed)) - pending_seeds - _data.keys()).pop()
        return seed

    def loss_per_existing_point(self):
        scale = self.value_scale()
        points = []
        loss_improvements = []
        for p, sem in self.data_sem.items():
            points.append((p, self._next_seed(p)))
            N = self.n_values(p)
            sem_improvement = (1 - sqrt(N - 1) / sqrt(N)) * sem
            loss_improvement = self.weight * sem_improvement / scale
            loss_improvements.append(loss_improvement)
        return points, loss_improvements

    def _add_to_pending(self, point):
        x, seed = self.unpack_point(point)
        if x not in self.pending_points:
            self.pending_points[x] = set()
        self.pending_points[x].add(seed)

    def _remove_from_to_pending(self, point):
        x, seed = self.unpack_point(point)
        if x in self.pending_points:
            self.pending_points[x].discard(seed)
            if not self.pending_points[x]:
                # pending_points[x] is now empty so delete the set()
                del self.pending_points[x]

    def _add_to_data(self, point, value):
        x, seed = self.unpack_point(point)
        if x not in self._data:
            self._data[x] = DataPoint()
        self._data[x][seed] = value

    def ask(self, n, tell_pending=True):
        """Return n points that are expected to maximally reduce the loss."""
        points, loss_improvements = self._ask_points_without_adding(n)
        loss_improvements = self._normalize_new_points_loss_improvements(
            points, loss_improvements)

        p, l = self.loss_per_existing_point()
        l = self._normalize_existing_points_loss_improvements(p, l)
        points += p
        loss_improvements += l

        loss_improvements, points = zip(*sorted(
            zip(loss_improvements, points), reverse=True))

        points = list(points)[:n]
        loss_improvements = list(loss_improvements)[:n]

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def n_values(self, point):
        pending_points = self.pending_points.get(point, [])
        return len(self._data[point]) + len(pending_points)

    def _mean_values_per_neighbor(self, neighbors):
        """The average number of neighbors of a 'point'."""
        return {p: sum(self.n_values(n) for n in ns) / len(ns)
            for p, ns in neighbors.items()}

    def _normalize_new_points_loss_improvements(self, points, loss_improvements):
        """If we are suggesting a new (not yet suggested) point, then its
        'loss_improvement' should be divided by the average number of values
        of its neigbors.

        This is because it will take a similar amount of points to reach
        that loss. """
        if len(self._data) < 4:
            return loss_improvements

        only_points = [p for p, s in points]
        neighbors = self._get_neighbor_mapping_new_points(only_points)
        mean_values_per_neighbor = self._mean_values_per_neighbor(neighbors)

        return [loss / mean_values_per_neighbor[p]
            for (p, seed), loss in zip(points, loss_improvements)]

    def _normalize_existing_points_loss_improvements(self, points, loss_improvements):
        """If the neighbors of 'point' have twice as much values
        on average, then that 'point' should have an infinite loss.

        We do this because this point probably has a incorrect
        estimate of the sem."""
        if len(self._data) < 4:
            return loss_improvements

        neighbors = self._get_neighbor_mapping_existing_points()
        mean_values_per_neighbor = self._mean_values_per_neighbor(neighbors)

        def needs_more_data(p):
            return mean_values_per_neighbor[p] > 1.5 * self.n_values(p)

        return [inf if needs_more_data(p) else loss
            for (p, seed), loss in zip(points, loss_improvements)]

    def _get_data(self):
        # change DataPoint -> dict for saving
        return {k: dict(v) for k, v in self._data.items()}


def add_average_mixin(cls):
    names = ('data', 'data_sem', 'mean_values_per_point',
             '_next_seed', 'loss_per_existing_point', '_add_to_pending',
             '_remove_from_to_pending', '_add_to_data', 'ask', 'n_values',
             '_normalize_new_points_loss_improvements',
             '_normalize_existing_points_loss_improvements',
             '_mean_values_per_neighbor',
             '_get_data')

    for name in names:
        setattr(cls, name, getattr(AverageMixin, name))

    return cls


class DataPoint(dict):
    """A dict-like data structure that keeps track of the
    length, sum, and sum of squares of the values.

    It has properties to calculate the mean, sample
    standard deviation, and standard error."""
    def __init__(self, *args, **kwargs):
        self.sum = 0
        self.sum_sq = 0
        self.n = 0
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        self._remove(key)
        self.sum += val
        self.sum_sq += val**2
        self.n += 1
        super().__setitem__(key, val)

    def _remove(self, key):
        if key in self:
            val = self[key]
            self.sum -= val
            self.sum_sq -= val**2
            self.n -= 1

    @property
    def mean(self):
        return self.sum / self.n

    @property
    def std(self):
        if self.n < 2:
            return np.nan
        numerator = self.sum_sq - self.n * self.mean**2
        if numerator < 0:
            # This means that the numerator is ~ -1e-15
            return 0
        return sqrt(numerator / (self.n - 1))

    @property
    def standard_error(self):
        if self.n < 2:
            return np.inf
        return self.std / sqrt(self.n)

    def __delitem__(self, key):
        self._remove(key)
        super().__delitem__(key)

    def pop(self, *args):
        self._remove(args[0])
        return super().pop(*args)

    def update(self, other=None, **kwargs):
        iterator = other if isinstance(other, Mapping) else kwargs
        for k, v in iterator.items():
            self[k] = v

    def assert_consistent_data_structure(self):
        vals = list(self.values())
        np.testing.assert_almost_equal(np.mean(vals), self.mean)
        np.testing.assert_almost_equal(np.std(vals, ddof=1), self.std)
        np.testing.assert_almost_equal(self.standard_error, scipy.stats.sem(vals))
        assert self.n == len(vals)
