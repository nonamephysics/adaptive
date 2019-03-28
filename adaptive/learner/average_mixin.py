# -*- coding: utf-8 -*-

from collections.abc import Mapping
from math import sqrt
import sys

import numpy as np
import scipy.stats

from adaptive.learner.learner1D import Learner1D


inf = sys.float_info.max


class AverageMixin:
    """The methods from this class are used in the
    `AverageLearner1D` and `AverageLearner2D.

    This cannot be used as a mixin class because of method resolution
    order problems. Instead use the `add_average_mixin` class decorator."""

    @property
    def data(self):
        return {k: v.mean for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: v.standard_error if v.n >= self.min_values_per_point else inf
            for k, v in self._data.items()}

    def mean_values_per_point(self):
        return np.mean([x.n for x in self._data.values()])

    def _next_seed(self, point, exclude=None):
        exclude = set(exclude) if exclude is not None else set()
        done_seeds = self._data.get(point, {}).keys()
        pending_seeds = self.pending_points.get(point, set())
        seed = len(done_seeds) + len(pending_seeds) + len(exclude)
        if any(seed in x for x in [done_seeds, pending_seeds, exclude]):
            # Means that the seed already exists, for example
            # when 'done_seeds[point] | pending_seeds [point] == {0, 2}'.
            # Only happens when starting the learner after cancelling/loading.
            return (set(range(seed)) - pending_seeds - done_seeds - exclude).pop()
        return seed

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
        points, loss_improvements = [], []

        # Take from the _seed_stack.
        self._fill_seed_stack(till=n)
        for i in range(n):
            exclude_seeds = set()
            point, nseeds, loss_improvement = self._seed_stack[i]
            for j in range(nseeds):
                seed = self._next_seed(point, exclude_seeds)
                exclude_seeds.add(seed)
                points.append((point, seed))
                loss_improvements.append(loss_improvement / nseeds)
                if len(points) == n:
                    break
            if len(points) == n:
                break

        # Remove the chosen points from the _seed_stack.
        if tell_pending:
            for p in points:
                self.tell_pending(p)
            nseeds_left = nseeds - j - 1  # of self._seed_stack[i]
            if nseeds_left > 0:  # not all seeds have been asked
                point, nseeds, loss_improvement = self._seed_stack[i]
                self._seed_stack[i] = (
                    point, nseeds_left,
                    loss_improvement * nseeds_left / nseeds
                )
                self._seed_stack = self._seed_stack[i:]
            else:
                self._seed_stack = self._seed_stack[i+1:]

        return points, loss_improvements

    def _fill_seed_stack(self, till):
        n = till - sum(nseeds for (_, nseeds, _) in self._seed_stack)
        if n < 1:
            return

        new_points, new_points_losses = self._new_points(n)
        existing_points, existing_points_losses = self._existing_points()

        points = new_points + existing_points
        loss_improvements = new_points_losses + existing_points_losses

        loss_improvements, points = zip(*sorted(
            zip(loss_improvements, points), reverse=True))

        # A mapping to check if a point already exists in the _seed_stack
        mapping = {point: i for i, (point, *_) in enumerate(self._seed_stack)}

        # Add points to the _seed_stack, it can happen that its
        # length exceeds the number of requested points.
        n_left = n
        for loss_improvement, (point, nseeds) in zip(
            loss_improvements, points):
            tup = (point, nseeds, loss_improvement)
            if point in mapping:  # point is inside _seed_stack
                # Combine the tuple with the same points existing in the
                # _seed_stack with the newly suggested
                # (nseeds, loss_improvements) pair.
                i = mapping[point]
                tup_old = self._seed_stack[i]
                sum_ = [sum(x) for x in zip(tup[1:], tup_old[1:])]
                self._seed_stack[i] = (point, *sum_)
            else:
                self._seed_stack.append(tup)
            n_left -= nseeds
            if n_left <= 0:
                break

    def n_values(self, point):
        """The total number of seeds (done or requested) at a point."""
        pending_points = self.pending_points.get(point, [])
        return len(self._data[point]) + len(pending_points)

    def _mean_values_per_neighbor(self, neighbors):
        """The average number of neighbors of a 'point'."""
        return {p: sum(self.n_values(n) for n in ns) / len(ns)
            for p, ns in neighbors.items()}

    def _new_points(self, n):
        """Add new points with at least self.min_values_per_point points
        or with as many points as the neighbors have on average."""
        points, loss_improvements = self._ask_points_without_adding(n)
        if len(self._data) < 4:
            points = [(p, self.min_values_per_point) for p, s in points]
            return points, loss_improvements

        only_points = [p for p, s in points]  # points are [(x, seed), ...]
        neighbors = self._get_neighbor_mapping_new_points(only_points)
        mean_values_per_neighbor = self._mean_values_per_neighbor(neighbors)

        points = []
        for p in only_points:
            n_neighbors = int(mean_values_per_neighbor[p])
            nseeds = max(n_neighbors, self.min_values_per_point)
            points.append((p, nseeds))

        return points, loss_improvements

    def _existing_points(self, fraction=0.1):
        """Increase the number of seeds by 10%."""
        if len(self.data) < 4:
            return [], []
        scale = self.value_scale()
        points = []
        loss_improvements = []

        neighbors = self._get_neighbor_mapping_existing_points()
        mean_values_per_neighbor = self._mean_values_per_neighbor(neighbors)

        for p, sem in self.data_sem.items():
            N = self.n_values(p)
            n_more = int(fraction * N)  # increase the amount of points by 10%
            n_more = max(n_more, 1)  # at least 1 point
            points.append((p, n_more))
            needs_more_data = mean_values_per_neighbor[p] > 1.5 * N
            if needs_more_data:
                loss_improvement = inf
            else:
                # This is the improvement considering we will add
                # n_more seeds to the stack.
                sem_improvement = (1 - sqrt(N) / sqrt(N + n_more)) * sem
                # We scale the values, sem(ys) / scale == sem(ys / scale).
                # and multiply them by a weight average_priority.
                loss_improvement = self.average_priority * sem_improvement / scale
            loss_improvements.append(loss_improvement)
        return points, loss_improvements

    def _get_data(self):
        # change DataPoint -> dict for saving
        return {k: dict(v) for k, v in self._data.items()}


def add_average_mixin(cls):
    for name in AverageMixin.__dict__.keys():
        if not name.startswith('__') and not name.endswith('__'):
            # We assume that we don't implement or overwrite
            # dunder / magic methods inside AverageMixin.
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
            # The sample standard deviation is not defined for
            # less than 2 values.
            return np.nan
        numerator = self.sum_sq - self.n * self.mean**2
        if numerator < 0:
            # This means that the numerator is ~ -1e-15
            # so nummerically it's 0.
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