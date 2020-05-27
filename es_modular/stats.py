# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32),
                        np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


class RunningStat(object):
    def __init__(self, shape, eps, obs_mean=None, obs_std=None):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps
        if obs_mean is not None:
            self.count = 1e6
            self.sum = 1e6 * obs_mean
            self.sumsq = (np.square(obs_std) + np.square(obs_mean)) * 1e6
        self._update()


    def _update(self):
        self._mean = self.sum / self.count
        self._std = np.sqrt(np.maximum(self.sumsq / self.count - np.square(self._mean), 1e-2))

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c
        self._update()

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def set_from_init(self, init_mean, init_std, init_count):
        if init_mean is not None:
            self.sum[:] = init_mean * init_count
            self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
            self.count = init_count
            self._update()