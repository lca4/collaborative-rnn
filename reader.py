#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from math import ceil


class Dataset(object):

    def __init__(self, nb_users, nb_items, seq_dict):
        self._seq_dict = seq_dict
        self._nb_users = nb_users
        self._nb_items = nb_items

    @property
    def nb_users(self):
        return self._nb_users

    @property
    def nb_items(self):
        return self._nb_items

    def iterate(self, uid, batch_size, subseq_len):
        seq = np.asarray(self._seq_dict[uid])
        # The final array will have:
        # - nb rows: batch_size
        # - nb cols: k * subseq_len + 1
        n = len(seq)
        k = max(1, int(ceil((n - batch_size) / (batch_size * subseq_len))))
        nb_cols = k * subseq_len + 1
        data = np.zeros(batch_size * nb_cols, dtype=int)
        data[:n] = seq
        data = data.reshape((batch_size, nb_cols))
        # The sequence_length array will be of shape (batch_size, k)
        valid = np.zeros(batch_size * k, dtype=int)
        s = int((n - (n / nb_cols)) / subseq_len)
        valid[:s] = subseq_len
        rem = ((n - (n / nb_cols))) % subseq_len
        if rem > 0:
            valid[s] = rem
        valid = valid.reshape((batch_size, k))
        offset = 0
        for i in range(k):
            inputs = data[:, offset:(offset+subseq_len)]
            targets = data[:, (offset+1):(offset+subseq_len+1)]
            yield (inputs, targets, valid[:,i])
            offset += subseq_len

    @classmethod
    def from_path(cls, path):
        data = collections.defaultdict(list)
        nb_users = 0
        nb_items = 0
        with open(path) as f:
            for line in f:
                u, i, t = map(int, line.strip().split())
                nb_users = max(u + 1, nb_users)  # Users are numbered 0 -> N-1.
                nb_items = max(i + 1, nb_items)  # Items are numbered 0 -> M-1.
                data[u].append((t, i))
        sequence = dict()
        for user, pairs in data.items():
            sequence[user] = tuple(i for t, i in sorted(pairs))
        return cls(nb_users, nb_items, sequence)
