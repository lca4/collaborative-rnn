#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np


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
        nb_batches = len(seq) // batch_size
        truncated = seq[:nb_batches * batch_size]
        data = truncated.reshape([batch_size, nb_batches])
        for i in range(0, data.shape[1] - subseq_len, subseq_len):
            inputs = data[:, i:(i+subseq_len)]
            targets = data[:, (i+1):(i+subseq_len+1)]
            yield (inputs, targets)

    @classmethod
    def from_path(cls, path):
        data = collections.defaultdict(list)
        nb_users = 0
        nb_items = 0
        with open(path) as f:
            for line in f:
                u, i, t = map(int, line.strip().split())
                nb_users = max(u + 1, nb_users)
                nb_items = max(i + 1, nb_items)
                data[u].append((t, i))
        sequence = dict()
        for user, pairs in data.items():
            sequence[user] = tuple(i for t, i in sorted(pairs))
        return cls(nb_users, nb_items, sequence)
