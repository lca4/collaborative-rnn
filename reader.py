from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from math import ceil


class Dataset(object):

    def __init__(self, num_users, num_items, seq_dict):
        self._seq_dict = seq_dict
        self._num_users = num_users
        self._num_items = num_items
        # These variables are set after calling `prepare_batches`.
        self._users_in_batches = None
        self._batches = None
        self._seq_lengths = None
        self._chunk_size = None

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def num_triplets(self):
        return sum(len(seq) for u, seq in self)

    @property
    def num_batches(self):
        if self._batches is None:
            raise RuntimeError("`prepare_batches` has not been called yet.")
        return len(self._batches)

    @property
    def users_in_batches(self):
        if self._users_in_batches is None:
            raise RuntimeError("`prepare_batches` has not been called yet.")
        return self._users_in_batches

    def __getitem__(self, u):
        return self._seq_dict[u]

    def __iter__(self):
        return self._seq_dict.iteritems()

    def truncate_seqs(self, max_size, keep_first=False):
        for user in self._seq_dict.keys():
            if keep_first:
                self._seq_dict[user] = self._seq_dict[user][:max_size]
            else:
                self._seq_dict[user] = self._seq_dict[user][-max_size:]

    def iter_batches(self, order=None):
        if order is None:
            order = range(self.num_batches)
        if self._batches is None:
            raise RuntimeError("`prepare_batches` has not been called yet.")
        cs = self._chunk_size
        def iter_batch(batch, seq_length):
            num_cols = batch.shape[1]
            for i, z in enumerate(range(0, num_cols - 1, cs)):
                inputs = batch[:,z:z+cs,:]
                targets = batch[:,(z+1):(z+cs+1),1]
                yield (inputs, targets, seq_length[:,i])
        for i in order:
            yield iter_batch(self._batches[i], self._seq_lengths[i])

    def prepare_batches(self, chunk_size, batch_size, batches_like=None):
        # Spread users over batches.
        if batches_like is not None:
            self._users_in_batches = batches_like.users_in_batches
        else:
            self._users_in_batches = Dataset._assign_users_to_batches(
                    batch_size, self._seq_dict)
        # Build the batches and record the corresponding valid sequence lengths.
        self._chunk_size = chunk_size
        self._batches = list()
        self._seq_lengths = list()
        for users in self._users_in_batches:
            lengths = tuple(len(self[u]) for u in users)
            num_chunks = int(ceil(max(max(lengths) - 1, chunk_size)
                    / chunk_size))
            num_cols = num_chunks * chunk_size + 1
            batch = np.zeros((batch_size, num_cols, 2), dtype=np.int32)
            seq_length = np.zeros((batch_size, num_chunks), dtype=np.int32)
            for i, (user, length) in enumerate(zip(users, lengths)):
                # Assign the values to the batch.
                batch[i,:length,0] = user
                batch[i,:length,1] = self[user]
                # Compute and assign the valid sequence lengths.
                q, r = divmod(max(0, min(num_cols, length) - 1), chunk_size)
                seq_length[i,:q] = chunk_size
                if r > 0:
                    seq_length[i,q] = r
            self._batches.append(batch)
            self._seq_lengths.append(seq_length)

    @staticmethod
    def _assign_users_to_batches(batch_size, seq_dict):
        lengths, users = zip(*sorted(((len(seq), u)
                for u, seq in seq_dict.iteritems()), reverse=True))
        return tuple(users[i:i+batch_size]
                for i in range(0, len(users), batch_size))

    @classmethod
    def from_path(cls, path):
        data = collections.defaultdict(list)
        num_users = 0
        num_items = 0
        with open(path) as f:
            for line in f:
                u, i, t = map(int, line.strip().split())
                num_users = max(u, num_users)  # Users are numbered 1 -> N.
                num_items = max(i, num_items)  # Items are numbered 1 -> M.
                data[u].append((t, i))
        sequence = dict()
        for user in range(1, num_users + 1):
            if user in data:
                sequence[user] = np.array([i for t, i in sorted(data[user])])
            else:
                sequence[user] = np.array([])
        return cls(num_users, num_items, sequence)
