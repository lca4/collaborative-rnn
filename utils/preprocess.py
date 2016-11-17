#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import argparse
import collections
import datetime
import itertools
import os.path
import time

from scipy.stats import entropy


BK_ENTROPY_CUTOFF = 2.5
LFM_ENTROPY_CUTOFF = 3.0

MIN_OCCURRENCES = 10
MIN_VALID_SEQ_LEN = 3
MAX_VALID_SEQ_LEN = 500


def parse_brightkite(path):
    """Parse the BrightKite dataset.

    This takes as input the file `loc-brightkite_totalCheckins.txt` available
    at the following URL: <https://snap.stanford.edu/data/loc-brightkite.html>.
    """
    # Format: [user] [check-in time] [latitude] [longitude] [location id].
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                usr, ts, lat, lon, loc = line.strip().split('\t')
            except ValueError:
                print("could not parse line {} ('{}'), ignoring".format(
                        i, line.strip()))
                continue
            dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            ts = time.mktime(dt.timetuple())
            yield (usr, loc, ts)


def parse_lastfm(path):
    """Parse the last.fm dataset.

    This takes as input the file
    `userid-timestamp-artid-artname-traid-traname.tsv` available at the
    following URL:
    <http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html>.
    """
    # Format: [user] [timestamp] [artist ID] [artist] [track ID] [track].
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                usr, ts, aid, artist, tid, track = line.strip().split('\t')
            except ValueError:
                print("could not parse line {} ('{}'), ignoring".format(
                        i, line.strip()))
                continue
            dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            ts = time.mktime(dt.timetuple())
            yield (usr, aid, ts)


def preprocess(stream, output_dir, prefix="processed", min_entropy=0.0):
    """Preprocess a stream of (user, item, timestamp) triplets.

    The preprocessing roughly includes the following steps:

    - remove items that occur infrequently,
    - remove users that consume very few items,
    - remove users who do not consume "diverse enough" items,
    - separate data into training and validation sets,
    - make sure that items in the validation sets appear at least once in the
      training set,
    - relabel items and users with consecutive integers.
    """
    # Step 1: read stream and count number of item occurrences.
    data = list()
    occurrences = collections.defaultdict(lambda: 0)
    for user, item, ts in stream:
        data.append((user, item, ts))
        occurrences[item] += 1
    # Step 2: remove items that occurred infrequently, create user seqs.
    tmp_dict = collections.defaultdict(list)
    for user, item, ts in data:
        if occurrences[item] < MIN_OCCURRENCES:
            continue
        tmp_dict[user].append((ts, item))
    # Step 3: order user sequences by timestamp.
    seq_dict = dict()
    for user, seq in tmp_dict.items():
        seq = [item for ts, item in sorted(seq)]
        seq_dict[user] = seq
    # Step 4: split into training and validation sets. Ignore users who
    # consumed few items or who do not meet entropy requirements.
    train = dict()
    valid = dict()
    for user, seq in seq_dict.items():
        if len(seq) <= MIN_OCCURRENCES:
            continue
        hist = collections.defaultdict(lambda: 0)
        for item in seq:
            hist[item] += 1
        if entropy(hist.values()) <= min_entropy:
            continue
        # Implementation note: round(0.025 * 100) gives 3.0 in Python, but 2.0
        # in Julia. Beware! Results might differ!
        cutoff = min(MAX_VALID_SEQ_LEN, max(MIN_VALID_SEQ_LEN,
                                            int(round(0.025 * len(seq)))))
        train[user] = seq[:-cutoff]
        valid[user] = seq[-cutoff:]
    # Step 5: relabel users and items, and remove items that do not appear in
    # the training sequences.
    items = set(itertools.chain(*train.values()))
    users = set(train.keys())
    user2id = dict(zip(users, range(1, len(users) + 1)))
    item2id = dict(zip(items, range(1, len(items) + 1)))
    train2 = dict()
    valid2 = dict()
    for user in users:
        train2[user2id[user]] = tuple(map(lambda x: item2id[x], train[user]))
        valid2[user2id[user]] = tuple(map(lambda x: item2id[x],
                filter(lambda x: x in items, valid[user])))
    # Step 6: write out the sequences.
    train_path = os.path.join(output_dir, "{}-train.txt".format(prefix))
    valid_path = os.path.join(output_dir, "{}-valid.txt".format(prefix))
    with open(train_path, "w") as tf, open(valid_path, "w") as vf:
        for uid in user2id.values():
            t = 1
            for iid in train2[uid]:
                tf.write("{} {} {}\n".format(uid, iid, t))
                t += 1
            for iid in valid2[uid]:
                vf.write("{} {} {}\n".format(uid, iid, t))
                t += 1
    print("Done.")


def main(args):
    if args.which == "brightkite":
        stream = parse_brightkite(args.path)
        cutoff = BK_ENTROPY_CUTOFF
    elif args.which == "lastfm":
        stream = parse_lastfm(args.path)
        cutoff = LFM_ENTROPY_CUTOFF
    else:
        raise RuntimeError("unknown dataset?!")
    preprocess(stream, args.output_dir,
               prefix=args.which,
               min_entropy=cutoff)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=("brightkite", "lastfm"))
    parser.add_argument("path")
    parser.add_argument("--output-dir", default="./")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
