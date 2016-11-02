#!/usr/bin/env python
from __future__ import absolute_import
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
MIN_VALID_SEQ_LEN = 3
MAX_VALID_SEQ_LEN = 500


def brightkite(path, output_dir):
    """Preprocess the BrightKite dataset.

    This takes as input the file `loc-brightkite_totalCheckins.txt` available
    at the following URL: <https://snap.stanford.edu/data/loc-brightkite.html>.
    """
    # Format: [user] [check-in time] [latitude] [longitude] [location id].
    # First step: read the data, convert the timestamps, and count the number
    # of times each item occurs.
    data = list()
    occurrences = collections.defaultdict(lambda: 0)
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
            data.append((usr, loc, ts))
            occurrences[loc] += 1
    # Second step: prune items that occurred less than 10 times, and assign ids
    # to them.
    next_uid = 0
    next_lid = 0
    usr2id = dict()
    loc2id = dict()
    tmp_dict = collections.defaultdict(list)
    for usr, loc, ts in data:
        if occurrences[loc] < 10:
            continue
        if usr not in usr2id:
            usr2id[usr] = next_uid
            next_uid += 1
        if loc not in loc2id:
            loc2id[loc] = next_lid
            next_lid += 1
        tmp_dict[usr2id[usr]].append((ts, loc2id[loc]))
    # Third step: order user sequences.
    seq_dict = dict()
    for uid, seq in tmp_dict.items():
        seq = [loc for ts, loc in sorted(seq)]
        seq_dict[uid] = seq
    # Fourth step: split into training and validation step. Ignore users who
    # consumed less than 10 items or who do not meet entropy requirements.
    train = dict()
    valid = dict()
    for uid, seq in seq_dict.items():
        if len(seq) <= 10:
            continue
        hist = collections.defaultdict(lambda: 0)
        for lid in seq:
            hist[lid] += 1
        if entropy(hist.values()) <= BK_ENTROPY_CUTOFF:
            continue
        # Implementation note: round(0.025 * 100) gives 3.0 in Python, but 2.0
        # in Julia. Beware! Results might differ!
        cutoff = max(MIN_VALID_SEQ_LEN, int(round(0.025 * len(seq))))
        train[uid] = seq[:-cutoff]
        valid[uid] = seq[-cutoff:]
    # Finally, prune items in the validation sequences that are not found in
    # the training sequences, and relabel everything.
    items = set(itertools.chain(*train.values()))
    users = set(train.keys())
    new_uid = dict(zip(users, range(len(users))))
    new_lid = dict(zip(items, range(len(items))))
    train2 = dict()
    valid2 = dict()
    for u in users:
        train2[new_uid[u]] = tuple(map(lambda x: new_lid[x], train[u]))
        valid2[new_uid[u]] = tuple(map(lambda x: new_lid[x],
                filter(lambda x: x in items, valid[u])))
    # Write out the sequences.
    train_path = os.path.join(output_dir, "brightkite-train.txt")
    valid_path = os.path.join(output_dir, "brightkite-valid.txt")
    with open(train_path, "w") as tf, open(valid_path, "w") as vf:
        for uid in train2.keys():
            t = 0
            for lid in train2[uid]:
                tf.write("{} {} {}\n".format(uid, lid, t))
                t += 1
            for lid in valid2[uid]:
                vf.write("{} {} {}\n".format(uid, lid, t))
                t += 1
    print("Done.")


def lastfm(path):
    """Preprocess the last.fm dataset.

    This takes as input the file
    `userid-timestamp-artid-artname-traid-traname.tsv` available at the
    following URL:
    <http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html>.
    """
    # Format: [user] [timestamp] [artist ID] [artist] [track ID] [track].
    pass


def main(args):
    if args.which == "brightkite":
        brightkite(args.path, args.output_dir)
    elif args.which == "lastfm":
        lastfm(args.path, args.output_dir)
    else:
        raise RuntimeError("unknown dataset?!")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=("brightkite", "lastfm"))
    parser.add_argument("path")
    parser.add_argument("--output-dir", default="./")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
