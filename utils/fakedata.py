#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np


def softmax(xs):
    zs = np.exp(xs - np.max(xs))
    return zs / zs.sum(axis=0)


def rnn_sequence(ws_in, ws_out, ws_h, initial_state):
    state = initial_state
    nb_items = ws_in.shape[0]
    while True:
        probs = softmax(np.dot(state, ws_out))
        event = np.random.choice(nb_items, p=probs)
        yield event
        state = np.tanh(np.dot(state, ws_h) + ws_in[event])


def main(args):
    ws_in = np.random.randn(args.nb_items, args.hidden_size)
    ws_out = np.random.randn(args.hidden_size, args.nb_items)
    train_path = "{}-train.txt".format(args.prefix)
    valid_path = "{}-valid.txt".format(args.prefix)
    with open(train_path, "w") as tf, open(valid_path, "w") as vf:
        for u in range(args.nb_users):
            ws_h = np.random.randn(args.hidden_size, args.hidden_size)
            initial_state = np.zeros(args.hidden_size)
            seq = rnn_sequence(ws_in, ws_out, ws_h, initial_state)
            for t in range(args.train_seq_length):
                tf.write("{} {} {}\n".format(u, next(seq), t))
            offset = args.train_seq_length
            for t in range(args.valid_seq_length):
                vf.write("{} {} {}\n".format(u, next(seq), offset + t))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default="fakedata")
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--nb-users', type=int, default=5)
    parser.add_argument('--nb-items', type=int, default=10)
    parser.add_argument('--train-seq-length', type=int, default=128)
    parser.add_argument('--valid-seq-length', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
