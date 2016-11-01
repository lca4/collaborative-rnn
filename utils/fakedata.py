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
    for u in range(args.nb_users):
        ws_h = np.random.randn(args.hidden_size, args.hidden_size)
        initial_state = np.zeros(args.hidden_size)
        seq = rnn_sequence(ws_in, ws_out, ws_h, initial_state)
        for t in range(args.seq_length):
            print("{} {} {}".format(t, u, next(seq)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--nb-users', type=int, default=5)
    parser.add_argument('--nb-items', type=int, default=10)
    parser.add_argument('--seq-length', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
