from __future__ import division
from __future__ import print_function

import argparse
import itertools
import numpy as np
import tensorflow as tf
import time

from math import sqrt
from cell import CollaborativeGRUCell
from reader import Dataset


class CollaborativeRNN(object):

    def __init__(self, num_users, num_items, is_training,
            chunk_size=128, batch_size=1, hidden_size=128,
            learning_rate=0.1, rho=0.9):
        self._batch_size = batch_size
        
        # placeholders for input data
        self._inputs = tf.placeholder(tf.int32, name="inputs",
                shape=[batch_size, chunk_size, 2])
        self._targets = tf.placeholder(tf.int32, name="targets",
                shape=[batch_size, chunk_size])
        self._seq_length = tf.placeholder(tf.int32, name="seq_length",
                shape=[batch_size])

        # RNN cell.
        cell = CollaborativeGRUCell(hidden_size, num_users, num_items)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        inputs = [tf.squeeze(input_, [1]) for input_
                in tf.split(self._inputs, chunk_size, axis=1)]
        states, _ = tf.nn.static_rnn(cell, inputs,
                initial_state=self._initial_state)

        # Compute the final state for each element of the batch.
        self._final_state = tf.gather_nd([self._initial_state] + states,
                tf.transpose(tf.stack(
                        [self._seq_length, tf.range(batch_size)])))

        # Output layer.
        # `output` has shape (batch_size * chunk_size, hidden_size).
        output = tf.reshape(tf.concat(states, axis=1), [-1, hidden_size])
        with tf.variable_scope("output"):
            ws = tf.get_variable("weights", [hidden_size, num_items + 1],
                                 dtype=tf.float32)
        # `logits` has shape (batch_size * chunk_size, num_items).
        logits = tf.matmul(output, ws)
        targets = tf.reshape(self._targets, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=logits)

        masked = loss * tf.to_float(tf.sign(targets))
        masked = tf.reshape(masked, [batch_size, chunk_size])
        self._cost = tf.reduce_sum(masked, axis=1)

        if not is_training:
            self._train_op = tf.no_op()
            return

        scalar_cost = tf.reduce_mean(masked)

        # Optimization procedure.
        optimizer = tf.train.RMSPropOptimizer(
                learning_rate, decay=rho, epsilon=1e-8)
        self._train_op = optimizer.minimize(scalar_cost)

        self._rms_reset = list()
        for var in tf.trainable_variables():
            slot = optimizer.get_slot(var, "rms")
            op = slot.assign(tf.zeros(slot.get_shape()))
            self._rms_reset.append(op)

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def seq_length(self):
        return self._seq_length

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def rms_reset(self):
        return self._rms_reset


def run_batch(session, model, iterator, initial_state):
    """Runs the model on all chunks of one batch."""
    costs = np.zeros(model.batch_size)
    sizes = np.zeros(model.batch_size)
    state = initial_state
    for inputs, targets, seq_len in iterator:
        fetches = [model.cost, model.final_state, model.train_op]
        feed_dict = {}
        feed_dict[model.inputs] = inputs
        feed_dict[model.targets] = targets
        feed_dict[model.seq_length] = seq_len
        feed_dict[model.initial_state] = state
        cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        sizes += seq_len
    with np.errstate(invalid='ignore'):
        errors = costs / sizes
    return (errors, np.sum(sizes), state)


def run_epoch(session, train_model, valid_model, train_iter, valid_iter,
        tot_size):
    """Runs the model on the given data."""
    start_time = time.time()

    train_errors = list()
    valid_errors = list()
    tot = 0

    next_tenth = tot_size / 10

    for train, valid in itertools.izip(train_iter, valid_iter):
        state = session.run(train_model.initial_state)
        # Training data.
        errors, num_triplets, state = run_batch(
                session, train_model, train, state)
        tot += num_triplets
        train_errors.extend(errors)
        # Validation data.
        errors, num_triplets, state = run_batch(
                session, valid_model, valid, state)
        tot += num_triplets
        valid_errors.extend(errors)

        if tot > next_tenth:
            print("log-loss: {:.3f} speed: {:.0f} wps".format(
                    np.nanmean(train_errors),
                    tot / (time.time() - start_time)))
            next_tenth += tot_size / 10

    return (np.nanmean(train_errors), np.nanmean(valid_errors))


def main(args):
    # Read (and optionally, truncate) the training and validation data.
    train_data = Dataset.from_path(args.train_path)
    if args.max_train_chunks is not None:
        size = args.max_train_chunks * args.chunk_size
        train_data.truncate_seqs(size)
    valid_data = Dataset.from_path(args.valid_path)
    if args.max_valid_chunks is not None:
        size = args.max_valid_chunks * args.chunk_size
        valid_data.truncate_seqs(size, keep_first=True)

    num_users = train_data.num_users
    num_items = train_data.num_items
    tot_size = train_data.num_triplets + valid_data.num_triplets

    train_data.prepare_batches(args.chunk_size, args.batch_size)
    valid_data.prepare_batches(args.chunk_size, args.batch_size,
            batches_like=train_data)

    settings = {
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "rho": args.rho,
    }

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_normal_initializer(
                mean=0, stddev=1/sqrt(args.hidden_size))
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = CollaborativeRNN(num_users, num_items,
                    is_training=True, **settings)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = CollaborativeRNN(num_users, num_items,
                    is_training=False, **settings)
        tf.global_variables_initializer().run()
        session.run(train_model.rms_reset)
        for i in range(1, args.num_epochs + 1):
            order = np.random.permutation(train_data.num_batches)
            train_iter = train_data.iter_batches(order=order)
            valid_iter = valid_data.iter_batches(order=order)

            train_err, valid_err = run_epoch(session, train_model, valid_model,
                    train_iter, valid_iter, tot_size)
            print("Epoch {}, train log-loss: {:.3f}".format(i, train_err))
            print("Epoch {}, valid log-loss: {:.3f}".format(i, valid_err))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="path to training data")
    parser.add_argument("valid_path", help="path to validation data")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="number of sequences processed in parallel")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="number of unrolled steps in BPTT")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="number of hidden units in the RNN cell")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="RMSprop learning rate")
    parser.add_argument("--max-train-chunks", type=int, default=None,
                        help="max number of chunks per user for training")
    parser.add_argument("--max-valid-chunks", type=int, default=None,
                        help="max number of chunks per user for validation")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="number of epochs to run")
    parser.add_argument("--rho", type=float, default=0.9,
                        help="RMSprop decay coefficient")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="enable display of debugging messages")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        print("arguments:")
        for key, val in vars(args).iteritems():
            print("{: <18} {}".format(key, val))
    main(args)
