#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import reader
import tensorflow as tf
import time

from math import sqrt


class CollabRNN(object):

    def __init__(self, nb_users, nb_items, is_training,
            subseq_len=128, batch_size=1, hidden_size=128, okp=0.5, ikp=0.5,
            learning_rate=0.1, rho=0.9):
        self._subseq_len = subseq_len
        self._batch_size = batch_size
        
        # placeholders for input data
        self._user_id = tf.placeholder(tf.int32, name="user_id", shape=[])
        self._input_data = tf.placeholder(tf.int32, name="input_data",
                                          shape=[batch_size, None])
        self._targets = tf.placeholder(tf.int32, name="targets",
                                       shape=[batch_size, None])
        self._sequence_length = tf.placeholder(tf.int32, name="sequence_length",
                                       shape=[batch_size])

        # RNN cell.
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        if is_training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=okp)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # Users embedding.
        with tf.device("/cpu:0"):
            users = tf.get_variable("users",
                    [nb_users, hidden_size, 3 * hidden_size], dtype=tf.float32)
            user = tf.nn.embedding_lookup(users, self._user_id)

        # Items embedding.
        with tf.device("/cpu:0"):
            items = tf.get_variable("items",
                    [nb_items, hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(items, self._input_data)

        # RNN (with dropout if training).
        if is_training:
            inputs = tf.nn.dropout(inputs, ikp)
        #inputs = [tf.squeeze(input_, [1]) 
        #          for input_ in tf.split(1, subseq_len, inputs)]
        outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                initial_state=self._initial_state)
        self._final_state = state

        # Ops to assign user matrices to the RNN and back.
        with tf.variable_scope("RNN/GRUCell/Gates/Linear") as scope:
            scope.reuse_variables()
            var = tf.get_variable("Matrix")
            self._assign_to_cell1 = (var[hidden_size:]
                    .assign(user[:,:2*hidden_size]))
            self._assign_to_user1 = (users[self._user_id,:,:2*hidden_size]
                    .assign(var[hidden_size:]))
        with tf.variable_scope("RNN/GRUCell/Candidate/Linear") as scope:
            scope.reuse_variables()
            var = tf.get_variable("Matrix")
            self._assign_to_cell2 = (var[hidden_size:]
                    .assign(user[:,2*hidden_size:]))
            self._assign_to_user2 = (users[self._user_id,:,2*hidden_size:]
                    .assign(var[hidden_size:]))

        # Output layer.
        # `output` has shape (batch_size * subseq_len, hidden_size).
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        with tf.variable_scope("output"):
            ws = tf.get_variable("weights", [hidden_size, nb_items],
                                 dtype=tf.float32)
            bs = tf.get_variable("bias", [nb_items], dtype=tf.float32)
        # `logits` has shape (batch_size * subseq_len, nb_items).
        logits = tf.matmul(output, ws) + bs

        # Loss function.
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * tf.shape(self._input_data)[1]], dtype=tf.float32)])
        self._cost = tf.reduce_sum(loss) / batch_size

        if not is_training:
            return

        # Optimization procedure.
        optimizer = tf.train.RMSPropOptimizer(
                learning_rate, decay=rho, epsilon=1e-6)
        self._train_op = optimizer.minimize(self._cost)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def assign_to_cell1(self):
        return self._assign_to_cell1

    @property
    def assign_to_cell2(self):
        return self._assign_to_cell2

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def subseq_len(self):
        return self._subseq_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def assign_to_user1(self):
        return self._assign_to_user1

    @property
    def assign_to_user2(self):
        return self._assign_to_user2

    @property
    def train_op(self):
        return self._train_op

    @property
    def user_id(self):
        return self._user_id

    @property
    def sequence_length(self):
        return self._sequence_length


def run_epoch(session, model, iterator, user_id, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    state = session.run(model.initial_state)
    session.run([model.assign_to_cell1, model.assign_to_cell2],
                feed_dict={model.user_id: user_id})
    for step, (x, y) in enumerate(iterator):
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        # Update initial state for next subsequence.
        h = model.initial_state
        feed_dict[h] = state

        cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.subseq_len

        if verbose:
            print("{} log-loss: {:.3f} speed: {:.0f} wps".format(
                  step, costs / iters,
                  iters * model.batch_size / (time.time() - start_time)))

    # Save the matrices back to the user embedding.
    session.run([model.assign_to_user1, model.assign_to_user2],
                feed_dict={model.user_id: user_id})

    return costs / iters


def main(args):
    train_data = reader.Dataset.from_path(args.train_path)
    valid_data = reader.Dataset.from_path(args.valid_path)

    nb_users = train_data.nb_users
    nb_items = train_data.nb_items

    tsettings = {
        "subseq_len": args.subseq_len,
        "batch_size": 1,
        "hidden_size": args.hidden_size,
        "okp": 1.0,
        "ikp": 1.0,
        "learning_rate": 0.01,
        "rho": 0.99,
    }

    vsettings = {
        "subseq_len": 1,
        "batch_size": 1,
        "hidden_size": args.hidden_size,
    }

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_normal_initializer(
                mean=0, stddev=1/sqrt(args.hidden_size))
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = CollabRNN(nb_users, nb_items, is_training=True,
                                    **tsettings)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = CollabRNN(nb_users, nb_items, is_training=False,
                                    **vsettings)
        tf.initialize_all_variables().run()
        for i in range(args.nb_epochs):
            train_err = 0
            valid_err = 0
            for u in np.random.permutation(nb_users):
                train_iter = train_data.iterate(u, args.subseq_len)
                train_err += run_epoch(session, train_model, train_iter, u,
                                       train_model.train_op, verbose=True)
                valid_iter = valid_data.iterate(u, 1)
                valid_err += run_epoch(session, valid_model, valid_iter, u,
                                       tf.no_op())
            train_err /= nb_users
            valid_err /= nb_users
            print("Epoch {}, train log-loss: {:.3f}".format(i + 1, train_err))
            print("Epoch {}, valid log-loss: {:.3f}".format(i + 1, valid_err))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="path to training data")
    parser.add_argument("valid_path", help="path to validation data")
    parser.add_argument("--logdir", default=".log",
                        help="directory for logs and checkpoints")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="number of hidden units in the RNN")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="RMSprop learning rate")
    parser.add_argument("--rho", type=float, default=0.9,
                        help="RMSprop decay coefficient")
    parser.add_argument("--subseq-len", type=int, default=128,
                        help="number of unrolled steps in BPTT")
    parser.add_argument("--nb-epochs", type=int, default=10,
                        help="number of epochs to run")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
