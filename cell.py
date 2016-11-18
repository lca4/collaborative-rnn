from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CollaborativeGRUCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, num_users, num_items):
        """Note: users are numbered 1 to N, items are numbered 1 to M. User and
        item "zero" is reserved for padding purposes.
        """
        self._num_units = num_units
        self._num_users = num_users
        self._num_items = num_items

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        # shape(inputs) = [batch_size, input_size]
        # shape(state) = [batch_size, num_units]
        with tf.variable_scope(scope or type(self).__name__):  # "CollaborativeGRUCell"
            with tf.variable_scope("Gates"):
                with tf.device("/cpu:0"):
                    users = tf.get_variable("users",
                            [self._num_users + 1, self._num_units, 2 * self._num_units],
                            dtype=tf.float32)
                    # shape(w_hidden_u) = [batch_size, num_units, 2 * num_units]
                    w_hidden_u = tf.nn.embedding_lookup(users, inputs[:,0])
                    items = tf.get_variable("items",
                            [self._num_items + 1, 2 * self._num_units],
                            dtype=tf.float32)
                    # shape(w_input_i) = [batch_size, 2 * num_units]
                    w_input_i = tf.nn.embedding_lookup(items, inputs[:,1])
                res = tf.batch_matmul(tf.expand_dims(state, 1), w_hidden_u)
                res = tf.sigmoid(tf.squeeze(res, [1]) + w_input_i)
                r, z = tf.split(1, 2, res)
            with tf.variable_scope("Candidate"):
                with tf.device("/cpu:0"):
                    users = tf.get_variable("users",
                            [self._num_users + 1, self._num_units, self._num_units],
                            dtype=tf.float32)
                    # shape(w_hidden_u) = [batch_size, num_units, num_units]
                    w_hidden_u = tf.nn.embedding_lookup(users, inputs[:,0])
                    items = tf.get_variable("items",
                            [self._num_items + 1, self._num_units],
                            dtype=tf.float32)
                    # shape(w_input_i) = [batch_size, num_units]
                    w_input_i = tf.nn.embedding_lookup(items, inputs[:,1])
                res = tf.batch_matmul(tf.expand_dims(r * state, 1), w_hidden_u)
                c = tf.sigmoid(tf.squeeze(res, [1]) + w_input_i)
            new_h = z * state + (1 - z) * c
        return new_h, new_h
