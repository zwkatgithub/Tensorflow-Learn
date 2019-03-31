import tensorflow as tf
from flags import FLAGS


class UserModel:
    # just define your net
    def __init__(self):
        pass

    def __call__(self, inputs_dict):
        return self._net(inputs_dict)
    def _net(self, inputs_dict):
        inputs = inputs_dict['inputs']
        out = tf.contrib.layers.fully_connected(inputs, num_outputs = 20, scope='fc1')
        out = tf.contrib.layers.fully_connected(out, num_outputs = 20, scope='fc2')
        logits = tf.contrib.layers.fully_connected(out, num_outputs = FLAGS.num_classes, scope='fc3')

        return logits


class UserModel2:
    # just define your net
    def __init__(self):
        pass

    def __call__(self, inputs_dict):
        return self._net(inputs_dict)
    def _net(self, inputs_dict):
        inputs = inputs_dict['inputs']
        out = tf.contrib.layers.fully_connected(inputs, num_outputs = 50, scope='fc1')
        out = tf.contrib.layers.fully_connected(out, num_outputs = 20, scope='fc2')
        logits = tf.contrib.layers.fully_connected(out, num_outputs = FLAGS.num_classes, scope='fc3')

        return logits


