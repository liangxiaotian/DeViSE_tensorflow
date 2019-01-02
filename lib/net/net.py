from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re


class Net(object):
    """Base Net class
    """

    def __init__(self, common_params, net_params):
        """
        common_params: a params dict
        net_params: a params dict
        """
        # pretrained variable collection
        # 用于收集训练的权重值
        self.pretrained_collection = []
        # trainable variable collection
        self.trainable_collection = []

    def _variable_init(self, name, shape, initializer, pretrain=True, train=True, stop_grad=False):
        """Helper to create a Variable
        Args:
          name: name of the Variable
          shape: list of ints
          initializer: initializer of Variable

        Returns:
          Variable Tensor
        """
        # var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        with tf.device('/cpu:0'):
            if stop_grad:
                var = tf.get_variable(name, shape, initializer=initializer, trainable=False, dtype=tf.float32)
            else:
                var = tf.get_variable(name, shape, initializer=initializer, trainable=True, dtype=tf.float32)
        if pretrain:
            self.pretrained_collection.append(var)
        if train:
            self.trainable_collection.append(var)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, pretrain=True, train=True, stop_grad=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with truncated normal distribution
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard devision of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

       Returns:
          Variable Tensor
        """
        initializer = tf.contrib.layers.variance_scaling_initializer()
        var = self._variable_init(name, shape, initializer, pretrain, train, stop_grad)

        if wd is not None:
            # weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('weight_losses', weight_decay)
        return var

    def conv2d(self, scope, input, kernel_size, stride=1, pretrain=True, train=True, istraining=True, stop_grad=False,
               padding='SAME'):
        """convolutional layer

        Args:
          input: 4-D tensor [batch_size, height, width, depth]
          scope: variable_scope name
          kernel_size: [k_height, k_width, in_channel, out_channel]
          stride: int32
        Return:
          output: 4-D tensor [batch_size, height/stride, width/stride, out_channels]
        """
        with tf.variable_scope(scope) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=kernel_size,
                                                      stddev=5e-2,
                                                      wd=self.weight_decay, pretrain=pretrain, train=train,
                                                      stop_grad=stop_grad)
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding=padding)
            biases = self._variable_init('biases', kernel_size[3:], tf.constant_initializer(0.0), pretrain, train)
            bias = tf.nn.bias_add(conv, biases)
            # add batch normalization before activation function
            conv1 = tf.layers.batch_normalization(bias, training=istraining)
            conv1 = self.prelu(bias)

        return conv1

    def max_pool(self, input, kernel_size, stride, padding='SAME'):
        """max_pool layer

        Args:
          input: 4-D tensor [batch_zie, height, width, depth]
          kernel_size: [k_height, k_width]
          stride: int32
        Return:
          output: 4-D tensor [batch_size, height/stride, width/stride, depth]
        """
        return tf.nn.max_pool(input, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride, stride, 1],
                              padding=padding)

    def local(self, scope, input, kernel_size, stride, prelu=True, pretrain=True, train=True, istraining=True,
              stop_grad=False):
        """Fully connection layer

        Args:
          scope: variable_scope name
          input: [batch_size, ???]
          out_dimension: int32
        Return:
          output: 2-D tensor [batch_size, out_dimension]
        """
        with tf.variable_scope(scope) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=kernel_size,
                                                      stddev=5e-2,
                                                      wd=self.weight_decay, pretrain=pretrain, train=train,
                                                      stop_grad=stop_grad)
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='VALID')
            conv = tf.layers.batch_normalization(conv, training=istraining)
            # remove bias
            if prelu:
                conv = self.prelu(conv)
            else:
                conv = tf.identity(conv, name=scope.name)

        return conv

    def prelu(self, _x):
        """parametric ReLU activation"""
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

    def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
        """leaky relu
        Return:
          y : Tensor
        """
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x

    def inference(self, images):
        """Build the yolo model

        Args:
          images:  4-D tensor [batch_size, image_height, image_width, channels]
        Returns:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        raise NotImplementedError

    def loss(self, predicts, labels):
        """Add Loss to all the trainable variables

        Args:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
          ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
          labels  : 3-D tensor of [batch_size, max_objects, 5]
          objects_num: 1-D tensor [batch_size]
        """
        raise NotImplementedError
