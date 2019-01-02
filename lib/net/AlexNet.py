from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from DeViSE_tensorflow.lib.net.net import Net


class AlexNet(Net):
    def __init__(self, common_params, net_params, test=False):
        super(AlexNet, self).__init__(common_params, net_params)
        # process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])
        self.embedding = int(net_params['embedding'])

    def inference(self, image, keep_prob, istraining):
        """Build the vgg16 model

        Args:
        images:  4-D tensor [batch_size, image_height, image_width, channels]
        Returns:
        predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        with tf.variable_scope('AlexNet') as scope:
            with tf.variable_scope('conv1') as scope:
                temp_conv = self.conv2d("conv1", image, kernel_size=[11, 11, 3, 96], stride=4,
                                        istraining=istraining, padding='VALID')

            temp_pool = self.max_pool(temp_conv, [3, 3], 2, padding='VALID')

            with tf.variable_scope('conv2') as scope:
                temp_conv = self.conv2d("conv2", temp_pool, kernel_size=[5, 5, 96, 256], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [3, 3], 2, padding='VALID')

            with tf.variable_scope('conv3') as scope:
                temp_conv = self.conv2d("conv3", temp_pool, kernel_size=[3, 3, 256, 384], stride=1,
                                        istraining=istraining)

            with tf.variable_scope('conv4') as scope:
                temp_conv = self.conv2d("conv4", temp_conv, kernel_size=[3, 3, 384, 384], stride=1,
                                        istraining=istraining)

            with tf.variable_scope('conv5') as scope:
                temp_conv = self.conv2d("conv5", temp_conv, kernel_size=[3, 3, 384, 256], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2, padding='VALID')
            # fc to conv
            fc6 = self.local('fc6', temp_pool, kernel_size=[6, 6, 256, 4096], stride=1, istraining=istraining)
            fc6_dropout = tf.nn.dropout(fc6, keep_prob=keep_prob)

            fc7 = self.local('fc7', fc6_dropout, kernel_size=[1, 1, 4096, 4096], stride=1, istraining=istraining)
            fc7_dropout = tf.nn.dropout(fc7, keep_prob=keep_prob)

            fc8 = self.local('fc8', fc7_dropout, kernel_size=[1, 1, 4096, self.num_classes], stride=1,
                             istraining=istraining)
            fc8_squeeze = tf.squeeze(fc8)
        return fc8_squeeze

    def softmax_loss(self, predicts, labels):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predicts)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection('losses', loss / self.batch_size)

        weight_loss = tf.add_n(tf.get_collection('weight_losses'), name='weight_total_loss')
        tf.add_to_collection('weight_losses', weight_loss / self.batch_size)

        total_loss = loss + weight_loss
        tf.summary.scalar('classfication_loss', loss)
        tf.summary.scalar('weight_total_loss', weight_loss)
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

    def test(self, predicts, labels):
        correct_prediction = tf.equal(tf.argmax(predicts, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)
        return accuracy
