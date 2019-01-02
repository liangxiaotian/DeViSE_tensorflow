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
        self.MARGIN = float(common_params['margin'])
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
                                        istraining=istraining, stop_grad=True, padding='VALID')

            temp_pool = self.max_pool(temp_conv, [3, 3], 2, padding='VALID')

            with tf.variable_scope('conv2') as scope:
                temp_conv = self.conv2d("conv2", temp_pool, kernel_size=[5, 5, 96, 256], stride=1,
                                        istraining=istraining, stop_grad=True)

            temp_pool = self.max_pool(temp_conv, [3, 3], 2, padding='VALID')

            with tf.variable_scope('conv3') as scope:
                temp_conv = self.conv2d("conv3", temp_pool, kernel_size=[3, 3, 256, 384], stride=1,
                                        istraining=istraining, stop_grad=True)

            with tf.variable_scope('conv4') as scope:
                temp_conv = self.conv2d("conv4", temp_conv, kernel_size=[3, 3, 384, 384], stride=1,
                                        istraining=istraining, stop_grad=True)

            with tf.variable_scope('conv5') as scope:
                temp_conv = self.conv2d("conv5", temp_conv, kernel_size=[3, 3, 384, 256], stride=1,
                                        istraining=istraining, stop_grad=True)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2, padding='VALID')
            # fc to conv
            fc6 = self.local('fc6', temp_pool, kernel_size=[6, 6, 256, 4096], stride=1, pretrain=True, train=True,
                             istraining=istraining, stop_grad=True)
            fc6_dropout = tf.nn.dropout(fc6, keep_prob=keep_prob)

            fc7 = self.local('fc7', fc6_dropout, kernel_size=[1, 1, 4096, 4096], stride=1, pretrain=True,
                             train=True, istraining=istraining, stop_grad=True)
            fc7_dropout = tf.nn.dropout(fc7, keep_prob=keep_prob)

            embedding = self.local('fc8', fc7_dropout, kernel_size=[1, 1, 4096, self.embedding], stride=1,
                                   pretrain=False, train=True, istraining=True)
            embedding_squeeze = tf.squeeze(embedding)
        return embedding_squeeze

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

    def rank_loss(self, predict_embedding, labels, label_embedding):
        loss = tf.constant(0.0000)
        # get a predict true embedding
        label_embedding = tf.cast(label_embedding, dtype=tf.float32)
        true_label_embedding = tf.gather(label_embedding, labels)  # get embedding from label_embedding
        x_mul = tf.multiply(predict_embedding, true_label_embedding)
        predic_true_distance = tf.reduce_sum(x_mul, axis=1)

        # get a random negative label embedding
        negative_list = tf.random_uniform([self.batch_size], minval=0, maxval=100, dtype=tf.int32,
                                          seed=tf.set_random_seed(100), name='negative')
        negative_embedding = tf.gather(label_embedding, negative_list)
        predic_neg_distance = tf.reduce_sum(tf.multiply(negative_embedding, predict_embedding), axis=1)

        loss = tf.add(loss, tf.maximum(0.0, (self.MARGIN - predic_true_distance + predic_neg_distance)))
        loss = tf.reduce_sum(loss)
        tf.summary.scalar('loss', loss)
        return loss

    def test(self, y_embedding, labels, label_embedding):
        predict_lable = []
        label_embedding = tf.cast(label_embedding, dtype=tf.float32)
        y_embedding = tf.expand_dims(y_embedding, axis=1)
        x_mul = tf.reduce_sum(tf.multiply(y_embedding, label_embedding), axis=2)
        predict_lable.append(tf.argmax(x_mul, axis=1))

        correct_prediction = tf.equal(predict_lable, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accurary', accuracy)
        return accuracy
