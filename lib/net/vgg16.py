from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from vgg.net.net import Net


class Vgg16(Net):
    def __init__(self, common_params, net_params, test=False):
        super(Vgg16, self).__init__(common_params, net_params)
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
        with tf.variable_scope('vgg_16') as scope:
            with tf.variable_scope('conv1') as scope:
                temp_conv = self.conv2d("conv1_1", image, kernel_size=[3, 3, 3, 64], stride=1, istraining=istraining)

                temp_conv = self.conv2d("conv1_2", temp_conv, kernel_size=[3, 3, 64, 64], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)

            with tf.variable_scope('conv2') as scope:
                temp_conv = self.conv2d("conv2_1", temp_pool, kernel_size=[3, 3, 64, 128], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv2_2", temp_conv, kernel_size=[3, 3, 128, 128], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)

            with tf.variable_scope('conv3') as scope:
                temp_conv = self.conv2d("conv3_1", temp_pool, kernel_size=[3, 3, 128, 256], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv3_2", temp_conv, kernel_size=[3, 3, 256, 256], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv3_3", temp_conv, kernel_size=[3, 3, 256, 256], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)

            with tf.variable_scope('conv4') as scope:
                temp_conv = self.conv2d("conv4_1", temp_pool, kernel_size=[3, 3, 256, 512], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv4_2", temp_conv, kernel_size=[3, 3, 512, 512], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv4_3", temp_conv, kernel_size=[3, 3, 512, 512], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)

            with tf.variable_scope('conv5') as scope:
                temp_conv = self.conv2d("conv5_1", temp_pool, kernel_size=[3, 3, 512, 512], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv5_2", temp_conv, kernel_size=[3, 3, 512, 512], stride=1,
                                        istraining=istraining)

                temp_conv = self.conv2d("conv5_3", temp_conv, kernel_size=[3, 3, 512, 512], stride=1,
                                        istraining=istraining)

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)
            # fc to conv
            fc6 = self.local('fc6', temp_pool, kernel_size=[7, 7, 512, 4096], stride=1, pretrain=True, train=True,
                             istraining=istraining)
            fc6 = tf.nn.dropout(fc6, keep_prob=keep_prob)

            embedding = self.local('fc7', fc6, kernel_size=[1, 1, 4096, self.embedding], stride=1, pretrain=False,
                                   train=True, istraining=istraining)
            embedding = tf.nn.dropout(embedding, keep_prob=keep_prob)

            logit, embedding_weights = self.ed_weights('fc8', embedding,
                                                       kernel_size=[1, 1, self.embedding, self.num_classes],
                                                       pretrain=False, train=True, istraining=istraining)

            embedding = tf.squeeze(embedding)
            logit = tf.squeeze(logit)
            embedding_weights = tf.squeeze(embedding_weights)
            print(embedding.shape, logit.shape, embedding.shape)
        return embedding, logit, embedding_weights

    def softmax_loss(self, predicts, labels):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predicts)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection('losses', loss / self.batch_size)

        weight_loss = tf.add_n(tf.get_collection('weight_losses'), name='weight_total_loss')
        tf.add_to_collection('weight_losses', weight_loss / self.batch_size)

        total_loss = loss + weight_loss
        tf.summary.scalar('classfication_loss', loss)
        tf.summary.scalar('s_weight_total_loss', weight_loss)
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

    def loss(self, predicts, labels, embedding, embedding_weights):
        with tf.variable_scope('classfication_loss') as scope:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predicts)
            loss = tf.reduce_mean(loss)
            tf.add_to_collection('losses', loss / self.batch_size)

        # ed_loss
        with tf.variable_scope('ed_loss') as scope:
            weights_trans = tf.transpose(embedding_weights)
            x_expand = tf.expand_dims(embedding, axis=1)
            x_sub = tf.subtract(x_expand, weights_trans)
            x_squ = tf.square(x_sub)
            x_sum = tf.reduce_sum(x_squ, axis=2)
            x_sqr = tf.sqrt(x_sum)
            one_hot_labels = tf.one_hot(labels, self.num_classes)
            mask = tf.multiply(one_hot_labels, x_sqr)
            ed_loss = tf.reduce_mean(tf.reduce_sum(mask, axis=1))
            # ed_loss = tf.exp(ed_loss, name="ed_loss")
            tf.add_to_collection('ed_loss', ed_loss / self.batch_size)

        weight_loss = tf.add_n(tf.get_collection('weight_losses'), name='weight_total_loss')
        tf.add_to_collection('weight_losses', weight_loss / self.batch_size)

        total_loss = loss + weight_loss + 0.008 * ed_loss
        tf.summary.scalar('classfication_loss', loss)
        tf.summary.scalar('weight_total_loss', weight_loss)
        tf.summary.scalar('ed_loss', ed_loss)
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

    def test(self, predicts, labels):
        correct_prediction = tf.equal(tf.argmax(predicts, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def test_ed(self, embedding, weights, labels):
        pre_labels = []
        weights = tf.transpose(weights)
        print(embedding.shape, weights.shape, labels.shape)
        for i in range(self.batch_size):
            _embedding = tf.gather(embedding, i)
            x_sub = tf.subtract(_embedding, weights)
            x_squra = tf.square(x_sub)
            x_sum = tf.reduce_sum(x_squra, axis=1)
            label = tf.argmin(x_sum)
            pre_labels.append(label)
            # print(pre_labels)
        correct_prediction = tf.reduce_mean(tf.cast(tf.equal(pre_labels, labels), dtype=tf.float32))
        return correct_prediction
