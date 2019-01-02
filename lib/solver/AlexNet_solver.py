from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from vgg.solver.solver import Solver


class AlexNetSolver(Solver):
    """AlexNet Solver
    """

    def __init__(self, dataset, net, common_params, solver_params, net_params):
        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.num_classes = int(common_params['num_classes'])
        self.learning_rate = solver_params['learning_rate']
        self.boundaries = solver_params['boundaries']
        self.moment = float(solver_params['moment'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])

        self.dataset = dataset
        self.net = net
        self.embedding = int(net_params['embedding'])

        self.construct_graph()

    def _train(self):
        # lr decay
        # inc_step_op = tf.assign_add(self.global_step, 1, name='increment_global_step')
        lr = tf.train.piecewise_constant(self.global_step, boundaries=self.boundaries, values=self.learning_rate)
        # lr_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.MomentumOptimizer(lr, self.moment)
            grads = opt.compute_gradients(self.total_loss)
            apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        return apply_gradient_op, lr

    def construct_graph(self):
        # constuct graph
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(name='images', shape=[self.batch_size, self.image_size, self.image_size, 3],
                                     dtype=tf.float32)
        self.labels = tf.placeholder(name='labels', shape=[self.batch_size], dtype=tf.int64)
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.logit = self.net.inference(self.images, self.keep_prob, self.is_training)
        # loss
        self.total_loss = self.net.softmax_loss(self.logit, self.labels)
        self.train_acc = self.net.test(self.logit, self.labels)

        tf.summary.scalar('loss', self.total_loss)
        # train op
        self.train_op, self.lr = self._train()

    def solve(self):
        saver1 = tf.train.Saver(self.net.pretrained_collection, write_version=tf.train.SaverDef.V2)
        saver2 = tf.train.Saver(self.net.trainable_collection, write_version=tf.train.SaverDef.V2)
        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        with tf.Session(config=tfconfig) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(init)
            print('Fix...')
            ckpt = tf.train.get_checkpoint_state(self.pretrain_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver1.restore(sess, ckpt.model_checkpoint_path)
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

            for step in range(self.max_iterators):
                start_time = time.time()
                np_images, np_labels = sess.run(self.dataset['train'])
                _, loss_value = sess.run([self.train_op, self.total_loss],
                                         feed_dict={self.images: np_images, self.labels: np_labels,
                                                    self.keep_prob: 0.5, self.is_training: True})
                duration = time.time() - start_time
                assert not np.isnan(loss_value), "loss = Nan"

                if step % 10 == 0:
                    num_examples_per_step = self.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch), lr: %.5f')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch, self.lr.eval()))

                sys.stdout.flush()

                if step % 100 == 0 and step != 0:
                    summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels,
                                                                  self.keep_prob: 1, self.is_training: False})
                    summary_writer.add_summary(summary_str, step)

                    np_images, np_labels = sess.run(self.dataset['test'])
                    accuracy = sess.run(self.train_acc,
                                        feed_dict={self.images: np_images, self.labels: np_labels,
                                                   self.keep_prob: 1, self.is_training: False})
                    print('Test accuracy: %g' % (accuracy))

                if step % 1000 == 0 and step != 0:
                    saver2.save(sess, self.train_dir + 'DeViSE/model.ckpt', self.global_step)

                if step % 5000 == 0 and step != 0:
                    total_acc = 0
                    total_acc_ed = 0
                    for i in range(250):
                        np_images, np_labels = sess.run(self.dataset['test'])
                        accuracy = sess.run(self.train_acc,
                                            feed_dict={self.images: np_images, self.labels: np_labels,
                                                       self.keep_prob: 1, self.is_training: False})
                        total_acc += accuracy

                    print('Test at 5000s accuracy: %g' % (total_acc / 250))
