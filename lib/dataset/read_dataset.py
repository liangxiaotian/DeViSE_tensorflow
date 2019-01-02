from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
# from PIL import Image, ImageDraw
import numpy as np
from multiprocessing import Queue
# from Queue import Queue
from threading import Thread

from vgg.dataset.dataset import DataSet


class TextDataset(DataSet):
    """数据集的读取类别"""

    def __init__(self, common_params, dataset_params, istraining):
        if istraining:
            self.data_path = str(dataset_params['train_path'])
        else:
            self.data_path = str(dataset_params['test_path'])

        self.num_classes = int(common_params['num_classes'])
        self.batch_size = int(common_params['batch_size'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.thread_num = int(dataset_params['thread_num'])

        # 创建队列
        self.record_queue = Queue(maxsize=1000)
        self.image_label_queue = Queue(maxsize=512)

        self.record_list = []
        input_file = open(self.data_path, 'r')

        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
            # 将数字转换为浮点型
            ss[1:] = [float(num) for num in ss[1:]]
            self.record_list.append(ss)

        self.record_point = 0
        self.record_numbet = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_numbet / self.batch_size)
        # 开启线程调用函数，读取数据
        t_record_producter = Thread(target=self.record_producter)
        # 设置所有线程一起结束
        t_record_producter.daemon = True
        t_record_producter.start()
        # print(t_record_producter)

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_customer(self):
        # 队列的操作，从文件队列中取数据信息，处理信息之后放入label队列
        while True:
            # 从文件队列里面读取记录
            item = self.record_queue.get()
            # print(item)
            # 根据读取的记录，读取图像和label
            out = self.record_propross(item)
            # print(out)
            # 添加进图像队列
            self.image_label_queue.put(out)

    def record_producter(self):
        """队列处理,图像列表随机生成"""
        while True:
            # 每循环列表一次随机生成一次列表
            if self.record_point % self.record_numbet == 0:
                # random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def record_propross(self, record):
        """数据读取与处理， record:image-path， label
        return：image:
        labels:
        """
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        image = cv2.resize(image, (self.height, self.width))
        labels = record[1]

        return [image, labels]

    def dense_to_one_hot(self, labels, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: 3-D ndarray [batch_size, max_objects, 5]
          objects_num: 1-D ndarray [batch_size]
        """
        images = []
        labels = []
        for i in range(self.batch_size):
            image, label = self.image_label_queue.get()
            # print(image, label, object_num)
            images.append(image)
            labels.append(label)

        images = np.asarray(images, dtype=np.float32)
        images = np.divide(np.subtract(images, 127.5), 128)
        labels = np.asarray(labels, dtype=np.int32)
        return images, labels
