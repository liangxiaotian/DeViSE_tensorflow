from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import time
from DeViSE_tensorflow.lib.utils.process_config import get_params
from DeViSE_tensorflow.lib.solver.DeViSE_solover import AlexNetSolver
from DeViSE_tensorflow.lib.net.DeViSE_AlexNet import AlexNet
from DeViSE_tensorflow.lib.dataset import cifar100_input
from DeViSE_tensorflow.lib.dataset.read_label_embedding import get_labels_embedding
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append('../')

common_params, dataset_params, net_params, solver_params = get_params()
# train_dataset = TextDataset(common_params, dataset_params, istraining=True)
# test_dataset = TextDataset(common_params, dataset_params, istraining=False)
train_dataset = cifar100_input.inputs(eval_data=False, data_dir=str(dataset_params['dataset_dir']),
                                      batch_size=int(common_params['batch_size']))
test_dataset = cifar100_input.inputs(eval_data=True, data_dir=str(dataset_params['dataset_dir']),
                                     batch_size=int(common_params['batch_size']))
label_embedding = get_labels_embedding(dataset_params['label_embedding_dir'])
dataset = {'train': train_dataset,
           'test': test_dataset,
           'label_embedding': label_embedding}
net = AlexNet(common_params, net_params)
solver = AlexNetSolver(dataset, net, common_params, solver_params, net_params)
solver.solve()
