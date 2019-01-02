common_params = {'image_size': '224',
                 'batch_size': '256',
                 'num_classes': '196',
                 'margin': '0.1',
                 }

dataset_params = {'dataset_dir': '/home/xiaotian/workspaces/data/cifar-100',
                  'label_embedding_dir': '../data/label_embedding.npy',
                  'train_path': '../data/train.txt',
                  'test_path': '../data/test.txt',
                  }

net_params = {'weight_decay': '0.0005',
              'embedding': '300'}

solver_params = {'learning_rate': [0.01, 0.001, 0.0001],
                 'boundaries': [3000, 6000],
                 'moment': '0.9',
                 'max_iterators': '100000',
                 'pretrain_model_path': '../models/pretrain/',
                 'train_dir': '../models/train/',
                 'fine_tune_projection_layer_dir': 'DeViSE',
                 'fine_tune_vs_layer_dir': 'VS',
                 }


def get_params():
    return common_params, dataset_params, net_params, solver_params
