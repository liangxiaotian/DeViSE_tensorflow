"""数据集读取的父类，包括初始化函数和批量读取函数"""


class DataSet(object):

    def __init__(self, common_params, dataset_params):
        """相关的参数设置"""
        raise NotImplementedError


    def batch(self):
        """get batch"""
        raise NotImplementedError