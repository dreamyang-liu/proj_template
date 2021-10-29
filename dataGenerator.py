from sklearn import preprocessing
from tqdm import tqdm
from Data import *
from argparser import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import torch
import multiprocessing

from utils import print_variable_shape

def example_process_worker(batch, return_list, idx):
    res = []
    for _ in tqdm(batch):
        ...
    return_list[idx] = res

def example_process(data, process=1):
    manager = multiprocessing.Manager()
    return_list = manager.dict()
    batch_size = int(len(data) / process) + 1
    jobs = []
    get_batch = lambda x: data[batch_size * x:batch_size * x + batch_size]
    for i in range(process):
        p = multiprocessing.Process(target=example_process_worker, args=(get_batch(i), return_list, i))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    res = []
    for i in range(process):
        res.extend(return_list[i])
    return res


class DataGenerator(object):

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.normlizer = StandardScaler(with_mean=0, with_std=1)
        self.normlizer.fit(self.feature['train'].values.reshape(-1, 1))

        self.label_normalizer = StandardScaler()
        self.label_normalizer.fit(self.label['train'].values.reshape(-1, 1))
    
    def _core_process(self, _feature):
        ...

    def get_eval_features(self, feature):
        _feature = feature['eval']
        return self._core_process(_feature)
    
    def inverse_label(self, prediction):
        return self.label_normalizer.inverse_transform(prediction)
    
    def get_train_features(self, normalize=False):
        _feature = self.feature['train']
        label = self.label['train'].values
        if normalize:
            label = self.label_normalizer.transform(label.reshape(-1, 1))
        _label = torch.tensor(label, dtype=torch.float)

        # print_variable_shape(R, C, u_out, u_in)
        return self._core_process(_feature), _label
    
    def get_vali_features(self, normalize=False):
        _feature = self.feature['vali']
        label = self.label['vali'].values
        if normalize:
            label = self.label_normalizer.transform(label.reshape(-1, 1))
        _label = torch.tensor(label, dtype=torch.float)

        # print_variable_shape(R, C, u_out, u_in)
        return self._core_process(_feature), _label




if __name__ == "__main__":
    file = 'train.csv'
    vdp = DataProcesser()
    vdp.read(file)
    # vdp.summary()
    feature, label = vdp.default_process(split_dataset=True, split_ratio=0.8, vali_set=True, eval=False)

    dg = DataGenerator(feature, label)
    dg.get_train_features()
