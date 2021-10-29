from abc import abstractmethod
import pandas as pd
import numpy as np
import pickle as pkl
from pandas.core import base
import torch
import sys
import pdb
import re

from typing import Union, Optional

class DataProcesser(object):

    def __init__(self, *args, **kwargs):
        self.data = Union[pd.DataFrame, str, None]
        self.raw = Union[pd.DataFrame, str, None] # Used for original data archive
    
    @staticmethod
    def _read_csv(filename):
        return pd.read_csv(filename)
    
    @staticmethod
    def _read_txt(filename):
        with open(filename, 'r') as f:
            return f.read()
    
    def read(self, filename:str):
        if filename.endswith(".csv"):
            self.data = self._read_csv(filename)
            self.raw = self._read_csv(filename)
        else:
            self.data = self._read_txt(filename)
            self.raw = self._read_txt(filename)
    
    @staticmethod
    def split_feature_label():
        raise NotImplementedError("split_feature_label not implemented in subclass")
            

if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    file = 'train.csv'
    vdp = VenDataProcesser()
    vdp.read(file)
    # vdp.summary()
    feature, label = vdp.default_process(split_dataset=True, split_ratio=0.8, vali_set=True, eval=False)