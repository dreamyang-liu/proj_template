import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch.autograd import Variable
from argparser import *
from utils import print_variable_shape

class Attention(nn.Module):

    def __init__(self, emb_dim, attn_emb_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(emb_dim, attn_emb_dim)
        self.context = nn.Parameter(torch.randn((1, attn_emb_dim))).to(TRAIN_DEVICE)
    
    def forward(self, X):
        """
        X: K * emb_dim
        """
        o = self.fc(X)
        o = torch.tanh(o)
        o = self.context @ o.t()
        o = F.softmax(o, dim=0)
        return o @ X

class Embedding(nn.Module):
    
    def __init__(self, candi_dim, emb_dim) -> None:
        super(Embedding, self).__init__()
        self._embedding = nn.Embedding(candi_dim, emb_dim)
    
    def forward(self, X):
        return self._embedding(X)

class Model(nn.Module):

    def __init__(self) -> None:
        super(Model, self).__init__()
        
    
    def forward(self, r, c, u_in, u_out):
        ...
