import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import csv
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import torch.nn.parameter as parameter
import time
import math
import copy

seq_len = 2  # 数据长度
batch = 3
input_size = 78  # 单个数据特征数
num_layers = 1  # 隐藏层层数
hidden_size = 2  # 隐藏层节点数量 = 输出向量的维度
input = torch.randn(seq_len, batch, input_size)  # 单词数量，句子数量，每个单词的特征维度
h0 = torch.randn(num_layers, batch, hidden_size)  # 隐藏层状态(层数，batch，每层节点数)
'''
GRU = nn.GRU(input_size, hidden_size, num_layers, bidirectional=False)
output, h1 = GRU(input, h0)
print(output.size(), '\n', output)
print(h1.size(), '\n', h1)
'''

num_inputs = input_size
num_hiddens = hidden_size
num_outputs = 16


def get_params():
    def _one(shape):
        return torch.randn(shape).normal_(std=0.01)  # normal为权重初始化

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens))

    W_xz, W_hz, b_z = _three()  # Update gate parameter
    W_xr, W_hr, b_r = _three()  # Reset gate parameter
    W_xh, W_hh, b_h = _three()  # Candidate hidden state parameter
    # Output layer parameters
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs)
    # Create gradient
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens):
    return (torch.zeros(size=(batch_size, num_hiddens)))


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        print(X.size())
        m = nn.Sigmoid()
        Z = m(torch.matmul(X.float(), W_xz) + torch.matmul(H.float(), W_hz) + b_z)
        R = m(torch.matmul(X.float(), W_xr) + torch.matmul(H.float(), W_hr) + b_r)
        h = nn.Tanh()
        H_tilda = h(torch.matmul(X.float(), W_xh) + torch.matmul(R * H.float(), W_hh) + b_h)
        H = Z * H.float() + (1 - Z) * H_tilda
        Y = torch.matmul(H.float(), W_hq) + b_q
        outputs.append(Y)

    return outputs, (H)


para = get_params()
state = init_gru_state(batch, num_hiddens)

print(gru(input, state, para))
