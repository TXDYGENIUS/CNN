import re
import csv
import numpy
import torch.nn as nn
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

# https://www.bilibili.com/video/BV13i4y1R7jB?spm_id_from=333.999.0.0
'''
bs, T = 2, 3  # 批大小（单次传递给程序用以训练的参数个数），输入序列长度
input_size, hidden_size = 2, 3  # 输入层特征大小，隐含层特征大小
input = torch.randn(bs, T, input_size)  # 随机初始化一个输入特征序列。(单词数量，句子数量，每个词特征维度)
h_prev = torch.zeros(bs, hidden_size)  # 初始隐含状态
# step1  调用RNN的API
rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # （输入维度，隐藏层神经元个数，输入数据的形式）
rnn_output, state_final = rnn(input, h_prev.unsqueeze(0))  # (在第0个位置增加维度)

print(input.size(), h_prev.unsqueeze(0).size())
print(rnn_output.size(), '\n', state_final)


# step2 实现RNN计算原理
def rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape(0)  # h的维度
    h_out = torch.zeros(bs, T, h_dim)
    for i in range(T):
        x = input[:, T, :]  # 获取当前时刻输入  [bs(全部)，第T时刻(第T句话)，特征维度(全部)]
        w_ih_batch = weight_ih[:, :, T]
        wih = torch.bmm(w_ih_batch, x)
        w_hh_batch = weight_hh[:, :, T]
        whh = torch.bmm(w_hh_batch, h_prev)
        h_prev = torch.tanh(wih + whh + bias_hh + bias_ih)
        h_out = h_prev[:, T, :]
    return h_out, h_prev
'''


print('+++++++++++++++++++++++++++++++++++++')
#########################################################################################################

batch_size = 1
file_anomalous = 'F:\pythonProject\MachineLearningCVE\yichang.csv'
file_normal = 'F:\pythonProject\MachineLearningCVE\zhengchang.csv'


# 读取文件转化成tensor并存入列表，删除第一行标题，返回tensor
def open_file(file):
    out = []
    laber = []
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for i in reader:
            out.append(i)
    del out[0]  # 删除第一行标题
    for j in range(len(out)):  # 数据集中最后一列为标签，若为BENIGN则是正常
        if out[j][-1] == 'Benign':
            laber.append(0)
        else:
            laber.append(1)
        del out[j][-1]  # 删除标签列方便下面保存数据
        for k in range(len(out[j])):
            out[j][k] = float(out[j][k])  # 把原数据的str转化为float
        out[j] = torch.DoubleTensor(out[j] + ([0.0, 0.0, 0.0]))  # 原长度为78，加三个0方便分为9*9
        out[j] = out[j].view([-1, 9, 9])
    outt = out[0]
    for k in range(len(out) - 1):
        outt = torch.cat((outt, out[k+1]), dim=0)
    laber = torch.tensor(laber)
    return outt, laber  # 输出为数据，标签


X, Y = open_file(file_normal)
data_dataset = TensorDataset(X, Y)  # 数据和标签打包
XX = []
YY = []
# data_dataset 是列表，每个数据为[数据，标签]
train_loader = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=True)

for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)
# print(X_train[0].size()) = torch.Size([1, 9, 9])
########################################################################################################

# RNN的使用方法：
bs, T = 1, 3  # 批大小（单次传递给程序用以训练的参数个数），输入序列长度
input_size, hidden_size = 9, 9  # 输入层特征大小，隐含层特征大小
# input = torch.randn(bs, T, input_size)  # 随机初始化一个输入特征序列。(单词数量，句子数量，每个词特征维度)
h_prev = torch.zeros((bs, hidden_size))  # 初始隐含状态
# step1  调用RNN的API
rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # （输入维度，隐藏层神经元个数，输入数据的形式）
X_train[0] = X_train[0].float()
rnn_output, state_final = rnn(X_train[0], h_prev.unsqueeze(0))  # (在第0个位置增加维度)

print(X_train[0].size(), h_prev.unsqueeze(0).size())
print(rnn_output.size(), '\n', state_final)

for i in range(len(X_train)):
    X_train[i] = X_train[i].float()
    rnn_output, state_final = rnn(X_train[i], h_prev.unsqueeze(0))  # (在第0个位置增加维度)
    print(rnn_output.size(), '\n', state_final)
###################################################################################

# LSTM 的使用方法：
rnn = nn.LSTM(9, 9, 2, bidirectional=False)  # (输入张量x中特征维度的大小,隐层张量h中特征维度的大小,隐含层的数量.是否用双向lstm)
h0 = torch.randn(2, 9, 9)  # 初始化的隐层张量h.(隐含层的数量,)
c0 = torch.randn(2, 9, 9)  # 初始化的细胞状态张量c.
# 定义隐藏层初始张量和细胞初始状态张量的参数含义:
# (num_layers * num_directions[两层*单向], batch_size[与输入的batch_size保持一致], hidden_size[与输入的hidden_size保持一致])
input = torch.randn([1, 9, 9])
output, (hn, cn) = rnn(input, (h0, c0))

# RGU使用方法
rnn = nn.GRU(9, 9, 2, bidirectional=False)
input = torch.randn(1, 9, 9)
h0 = torch.randn(2, 9, 9)  # 初始化的隐层张量h.
output, hn = rnn(input, h0)
