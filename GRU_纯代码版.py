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

time_start = time.time()
batch_size = 1
learning_rate = 0.1
file_anomalous = 'F:\pythonProject\MachineLearningCVE\PNN测试.csv'
file_test = 'F:\pythonProject\MachineLearningCVE\有顺序测试集.csv'
label_class = [0, 1]  # 所有的标签类别
criterion = nn.BCELoss()  # 损失函数


# 标准化函数(所有元素除以数据平均值)
def standardization(data):
    for i in range(len(data)):
        sample_sum = torch.sum(data[i]) / data.view(-1).size()[-1]
        data[i] = data[i] / sample_sum
    return data


def open_file(file):
    out = []
    laber = []
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for i in reader:
            out.append(i)
    del out[0]  # 删除第一行标题
    for j in range(len(out)):
        if out[j][-1] == 'BENIGN':
            laber.append(0)
        else:
            laber.append(1)
        del out[j][-1]
        for k in range(len(out[j])):
            out[j][k] = float(out[j][k])  # 把原数据的str转化为float
        out[j] = torch.DoubleTensor(out[j])  # 原长度为78
        out[j] = out[j].view([-1, 1, 78])
    outt = out[0]
    for k in range(len(out) - 1):
        outt = torch.cat((outt, out[k]), dim=0)
    laber = torch.tensor(laber)
    return outt, laber


def get_data(file):
    X, Y = open_file(file)
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

    # print(X_train[0].size()) = [1,78]

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = get_data(file_anomalous)


# print(len(X_train))=2880
# print(X_train[0].size())=([5, 1, 78])
# print(Y_train[0].size())=([5])


def GRU_forward(inputs, initial_states, w_ih, w_hh, b_ih, b_hh):
    prev_h = initial_states
    bs, T, i_size = inputs.shape  # batch_size, ?,单个元素特征维度
    h_size = w_ih.shape[0] // 3  # 三个与输入input相关的矩阵，所以除以3分开 = 隐含神经元个数 = 中间层
    # 对权重扩维
    batch_w_ih = w_ih.unsqueeze(0).tile(bs, 1, 1)  # 原本的w_ih为二维，而输入为三维，扩充一维，tile把扩充的第一维乘以batch
    batch_w_hh = w_hh.unsqueeze(0).tile(bs, 1, 1)
    # 输出
    output = torch.zeros(bs, T, h_size)
    for t in range(T):
        x = inputs[:, t, :]  # 取input的第t个时刻(当前时刻输入的特征向量)  x.size()=[bs , i_size]
        w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [bs , 3*h_size , 1]
        w_times_x = w_times_x.squeeze(-1)  # [bs , 3*h_size]

        w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [bs , 3*h_size , 1]
        w_times_h_prev = w_times_h_prev.squeeze(-1)  # [bs , 3*h_size]
        # 重置门
        r_t = torch.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size] + b_hh[:h_size] + b_ih[:h_size])
        # 更新门
        z_t = torch.sigmoid(w_times_x[:, h_size:2 * h_size] + w_times_h_prev[:, h_size:2 * h_size] +
                            b_hh[h_size:2 * h_size] + b_ih[h_size:2 * h_size])
        # 中间状态
        n_t = torch.tanh(w_times_x[:, 2 * h_size:3 * h_size] + b_ih[2 * h_size:3 * h_size]) + \
              r_t * (w_times_h_prev[:, 2 * h_size:3 * h_size] + b_hh[2 * h_size:3 * h_size])
        # 新的隐含状态
        prev_h = (1 - z_t) * n_t + z_t * prev_h
        output[:, t, :] = prev_h

    return output, prev_h


bs, T, i_size, h_size = 2, 3, 4, 2
inputs = torch.randn(bs, T, i_size)
h0 = torch.randn(bs, h_size)
model = nn.GRU(i_size, h_size, batch_first=True)
output, h_final = model(inputs, h0.unsqueeze(0))


for i in range(10):
    output, h_final = model(inputs, h_final)
    for k, v in model.named_parameters():
        print(k, v)
        break