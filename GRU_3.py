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


# 此为半成品，除了W_hq,b_q之外其余参数无法随反向传播而更新
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


#####################  GRU  #################################
class GRU(nn.Module):

    def __init__(self, num_inputs, num_hiddens, num_outputs):  # 每个数据特征维度=78，GRU中间层=16，GRU输出层=1
        super(GRU, self).__init__()
        self.W_xz = torch.randn(num_inputs, num_hiddens)
        self.W_xr = torch.randn(num_inputs, num_hiddens)
        self.W_xh = torch.randn(num_inputs, num_hiddens)
        self.W_hz = torch.randn(num_hiddens, num_hiddens)
        self.W_hr = torch.randn(num_hiddens, num_hiddens)
        self.W_hh = torch.randn(num_hiddens, num_hiddens)
        self.b_z = torch.zeros(num_hiddens)
        self.b_r = torch.zeros(num_hiddens)
        self.b_h = torch.zeros(num_hiddens)
        self.W_hq = torch.randn((num_hiddens, num_outputs))
        self.b_q = torch.zeros(num_outputs)
        self.state = torch.zeros(batch_size, num_hiddens)

        self.params = [self.W_xz, self.W_hz, self.b_z, self.W_xr, self.W_hr, self.b_r, self.W_xh, self.W_hh, self.b_h,
                       self.W_hq, self.b_q]
        for param in self.params:
            param.requires_grad = True

    def forword(self, inputs, state):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = self.params
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        for i in range(len(inputs)):  # input = [1,5,78],此处循环1(总数据/5)次，每个input[i]为[5(batch),78]

            Z = sigmoid(torch.matmul(inputs[i].float(), W_xz) + torch.matmul(state.float(), W_hz) + b_z)
            # Z = standardization(Z)

            R = sigmoid(torch.matmul(inputs[i].float(), W_xr) + torch.matmul(state.float(), W_hr) + b_r)
            # R = standardization(R)

            state_tilda = tanh(torch.matmul(inputs[i].float(), W_xh) + torch.matmul(R * state.float(), W_hh) + b_h)

            # state_tilda = standardization(state_tilda)
            state = Z * state.float() + (1 - Z) * state_tilda  # [5,16]
            # print(state, 'SS')
            output = torch.matmul(state.float(), W_hq) + b_q

            output = sigmoid(output.view(-1))

        return output, state  # outputs为5(batch)个输出的列表，H为最终的中间层


gru = GRU(78, 16, 1)
for k, v in gru.named_parameters():
    print(k, v, v.size())


def train(epoch, X_train, Y_train):
    hidden = gru.state
    sigmoid = nn.Sigmoid()
    for e in range(epoch):
        for i in range(10):
            # X_train[i] = standardization(X_train[i])
            X_train[i] = Variable(X_train[i].float(), requires_grad=True)
            torch.autograd.set_detect_anomaly(True)
            # 然后将模型结构中的梯度归0
            for param in gru.params:
                if param.grad is not None:
                    param.grad = None
            # 计算新的state
            output, hidden = gru.forword(X_train[i], hidden)

            print(gru.W_hq)
            # 计算损失值
            loss = criterion(output, Y_train[i].float())

            # 反向传播
            loss.backward(retain_graph=True)
            # 更新参数
            for param in gru.params:
                param.data = param.data - param.grad * learning_rate
                # print(param.grad,'/*/*/*/')

            # print(loss)
    return hidden


# 测试函数
def test(X_test, Y_test, hidden):
    a = 0
    for i in range(len(X_test)):
        X_test[i] = X_test[i].to(torch.float32)
        # print(X_test[i].size())
        # print(hidden.size())
        outputs, hidden = gru.forword(X_test[i].squeeze(1), hidden)
        hidden = hidden.data
        outputs = outputs.view(-1)  # 由torch.Size([2，1])变成torch.Size([2]) 和标签y_train[i]匹配

        for j in range(batch_size):
            if outputs[j].item() >= 0.5 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() < 0.5 and Y_test[i][j].item() == 0:
                a = a + 1

    print(a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


state = train(10, X_train, Y_train)

test(X_test, Y_test, state)
time_end = time.time()
print('time cost', time_end - time_start, 's')
