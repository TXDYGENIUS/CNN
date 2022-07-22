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
batch_size = 100
learning_rate = 0.001
file_anomalous = 'F:\pythonProject\MachineLearningCVE\PNN测试.csv'
file_2018 = 'F:\pythonProject\MachineLearningCVE\yichang.csv'


# device = torch.device('cuda:0')


# 可以任意使用batch，GPU速度明显快于CPU，输入数据集标准化
# 读取文件转化成tensor并存入列表，删除第一行标题，返回tensor
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
        out[j] = out[j].view(1, -1)
    outt = out[0]
    for k in range(len(out) - 1):
        outt = torch.cat((outt, out[k + 1]), dim=0)
    laber = torch.tensor(laber)

    return outt, laber


'''
out, label = open_file(file_anomalous)
print(out.size(), '//', label.size())  =  torch.Size([18000, 78]) // torch.Size([18000])
'''


def get_data(file):
    X, Y = open_file(file)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=666)
    return X_train, X_test, Y_train, Y_test


def mini_batch(X, Y, batch_size):
    X = X.view(-1, batch_size, X.size()[-1])
    Y = Y.view(-1, batch_size)
    return X, Y


def standardization(data):
    for i in range(len(data)):
        sample_sum = torch.sum(data[i]) / data.view(-1).size()[-1]
        data[i] = data[i] / sample_sum
    return data


X_train, X_test, Y_train, Y_test = get_data(file_anomalous)
X_train, Y_train = mini_batch(X_train, Y_train, batch_size)
X_test, Y_test = mini_batch(X_test, Y_test, batch_size)


# X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


# print(X_train.size(), X_test.size(), Y_train.size(), Y_test.size())
# torch.Size([4800, 3, 78]) torch.Size([1200, 3, 78]) torch.Size([4800, 3]) torch.Size([1200, 3])


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        # 实例化预定义的nn.GRU, 它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        inputs = torch.nn.functional.normalize(inputs)  # 正则化
        inputs = inputs.unsqueeze(0)
        # print(inputs.size(), hidden.size())=torch.Size([1, 3, 78]) torch.Size([1, 3, 16])
        rr, hn = self.gru(inputs, hidden)
        return self.sigmoid(self.linear(rr)), hn

    def initHidden(self):
        return torch.randn(self.num_layers, batch_size, self.hidden_size)


input_size, hidden_size = 78, 16  # 输入层特征大小，隐含层特征大小
model = GRU(input_size, hidden_size, 1, num_layers=1)  # 输出output_size = 1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 用Adam优化器
criterion = nn.BCELoss()  # 损失函数


# model = model.to(device)


def train(epoch, X_train, Y_train):
    hidden = model.initHidden()
    #hidden = hidden.to(device)
    for e in range(epoch):
        for i in range(len(X_train)):
            X_train[i] = Variable(X_train[i].float(), requires_grad=True)
            # 然后将模型结构中的梯度归0
            optimizer.zero_grad()
            output, hidden = model(X_train[i].float(), hidden.float())
            hidden = hidden.data
            loss = criterion(output.view(-1), Y_train[i].float())
            # 损失进行反向传播
            loss.backward(retain_graph=True)
            # 更新模型中所有的参数
            optimizer.step()
            # 返回结果和损失的值
            # print(output, '\n', loss.item())


train(8, X_train, Y_train)


def test(X_test, Y_test):
    a = 0
    hidden = model.initHidden()
#    hidden = hidden.to(device)
    for i in range(len(X_test)):
        X_test[i] = X_test[i].to(torch.float32)

        model.eval()

        outputs, hidden = model(X_test[i].squeeze(1).float(), hidden.float())
        hidden = hidden.data

        outputs = outputs.view(-1)

        for j in range(batch_size):
            if outputs[j].item() >= 0.5 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() < 0.5 and Y_test[i][j].item() == 0:
                a = a + 1

    print(a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


PATH = "./Final_GRU.txt"

torch.save(model.state_dict(), PATH)  # 保存模型的状态字典
# 首先实例化模型的类对象
model = GRU(input_size, hidden_size, 1, num_layers=1)
# model.to(device)
# 加载训练阶段保存好的模型的状态字典
model.load_state_dict(torch.load(PATH))

test(X_test, Y_test)
time_end = time.time()
print('总时长 ： ', time_end - time_start)
