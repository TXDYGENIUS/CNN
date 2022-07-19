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
batch_size = 10
learning_rate = 0.001
file_anomalous = 'F:\pythonProject\MachineLearningCVE\PNN测试.csv'
device = torch.device('cuda:0')

def Normalization(data):
    '''样本数据归一化
    input:data(mat):样本特征矩阵
    output:Nor_feature(mat):归一化的样本特征矩阵
    '''
    m, n = data.size()
    # print(n)
    x = copy.deepcopy(data)  # 深复制，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
    # =号复制类似于贴标签，其实共用一块内存
    sample_sum = torch.sqrt(torch.sum(torch.square(data), 1))  # 数据平方(square)之后，行求和(axis=1),在开方(sqrt)
    Nor_feature = torch.ones(m, n)  # PS . 如果直接用copy.deepcopy会使结果全0
    for i in range(m):
        for j in range(n):
            # print(data[i][j], sample_sum[i], data[i][j] / sample_sum[i].item(), '++++++')
            Nor_feature[i][j] = data[i][j].item() / sample_sum[i].item()
    return Nor_feature
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义第一层卷积神经网络，输入通道维度=1，输出通道维度=6，卷积核大小=3*3  https://www.cnblogs.com/expttt/p/12397330.html
        self.conv1 = nn.Conv2d(1, 1, (2, 2))

        # 定义全连接网络

        # self.fc2 = nn.Linear(8, 2)
        self.fc3 = nn.Linear(12, 1)

        # self.fc1 = nn.Linear(15, 8)

    def forward(self, x):
        x = x.view(batch_size, 1, 6, 13)
        # print(x, '***************')
        x = F.relu(self.conv1(x))
        # print(x.size())
        # print(x, '++++++++++++')
        x = F.max_pool2d(x, (2, 2))
        # print(x.size())
        # print(x, '-------------')
        x = x.view(x.shape[0], -1)  # 铺平，转化成1行x列的数据方便输入全连接层
        # print(x.size())
        # print(x, '/////////////')
        x = torch.sigmoid(self.fc3(x))
        return x


model_CNN = CNN()
optimizer = optim.Adam(model_CNN.parameters(), lr=learning_rate)  # 用Adam优化器
criterion = nn.BCELoss()  # 损失函数


def train(epoch, X_train, Y_train):
    for q in range(epoch):
        for i in range(len(X_train)):
            inputs = X_train[i]
            inputs = standardization(inputs)
            # print(label)
            optimizer.zero_grad()
            output = model_CNN(inputs.float())
            output = output.view(-1)

            loss = criterion(output, Y_train[i].float())
            loss.backward()
            optimizer.step()
            # print(output)
            # print(loss.item(), '++++++++++++')


train(6, X_train, Y_train)


def test(X_test, Y_test):
    a = 0
    x = torch.zeros(3)
    for i in range(len(X_test)):
        inputs = X_test[i]

        inputs = standardization(inputs)
        outputs = model_CNN(inputs.float())
        # print(outputs)
        for j in range(batch_size):
            # print(outputs[j].item(), '/-*', Y_test[i][j].item())
            if outputs[j].item() >= 0.5 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() < 0.5 and Y_test[i][j].item() == 0:
                a = a + 1

    print('正确率 ： ', a / len(X_test) / batch_size)


test(X_test, Y_test)
