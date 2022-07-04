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
batch_size = 3
learning_rate = 0.001
file_anomalous = 'F:\pythonProject\MachineLearningCVE\PNN测试.csv'
device = torch.device('cuda:0')


# CNN_4 与CNN_3 的区别主要是神经网络最终输出结果为1还是【1，0】
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
            laber.append([0, 1])
        else:
            laber.append([1, 0])
        del out[j][-1]
        out[j] = out[j] + [0, 0, 0]
        for k in range(len(out[j])):
            out[j][k] = float(out[j][k])  # 把原数据的str转化为float
        out[j] = torch.DoubleTensor(out[j])  # 原长度为78
        out[j] = out[j].view([-1, 9, 9])
    outt = out[0]
    for k in range(len(out) - 1):
        outt = torch.cat((outt, out[k + 1]), dim=0)
    laber = torch.tensor(laber)
    return outt, laber


# print(open_file(file_anomalous)[0].size()) = ([18000, 9, 9])

def get_data(file):
    X, Y = open_file(file)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=666)
    return X_train, X_test, Y_train, Y_test


def mini_batch(X, Y, batch_size):
    data_dataset = TensorDataset(X, Y)
    out = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=True)
    '''
    x_loader = DataLoader(dataset=X, batch_size=batch_size, shuffle=True)
    y_loader = DataLoader(dataset=Y, batch_size=batch_size, shuffle=True)
    '''
    return out


def standardization(data):
    for i in range(len(data)):
        sample_sum = torch.sum(data[i]) / data.view(-1).size()[-1]
        data[i] = data[i] / sample_sum
    return data


X_train, X_test, Y_train, Y_test = get_data(file_anomalous)

# X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)

train_data = mini_batch(X_train, Y_train, batch_size)
test_data = mini_batch(X_test, Y_test, batch_size)
# print(len(train_data))
# print(len(test_data))

'''
for i, data in enumerate(a):
    print(i, data.size())  0 torch.Size([10, 9, 9])
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义第一层卷积神经网络，输入通道维度=1，输出通道维度=6，卷积核大小=3*3  https://www.cnblogs.com/expttt/p/12397330.html
        self.conv1 = nn.Conv2d(1, 1, (2, 2))

        # 定义全连接网络

        # self.fc2 = nn.Linear(8, 2)
        self.fc3 = nn.Linear(16, 2)

        # self.fc1 = nn.Linear(15, 8)

    def forward(self, x):
        x = x.view(batch_size, 1, 9, 9)
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


model = CNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 用Adam优化器
criterion = nn.BCELoss()  # 损失函数


def train(epoch, X_Y):
    for q in range(epoch):
        for i, data in enumerate(X_Y):
            inputs, label = data
            inputs = standardization(inputs)
            print(label)
            optimizer.zero_grad()
            output = model(inputs.float())
            # output = output.view(-1)
            print(output)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()

            print(loss.item(), '++++++++++++')


def test(X_Y):
    a = 0
    normal = torch.tensor([0, 1])
    anomalous = torch.tensor([1, 0])
    for i, data in enumerate(X_Y):
        inputs, label = data
        inputs = standardization(inputs)
        outputs = model(inputs.float())
        print(outputs)
        for j in range(batch_size):
            if outputs[j][0] > outputs[j][1] and label[j][0] == anomalous[0]:
                a = a + 1
            if outputs[j][0] <= outputs[j][1] and label[j][0] == normal[0]:
                a = a + 1

    print('正确率 ： ', a / len(X_Y) / batch_size)


train(6, train_data)
# 首先设定模型的保存路径
PATH = "./cifar_net1.txt"
torch.save(model.state_dict(), PATH)  # 保存模型的状态字典
# 首先实例化模型的类对象
model = CNN()
# model.to(device)
# 加载训练阶段保存好的模型的状态字典
model.load_state_dict(torch.load(PATH))

test(test_data)
time_end = time.time()
print('用时：', time_end - time_start, 's')
