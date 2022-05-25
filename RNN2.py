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
import random
import time
import math

batch_size = 1
learning_rate = 0.008
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


X, Y = open_file(file_anomalous)
data_dataset = TensorDataset(X, Y)  # 数据和标签打包
XX = []
YY = []
# data_dataset 是列表，每个数据为[数据，标签]
train_loader = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=False)

for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)

# print(X_train[0].size()) = [1,78]


# RNN的使用方法：
bs, T = 1, 3  # 批大小（单次传递给程序用以训练的参数个数），输入序列长度
input_size, hidden_size = 78, 16  # 输入层特征大小，隐含层特征大小
# input = torch.randn(bs, T, input_size)  # 随机初始化一个输入特征序列。(单词数量，句子数量，每个词特征维度)
h_prev = torch.zeros((bs, hidden_size))  # 初始隐含状态
# step1  调用RNN的API
rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)  # （输入维度，隐藏层神经元个数，输入数据的形式）
X_train[0] = X_train[0].float()
rnn_output, state_final = rnn(X_train[0], h_prev.unsqueeze(0))  # (在第0个位置增加维度)
'''
print(X_train[0].size(), h_prev.unsqueeze(0).size())
print(rnn_output.size(), '\n', state_final)

for i in range(len(X_train)):
    X_train[i] = X_train[i].float()
    rnn_output, state_final = rnn(X_train[i], h_prev.unsqueeze(0))  # (在第0个位置增加维度)
    print(rnn_output, '\n', state_final)
'''


# 使用nn.RNN构建完成传统RNN使用类

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """初始化函数中有4个参数, 分别代表RNN输入最后一维尺寸, RNN的隐层最后一维尺寸, RNN层数"""
        super(RNN, self).__init__()
        self.BatchNorm2d = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = 1
        # 实例化预定义的nn.RNN, 它的三个参数分别是input_size, hidden_size, num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        """完成传统RNN中的主要逻辑, 输入参数input代表输入张量, 它的形状是1 x n_letters
           hidden代表RNN的隐层张量, 它的形状是self.num_layers x 1 x self.hidden_size"""
        input = torch.nn.functional.normalize(input)  # 正则化
        # 因为预定义的nn.RNN要求输入维度一定是三维张量, 因此在这里使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        input = self.BatchNorm2d(input.unsqueeze(0))
        # 将input和hidden输入到传统RNN的实例化对象中，如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(input.squeeze(0), hidden)
        # 将从RNN中获得的结果通过线性变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.sigmoid(self.linear(rr)), hn


'''
    def initHidden(self):
        """初始化隐层张量"""
        # 初始化一个（self.num_layers, 1, self.hidden_size）形状的0张量
        return torch.zeros(self.num_layers, 1, self.hidden_size)
'''

model = RNN(input_size, hidden_size, num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 用Adam优化器
criterion = nn.BCELoss()  # 损失函数

hidden = torch.randn(1, 1, hidden_size)


def trainRNN(epoch, line_tensor, category_tensor, hidden):
    """定义训练函数, 它的两个参数是category_tensor类别的张量表示, 相当于训练数据的标签,
       line_tensor名字的张量表示, 相当于对应训练数据"""
    # 在函数中, 首先通过实例化对象rnn初始化隐层张量
    hidden = hidden
    for e in range(epoch):
        # 下面开始进行训练, 将训练数据line_tensor的每个字符逐个传入rnn之中, 得到最终结果
        for i in range(len(line_tensor)):
            line_tensor[i] = Variable(line_tensor[i].float(), requires_grad=True)
            # 然后将模型结构中的梯度归0
            optimizer.zero_grad()
            # print(line_tensor[i].size(),hidden.size())
            output, hidden = model(line_tensor[i].squeeze(0).float(), hidden.float())
            hidden = hidden.data
            # 因为我们的rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满足于category_tensor
            # 进行对比计算损失, 需要减少第一个维度, 这里使用squeeze()方法
            loss = criterion(output.view([1]), category_tensor[i].float())
            # 损失进行反向传播
            loss.backward()
            # 更新模型中所有的参数
            optimizer.step()
            # 返回结果和损失的值

    return hidden


hidden = trainRNN(6, X_train, Y_train, hidden)
PATH = "./RNN_net1.txt"
torch.save(model.state_dict(), PATH)

# 加载训练阶段保存好的模型的状态字典
model.load_state_dict(torch.load(PATH))


# 测试函数
def test(X_test, Y_test):
    a = 0
    for i in range(len(X_test)):
        X_test[i] = X_test[i].to(torch.float32)
        print(X_test[i].size())
        print(hidden.size())
        outputs = model(X_test[i].squeeze(1), hidden.float())
        #print(outputs.size())
        #outputs = outputs.view(-1)  # 由torch.Size([2，1])变成torch.Size([2]) 和标签y_train[i]匹配

        for j in range(batch_size):
            if outputs[j].item() - 0.5 >= 0 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() - 0.5 < 0 and Y_test[i][j].item() == 0:
                a = a + 1

    print(a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


test(X_train, Y_train)
