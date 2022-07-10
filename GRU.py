import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import csv
from torch.autograd import Variable

batch_size = 1
learning_rate = 0.008
file_anomalous = 'F:\pythonProject\MachineLearningCVE\PNN测试.csv'
file_normal = 'F:\pythonProject\MachineLearningCVE\zhengchang.csv'
file_2018 = 'F:\pythonProject\MachineLearningCVE\Sunday2.csv'


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


def get_data(file):
    X, Y = open_file(file)
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

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = get_data(file_anomalous)


# print(X_train[0].size()) = [1,78]

# RGU使用方法
# 使用nn.GRU构建完成传统RNN使用类

# GRU与传统RNN的外部形式相同, 都是只传递隐层张量, 因此只需要更改预定义层的名字


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        # 实例化预定义的nn.GRU, 它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = torch.nn.functional.normalize(input)  # 正则化
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.sigmoid(self.linear(rr)), hn

    def initHidden(self):
        return torch.randn(self.num_layers, 1, self.hidden_size)


input_size, hidden_size = 78, 16  # 输入层特征大小，隐含层特征大小
model = GRU(input_size, hidden_size, 1, num_layers=1)  # 输出output_size = 1
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 用Adam优化器
criterion = nn.BCELoss()  # 损失函数


def train(epoch, X_train, Y_train):
    hidden = model.initHidden()
    for e in range(epoch):
        for i in range(len(X_train)):
            X_train[i] = Variable(X_train[i].float(), requires_grad=True)
            # 然后将模型结构中的梯度归0
            optimizer.zero_grad()
            # print(line_tensor[i].size(),hidden.size())
            output, hidden = model(X_train[i].squeeze(0).float(), hidden.float())
            hidden = hidden.data
            # 因为我们的rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满足于category_tensor
            # 进行对比计算损失, 需要减少第一个维度, 这里使用squeeze()方法


            loss = criterion(output.view([1]), Y_train[i].float())
            # 损失进行反向传播
            loss.backward()
            # 更新模型中所有的参数
            optimizer.step()
            # 返回结果和损失的值
            # print(output, '\n', loss.item())
    return hidden


hidden = train(1, X_train, Y_train)


# 测试函数
def test(X_test, Y_test, hidden):
    a = 0
    for i in range(len(X_test)):
        X_test[i] = X_test[i].to(torch.float32)
        # print(X_test[i].size())
        # print(hidden.size())
        outputs, hidden = model(X_test[i].squeeze(1), hidden.float())
        hidden = hidden.data
        # outputs = outputs.view(-1)  # 由torch.Size([2，1])变成torch.Size([2]) 和标签y_train[i]匹配
        for k, v in model.named_parameters():
            print(k, v, v.size())
            break
        for j in range(batch_size):
            if outputs[j].item() >= 0.5 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() < 0.5 and Y_test[i][j].item() == 0:
                a = a + 1

    print(a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


test(X_train, Y_train, hidden)
