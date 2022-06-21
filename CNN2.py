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

batch_size = 1
learn = 0.001
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
        if out[j][-1] == 'BENIGN':
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


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(  # 第一层卷积神经网络，输入通道数1，输出通道数1，卷积核2*2
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(2, 2),
            ),
            nn.BatchNorm2d(1),  # 归一化处理，通道数（1）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),# 池化层2*2
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
        )

        self.fc1 = nn.Linear(1 * 1 * 16, 1)# 全连接层
        # self.fc2 = nn.Linear(6, 2)
        # self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.nn.functional.normalize(x)  # 正则化
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc1(x))
        return x


model = cnn()
optimizer = optim.Adam(model.parameters(), lr=0.008)  # 用Adam优化器
criterion = nn.BCELoss()  # 损失函数


# 训练函数
def train(epoch_num, X_train, Y_train):  # 训练轮数
    # train_loader, test_loader = load_data()
    for epoch in range(epoch_num):
        for i in range(len(X_train)):
            try:
                X_train[i] = Variable(X_train[i].unsqueeze(1).float(), requires_grad=True)
                X_train[i] = X_train[i].view(batch_size, 1, X_train[i].size(-1), -1)
                optimizer.zero_grad()
                output = model(X_train[i])
                output = output.view(-1)
                Y_train[i] = Y_train[i].view(-1)
                loss = criterion(output, Y_train[i].float())
                loss.backward()
                optimizer.step()
                # print(loss.item())
            except RuntimeError:
                pass


train(30, X_train, Y_train)

PATH = "./cifar_net1.txt"
torch.save(model.state_dict(), PATH)

model = cnn()
# 加载训练阶段保存好的模型的状态字典
model.load_state_dict(torch.load(PATH))


# 测试函数
def test(X_test, Y_test):
    a = 0
    for i in range(len(X_test)):
        X_test[i] = X_test[i].to(torch.float32)
        outputs = model(X_test[i].unsqueeze(1))
        outputs = outputs.view(-1)  # 由torch.Size([2，1])变成torch.Size([2]) 和标签y_train[i]匹配

        for j in range(batch_size):
            if outputs[j].item() - 0.5 >= 0 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() - 0.5 < 0 and Y_test[i][j].item() == 0:
                a = a + 1

    print(a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


'''
#############################################################################
for epoch in range(epoch_num):
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        data, target = Variable(data), Variable(target.long())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# '''
