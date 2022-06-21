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
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# https://www.cnblogs.com/pinard/p/6243025.html
# https://blog.csdn.net/weixin_44781900/article/details/104839136?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-104839136-blog-80009377.pc_relevant_antiscanv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-104839136-blog-80009377.pc_relevant_antiscanv3&utm_relevant_index=6
#########################################################################################################
features = 16
batch_size = 1
file_anomalous = 'F:\pythonProject\MachineLearningCVE\yichang.csv'
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
    for j in range(len(out)):  # 数据集中最后一列为标签，若为BENIGN则是正常
        if out[j][-1] == 'BENIGN':
            laber.append(0)
        else:
            laber.append(1)
        del out[j][-1]  # 删除标签列方便下面保存数据
        for k in range(len(out[j])):
            out[j][k] = float(out[j][k])  # 把原数据的str转化为float
        out[j] = torch.DoubleTensor(out[j])  # 原长度为78
        out[j] = out[j].view([1, -1])
    outt = out[0]
    for k in range(len(out) - 1):
        outt = torch.cat((outt, out[k+1]), dim=0)
    laber = torch.tensor(laber)
    return outt, laber  # 输出为数据，标签


X, Y = open_file(file_anomalous)


# 数据分训练集测试集
def classification(X, Y):
    data_dataset = TensorDataset(X, Y)  # 数据和标签打包
    XX = []
    YY = []
    # data_dataset 是列表，每个数据为[数据，标签]
    train_loader = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=True)

    for idx, datat in enumerate(train_loader):
        inputs, labels = datat
        XX.append(inputs)
        YY.append(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.3, random_state=666)
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = classification(X, Y)

########################################################################################################
X = X_train[0].reshape(1, -1)

for i in range(len(X_train) - 1):
    X = torch.cat((X, X_train[i + 1].reshape(1, -1)), dim=0)  # print(X.size()) = ([3173, 81])
XX = X_test[0].reshape(1, -1)
for i in range(len(X_test) - 1):
    XX = torch.cat((XX, X_test[i + 1].reshape(1, -1)), dim=0)  # torch.Size([794, 81])

PCA(copy=False, n_components=features, whiten=True)
model = PCA(n_components=features)  # 保留的特征数量
model.fit(X)
X_new = model.fit_transform(X)
X_new = torch.tensor(X_new)
print(model.explained_variance_ratio_)
print(X_new[0])

model = PCA(n_components=features)  # 保留的特征数量
model.fit(XX)
XX_new = model.fit_transform(XX)
XX_new = torch.tensor(XX_new)
print(XX_new.size())


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
            nn.MaxPool2d(kernel_size=2),  # 池化层2*2
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

        self.fc1 = nn.Linear(features, 1)  # 全连接层
        # self.fc2 = nn.Linear(16, 2)
        # self.fc3 = nn.Linear(18, 1)

    def forward(self, x):
        x = torch.nn.functional.normalize(x)  # 正则化
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(-1, x.shape[0])
        x = x.float()
        # x = self.relu(self.fc1(x))
        # x = x.float()
        # x = self.relu(self.fc2(x))

        x = self.sigmoid(self.fc1(x))
        return x


model = cnn()
optimizer = optim.Adam(model.parameters(), lr=0.08)  # 用Adam优化器
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


train(18, X_new, Y_train)

PATH = "./cifar_net1.txt"
# torch.save(model.state_dict(), PATH)

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
            if outputs[j].item() >= 0.5 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() < 0.5 and Y_test[i][j].item() == 0:
                a = a + 1

    print(a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


test(XX_new, Y_test)
