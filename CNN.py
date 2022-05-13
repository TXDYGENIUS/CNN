import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import math

batch_size = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络，输入通道维度=1，输出通道维度=6，卷积核大小=3*3  https://www.cnblogs.com/expttt/p/12397330.html
        self.conv1 = nn.Conv2d(1, 1, (3, 3))
        # 定义第二层卷积神经网络，输入通道维度=6，输出通道维度=16，卷积核大小=3*3  https://blog.csdn.net/weixin_38481963/article/details/109924004
        self.conv2 = nn.Conv2d(1, 1, (3, 3))
        # 定义三层全连接网络
        self.fc1 = nn.Linear(15, 8)  # 第一城输出15，第二层就输入10  （10为任意值）
        self.fc2 = nn.Linear(8, 2)  # 3也为任意值
        self.fc3 = nn.Linear(2, 1)  # 10 为最终输出10分类（0~9）

    def forward(self, x):
        # 在2*2的池化窗口下进行最大值池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3))
        x = x.view(-1, self.num_fat_features(x))  # 铺平，转化成1行x列的数据方便输入全连接层
        # print(x.size())=([2, 15])
        x = F.relu(self.fc1(x))  # view中-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def num_fat_features(self, x):
        # 计算size，除了第0个维度上的batch_size
        size = x.size()[1:]  # 把三维向量后边那两维取出
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net = net.double()
# 创建优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.01)  # SGD = 随机梯度下降   （net网络中所有参数，学习率=0.01）
# 将优化器实现梯度清零
optimizer.zero_grad()

# output = net(input)
criterion = nn.BCELoss()
# loss = criterion(output, target)
# loss.backward()  #对损失值反向传播
optimizer.step()  # 对于参数进行更新

normal = np.loadtxt("Normal")
anomalous = np.loadtxt("Anomalous")

all_requests = np.concatenate([normal, anomalous])  # 把两个矩阵第一个维度相加
X = all_requests
X = torch.tensor(X)
X = X.reshape(len(X), 56, 35)
# print(X.shape)=torch.Size([60668, 1960])
# print(X[0])
# print(X[0].shape)  =torch.Size([1960])
# print(len(X[0][0]),'////////////////')

# print(XX[0].shape)=torch.Size([56, 35])
y_normal = np.zeros(shape=(normal.shape[0]))
y_anomalous = np.ones(shape=(anomalous.shape[0]))
y = np.concatenate([y_normal, y_anomalous])
y = torch.tensor(y)
# -----------------------
data_dataset = TensorDataset(X, y)

XX = []
YY = []
train_loader = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=True)
for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)

# print(X_train[0].size())=[1,11]

print('++++++++' * 10)

for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(len(X_train)):

        # 首先将优化器梯度归零
        optimizer.zero_grad()

        # 输入图像张量进网络, 得到输出张量outputs
        outputs = net(X_train[i].unsqueeze(1))
        outputs = outputs.view(-1)  # 由torch.Size([2，1])变成torch.Size([2]) 和标签y_train[i]匹配
        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, y_train[i])
        print(outputs)
        # 反向传播+参数更新, 是标准代码的标准流程
        loss.backward()
        optimizer.step()

        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            # print('[%d, %5d] loss: %.3f' %
            # (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 首先设定模型的保存路径
PATH = './cifar_net.txt'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)

# 首先实例化模型的类对象
net = Net()
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))
print(len(X_test))
a = 0
for i in range(len(X_test)):
    X_test[i] = X_test[i].to(torch.float32)
    outputs = net(X_test[i].unsqueeze(1))
    outputs = outputs.view(-1)  # 由torch.Size([2，1])变成torch.Size([2]) 和标签y_train[i]匹配

    for j in range(2):
        if outputs[j].item() - 0.5 >= 0 and y_test[i][j].item() == 1:
            a = a + 1
        if outputs[j].item() - 0.5 < 0 and y_test[i][j].item() == 0:
            a = a + 1

print(a / len(X_test) / 2)  # 因为每个output有两个数据
