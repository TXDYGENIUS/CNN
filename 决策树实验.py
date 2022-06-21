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

features = 8
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
        outt = torch.cat((outt, out[k + 1]), dim=0)
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

from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

##################################
XT = torch.tensor([item.detach().numpy() for item in X_train])
YT = torch.tensor([item.detach().numpy() for item in Y_train])
XX = torch.tensor([item.detach().numpy() for item in X_test])
YY = torch.tensor([item.detach().numpy() for item in Y_test])
XT = XT.reshape([(len(XT)), -1])
YT = YT.reshape([len(YT)])
XX = XX.reshape([len(XX), -1])
YY = YY.reshape([len(YY)])
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model = tree.DecisionTreeClassifier(criterion='entropy', random_state=30, splitter='random', max_depth=8)
model = model.fit(XT, YT)  # 训练模型
y_hat = model.predict(XX)  # 预测结果
y_hat = torch.tensor(y_hat)
score = model.score(XX, YY)  # 匹配结果得到准确率
print(score)

feature_name = []
for i in range(78):
    feature_name.append(i)
dot_data = tree.export_graphviz(model
                                , out_file=None
                                , feature_names=feature_name
                                , class_names=["0", "1"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)

# print(graph)
# print(model.tree_.compute_feature_importances(normalize=False))
'''
# 特征重要性列表
print(*zip(feature_name, model.feature_importances_))
print('---' * 60)
clf = model
children_left = clf.tree_.children_left  # 左节点编号
children_right = clf.tree_.children_right  # 右节点编号
feature = clf.tree_.feature  # 分割的变量
threshold = clf.tree_.threshold  # 分割阈值
impurity = clf.tree_.impurity  # 不纯度(gini)
n_node_samples = clf.tree_.n_node_samples  # 样本个数
value = clf.tree_.value  # 样本分布

# -------------打印------------------------------
print("children_left:", children_left)
print("children_right:", children_right)
print("feature:", feature)
print("threshold:", threshold)
print("impurity:", impurity)
print("n_node_samples:", n_node_samples)
print("value:", value)
print('---' * 30)

from inspect import getmembers

#print(getmembers(clf.tree_))
'''
result = model.tree_.compute_feature_importances(normalize=False)
result = torch.tensor(result)


# feature = torch.tensor(feature_name)
# data_dataset = TensorDataset(feature,result )  # 数据和标签打包


def choose(X, result):
    A = []
    for i in range(len(X)):
        B = []
        for j in range(78):
            if len(B) < 16:
                if result[j] != 0:
                    B.append(X[i][j].item())
        if len(B) < 16:
            for k in range(16 - len(B)):
                B.append(0)
        A.append(B)
    return A


X_train = choose(XT, result)
X_test = choose(XX, result)
X_train = np.array(X_train)
X_train = torch.tensor(X_train)
X_test = np.array(X_test)
X_test = torch.tensor(X_test)
Y_train = YT
Y_test = YY


################################################################################################

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
                print(loss.item())
            except RuntimeError:
                pass


train(6, X_train, Y_train)
