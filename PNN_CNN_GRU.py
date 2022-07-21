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
label_class = [0, 1]


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

def Normalization(data):
    '''样本数据归一化
    input:data(mat):样本特征矩阵
    output:Nor_feature(mat):归一化的样本特征矩阵
    '''
    m, n = data.size()
    # print(n)
    # x = copy.deepcopy(data)  # 深复制，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
    # =号复制类似于贴标签，其实共用一块内存
    sample_sum = torch.sqrt(torch.sum(torch.square(data), 1))  # 数据平方(square)之后，行求和(axis=1),在开方(sqrt)
    Nor_feature = torch.ones(m, n)  # PS . 如果直接用copy.deepcopy会使结果全0
    for i in range(m):
        for j in range(n):
            # print(data[i][j], sample_sum[i], data[i][j] / sample_sum[i].item(), '++++++')
            Nor_feature[i][j] = data[i][j].item() / sample_sum[i].item()
    return Nor_feature


def distance(X, Y):
    # 计算两个样本之间的距离
    return torch.sqrt(torch.sum(torch.square(X - Y)))


def distance_mat(Nor_trainX, Nor_testX):
    '''计算待测试样本与所有训练样本的欧式距离
    input:Nor_trainX(mat):归一化的训练样本
          Nor_testX(mat):归一化的测试样本
    output:Euclidean_D(mat):测试样本与训练样本的距离矩阵
    '''
    m, n = Nor_trainX.size()
    p = Nor_testX.size()[0]
    Euclidean_D = torch.zeros((p, m))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i, j] = distance(Nor_testX[i, :], Nor_trainX[j, :])  # [i,:]切片，取第i行所有元素
            # [i,j]为第i个测试集test 与 第j个训练集train 的欧氏距离
    return Euclidean_D


def Gauss(Euclidean_D, sigma):
    '''测试样本与训练样本的距离矩阵对应的Gauss矩阵
    input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
          sigma(float):Gauss函数的标准差
    output:Gauss(mat):Gauss矩阵
    '''
    m, n = Euclidean_D.size()
    Gauss = torch.zeros((m, n))
    for i in range(m):
        for j in range(n):
            Gauss[i, j] = math.exp(- Euclidean_D[i, j] / (2 * (sigma ** 2)))
    return Gauss


def Prob_mat(Gauss_mat, label):
    '''测试样本属于各类的概率和矩阵
    input:Gauss_mat(mat):Gauss矩阵
          labelX(list):训练样本的标签矩阵
    output:Prob_mat(mat):测试样本属于各类的概率矩阵
    '''
    # 求概率和矩阵
    p, m = Gauss_mat.size()
    Prob = torch.zeros((p, 2))  # label的类别数 [0,1]= 2
    for i in range(p):
        for j in range(m):
            for s in range(len(label_class)):
                if label[j] == label_class[s]:  # Prob[i, s] 第i个测试集的第s类(s = 0 or 1)
                    Prob[i, s] += Gauss_mat[i, j]  # Gauss_mat[i, j] 是第i个测试集与第j个训练集的Gauss差
                    # Prob最后结果就是第i个训练集与所有标签为0的测试集Gauss差值的和与标签为1的测试集Gauss差值的和。
    Prob_mat = copy.deepcopy(Prob)  # 深复制，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
    # =号复制类似于贴标签，其实共用一块内存
    Prob_mat = Prob_mat / torch.sum(Prob, 1).unsqueeze(1)
    return Prob_mat


def class_results(Prob):
    '''分类结果
    input:Prob(mat):测试样本属于各类的概率矩阵
          label_class(list):类别种类列表
    output:results(list):测试样本分类结果
    '''
    arg_prob = torch.argmax(Prob, dim=1)  # 获取dim = 1(每行)维度中数值最大的那个元素的索引
    results = []
    for i in range(len(arg_prob)):
        results.append(label_class[arg_prob[i]])
    return results


time_data = time.time()
print('提取数据用时 ： ', time_data - time_start)


# ***************GRU********************
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


input_size, hidden_size = X_train[0].size()[-1], 16  # 输入层特征大小，隐含层特征大小
model_GRU = GRU(input_size, hidden_size, 1, num_layers=1)  # 输出output_size = 1
optimizer_GRU = optim.Adam(model_GRU.parameters(), lr=learning_rate)  # 用Adam优化器
criterion_GRU = nn.BCELoss()  # 损失函数


# model_GRU = model_GRU.to(device)

def test(X_test, Y_test):
    a = 0
    hidden = model_GRU.initHidden()

    for i in range(len(X_test)):
        X_test[i] = X_test[i].to(torch.float32)
        model_GRU.eval()
        outputs, hidden = model_GRU(X_test[i].squeeze(1).float(), hidden.float())
        hidden = hidden.data

        outputs = outputs.view(-1)

        for j in range(batch_size):
            if outputs[j].item() >= 0.5 and Y_test[i][j].item() == 1:
                a = a + 1
            if outputs[j].item() < 0.5 and Y_test[i][j].item() == 0:
                a = a + 1

    print('GRU 正确率 ： ', a / len(X_test) / batch_size)  # 因为每个output有batch个数据
    return (a / len(X_test) / batch_size)


PATH = "./GRU.txt"

# 加载训练阶段保存好的模型的状态字典
model_GRU.load_state_dict(torch.load(PATH))
test(X_test, Y_test)
time_GRU = time.time()
print('GRU训练用时 ： ', time_GRU - time_data)


# ***************CNN********************
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

# 首先设定模型的路径
PATH = "./Final_CNN.txt"

# torch.save(model_CNN.state_dict(), PATH)  # 保存模型的状态字典
# 首先实例化模型的类对象
model_CNN = CNN()
# model.to(device)
# 加载训练阶段保存好的模型的状态字典
model_CNN.load_state_dict(torch.load(PATH))


def test(X_test, Y_test):
    a = 0
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

    print('CNN正确率 ： ', a / len(X_test) / batch_size)


test(X_test, Y_test)
time_CNN = time.time()
print('CNN训练用时 ： ', time_CNN - time_GRU)

# ***************PNN********************
sigma = 0.03


def request(inputs):
    for i in range(len(inputs)):
        if inputs[i] >= 0.5:
            inputs[i] = 1
        else:
            inputs[i] = 0
    return inputs


def ALL_test(X_test, Y_test):
    a = 0

    for i in range(len(X_test)):
        ###########################
        inputs = X_test[i]
        inputs = standardization(inputs)
        outputs_cnn = model_CNN(inputs.float())
        outputs_cnn = outputs_cnn.view(-1)
        ############################
        hidden = model_GRU.initHidden()
        X_test[i] = X_test[i].to(torch.float32)
        outputs, hidden = model_GRU(X_test[i].squeeze(1).float(), hidden.float())
        hidden = hidden.data
        outputs_gru = outputs.view(-1)
        #############################
        X_train[i] = Normalization(X_train[i])
        X_test[i] = Normalization(X_test[i])
        mat = distance_mat(X_train[i], X_test[i])
        gauss = Gauss(mat, sigma)
        prob = Prob_mat(gauss, Y_train[i])
        result = class_results(prob)
        result = torch.tensor(result)
        # (result.size())
        # print(outputs_cnn.size())
        # print(outputs_gru.size())
        # print(Y_test[0].size())

        ALL_result = request(result) + request(outputs_gru) + request(outputs_cnn)
        for j in range(batch_size):
            if ALL_result[j] >= 2 and Y_test[i][j] == 1:
                a = a + 1
            if ALL_result[j] <= 1 and Y_test[i][j] == 0:
                a = a + 1
            # print(result[j].item(), ' ', outputs_gru[j].item(), ' ', outputs_cnn[j].item(), '  ', Y_test[i][j].item())
    print('Accuracy : ', a / len(X_test) / batch_size)


ALL_test(X_test, Y_test)

time_end = time.time()
print('总时长 ： ', time_end - time_start)
