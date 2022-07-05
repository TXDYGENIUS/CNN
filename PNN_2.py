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
device = torch.device('cuda:0')

file_anomalous = 'F:\pythonProject\MachineLearningCVE\PNN测试.csv'
file_normal = 'F:\pythonProject\MachineLearningCVE\yichang.csv'
file_2018 = 'F:\pythonProject\MachineLearningCVE\Sunday2.csv'
label_class = [0, 1]  # 所有的标签类别


# https://blog.csdn.net/Luqiang_Shi/article/details/84973340

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
        out[j] = out[j].view([-1, 78])
    outt = out[0]
    for k in range(len(out) - 1):
        outt = torch.cat((outt, out[k + 1]), dim=0)
    laber = torch.tensor(laber)
    return outt, laber


def get_data(file):
    X, Y = open_file(file)
    data_dataset = TensorDataset(X, Y)  # 数据和标签打包
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.999, random_state=666)

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = get_data(file_anomalous)


# X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


class PNN(nn.Module):
    def __init__(self, train_data, train_label, test_data):
        super(PNN, self).__init__()
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.sigma = 0.03

    def Normalization(self, data):
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

    def distance(self, X, Y):
        # 计算两个样本之间的距离
        return torch.sqrt(torch.sum(torch.square(X - Y)))

    def distance_mat(self, Nor_trainX, Nor_testX):
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
                Euclidean_D[i, j] = self.distance(Nor_testX[i, :], Nor_trainX[j, :])  # [i,:]切片，取第i行所有元素
                # [i,j]为第i个测试集test 与 第j个训练集train 的欧氏距离
        return Euclidean_D

    def Gauss(self, Euclidean_D, sigma):
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

    def Prob_mat(self, Gauss_mat, label):
        '''测试样本属于各类的概率和矩阵
        input:Gauss_mat(mat):Gauss矩阵
              labelX(list):训练样本的标签矩阵
        output:Prob_mat(mat):测试样本属于各类的概率矩阵
        '''
        # 求概率和矩阵
        p, m = Gauss_mat.size()
        Prob = torch.zeros((p, len(label_class)))  # label的类别数 [0,1]= 2
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

    def class_results(self, Prob):
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

    def forward(self, train_data, train_label, test_data):
        X_train = self.Normalization(train_data)
        X_test = self.Normalization(test_data)
        mat = self.distance_mat(X_train, X_test)
        gauss = self.Gauss(mat, self.sigma)
        prob = self.Prob_mat(gauss, train_label)
        result = self.class_results(prob)
        return result


model = PNN(X_train, Y_train, X_test)


# model = model.to(device)


def test_PNN(model, Y_test):
    a = 0
    result = model(X_train, Y_train, X_test)

    for i in range(len(result)):
        if Y_test[i] == result[i]:
            a = a + 1

    print(a / len(result), '  sigma = ', model.sigma)


test_PNN(model, Y_test)

time_end = time.time()
print('time = ', time_end - time_start, 's')
