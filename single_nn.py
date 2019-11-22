#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: single_nn.py
@time: 2019/11/12 14:54
@desc:
'''

import numpy as np
import matplotlib.pyplot as plt

def getData(files):
    data_list=[]
    label_list=[]

    for index,file in enumerate(files):
        d=np.loadtxt(file)
        data_list.append(d)
        len,dim=d.shape
        for i in range(len):
            label=np.zeros([dim])
            label[index]=1
            label_list.append(label)

    labels=np.asarray(label_list)
    len,_=labels.shape
    data=np.asarray(data_list).reshape([len,-1])

    return data,labels


def sigmoid(x):
    y=1.0/(1.0+np.exp(-x))
    return y

def standardize(x):

    len,dim=x.shape
    std_x=np.zeros([len,dim+1])

    for index,i in enumerate(x):
        std_x[index,:-1]=i

    std_x[:,-1]=1

    return std_x

def  getloss(logits,label):
    loss=np.sum(np.square(logits-label))/2
    return loss

def sigmoid_de(x):

    ress= x*(1-x)
    return ress

def tanh_de(x):
    return 1-np.square(x)

def getaccurate(train_x,labels,Ws):

    count=0;
    for index,x in enumerate(train_x):
    # forward
        for idx,w in enumerate(Ws):
            if idx == len(Ws) - 1:
                layer = sigmoid(np.dot(x, w))
            else:
                layer = np.tanh(np.dot(x, w))
            x = standardize(layer.reshape([1, -1])).flatten()

        x=x[:-1]
        if np.argmax(x)==np.argmax(labels[index]):
            count+=1

    return count/len(train_x)

def trainop(train_x,train_y,Ws,yita):

    layers=[]
    layers.append(train_x)

    x=train_x

    #forward
    for index,w in enumerate(Ws):
        if index==len(Ws)-1:
            layer=sigmoid(np.dot(x,w))
        else:
            layer=np.tanh(np.dot(x,w))
        x=standardize(layer.reshape([1,-1])).flatten()
        layers.append(x)

    logits=x[:-1]

    layers[-1]=logits

    layer_num=len(Ws)
    loss=getloss(logits,train_y)

    init_de=yita*(train_y-logits)

    #backward
    for i in range(layer_num,0,-1):
        if i==layer_num:
            init_de=init_de*sigmoid_de(layers[i])
        else:
            init_de=init_de*tanh_de(layers[i][:-1])
        w_delta=np.dot(layers[i-1].reshape([-1,1]),np.asarray(init_de).reshape([1,-1]))
        init_de = np.dot(np.asarray(init_de).reshape(1, -1),Ws[i - 1][:-1,:].T )
        Ws[i-1]+=w_delta

    return Ws,loss


def train(train_x,train_y,units,yita,step=20000):
    '''
    :param train_x:数据
    :param train_y:标签
    :param units:层数
    :param yita: 学习率
    :return:
    '''
    train_x=standardize(train_x)
    length,dim=train_x.shape
    _,classnum=train_y.shape

    W_li=[]
    layers=[]
    #前向传播
    h=dim
    for i in range(len(units)):
        u = units[i]
        w=np.random.random((h,u))
        W_li.append(w)
        h=u+1

    w=np.random.random((h,classnum))
    W_li.append(w)

    min_loss=1000
    losses=[]
    steps=[]
    for i in range(step):

        for j in range(length):
            W_li,loss=trainop(train_x[j],train_y[j],W_li,yita)


        losses.append(loss)
        steps.append(i)

        print(loss)

    accuracy=getaccurate(train_x,train_y,W_li)

    print(accuracy)

    plt.plot(steps,losses)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("single sample update")
    plt.show()





if __name__=='__main__':
    filenames=['w1.txt','w2.txt','w3.txt']
    train_x,train_y=getData(filenames)

    #隐含层节点数目,三层前向传播网络：输入层+一层隐含层+输出层
    units=[5]
    train(train_x,train_y,units,0.01)

