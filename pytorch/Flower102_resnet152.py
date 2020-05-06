import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import torchvision as tv
import torch as t
from torchvision.transforms import ToPILImage
import pandas as pd
from torchvision import models as torchmd

learning_rate = 0.001
root = 'D:\\AI\\AI研习社\\102种鲜花分类\\54_data\\'


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        count = 0
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            if count == 0:  # 第一列是列名，过滤掉
                count += 1
                continue
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split(',')  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgpath = root+'train\\'+words[0]
            # imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            imgs.append((imgpath, int(words[1])))
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


class TestDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  # 初始化一些需要传入的参数
        super(TestDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        count = 0
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            if count == 0:  # 第一列是列名，过滤掉
                count += 1
                continue
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split(',')  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgpath = root+'test\\'+words[0]
            # imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            imgs.append((imgpath, int(words[1])))
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_data = MyDataset(txt=root + 'train.csv', transform=data_transforms['train'])
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=3)
    print('num_of_trainData:', len(train_loader))

    test_data = MyDataset(txt=root + 'test.csv', transform=data_transforms['val'])
    test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=3)
    print('num_of_testData:', len(test_loader))

    predict_data = TestDataset(txt=root + 'predict.csv', transform=data_transforms['val'])
    predict_loader = DataLoader(dataset=predict_data, batch_size=1, shuffle=False, num_workers=1)
    print('num_of_predictData:', len(predict_loader))

    net = torchmd.resnet152(True)
    net.classifier = nn.Linear(2048, 102)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD（螺旋）随机梯度下降法，

    """"""
    for epoch in range(10):  # 设置训练的迭代次数
        running_loss = 0.0
    # 在测试集中迭代数据
        for i, data in enumerate(train_loader, 0):  # enumerate枚举数据并从下标0开始

            # 输入数据
            # 读取数据的数据内容和标签
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零，也就是把loss关于weight的导数变成0.
            optimizer.zero_grad()

            # forward+backward
            # 得到网络的输出
            outputs = net(inputs)

            # 计算损失值，将输出的outputs和原来导入的labels作为loss函数的输入就可以得到损失了：
            loss = criterion(outputs, labels)  # output 和 labels的交叉熵损失
            # 计算得到loss后就要回传损失。
            # loss.backward()
            # loss.backward(),有时候，我们并不想求所有Variable的梯度。那就要考虑如何在Backward过程中排除子图（ie.排除没必要的梯度计算）。
            # 可以通过Variable的两个参数（requires_grad和volatile）与电路的连接有关啊这样记住吧哈哈哈

            #lossfunc = nn.CrossEntropyLoss()
            #loss = lossfunc(outputs, labels)
            loss.backward()
            # 更新参数

            # 回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，optimizer.step()就是用来更新参数的。optimizer.step()后，
            # 你就可以从optimizer.param_groups[0][‘params’]里面看到各个层的梯度和权值信息。
            optimizer.step()  # 利用计算的得到的梯度对参数进行更新

            # 打印log信息
            running_loss += loss.item()  # #用于从tensor中获取python数字
            if i % 200 == 199:  # 每200个batch打印一次训练状态
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        print('Finished Training')
        # 此处仅仅训练了两个epoch（遍历完一遍数据集），我们来看看网络有没有效果

    """    """
    print('Begain Testing')
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    with t.no_grad():
        for data in test_loader:
            # 读取数据的数据内容及标签
            images, labels = data
            # 得到网络的输出
            outputs = net(images)
            # 得到预测值
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            # 累积计算预测正确的数据集的大小
            correct += (predicted == labels).sum()  # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
    print('Accuracy of the network on the  test images: %d %%' % (100 * correct // total))


    print('Begain Predict')
    preresult = []
    with t.no_grad():
        for data in predict_loader:
            # 读取数据的数据内容及标签
            images, labels = data
            # 得到网络的输出
            outputs = net(images)
            # 得到预测值
            _, predicted = t.max(outputs.data, 1)
            # 累积计算预测正确的数据集的大小
            preresult.append([images, predicted])

    column = ['filename', 'label']
    resultfile = pd.DataFrame(columns=column, data=preresult)
    resultfile.to_csv('D:\\AI\\AI研习社\\102种鲜花分类\\54_data\\preresult2.csv')
    print('save End')


if __name__ == '__main__':
    main()
