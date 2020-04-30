#加载使用的各种包
import torchvision as tv
import torch as t
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import ToPILImage


def main():
    #扩充数据集防止过拟合
    show = ToPILImage() #可以把Tensor转换成Image,方便可视化
    transform = transforms.Compose([  #transforms.Compose就是将对图像处理的方法集中起来
        transforms.RandomHorizontalFlip(),#水平翻转
        transforms.RandomCrop((32, 32), padding=4),
        transforms.ToTensor(),#转为Tensor
        #在做数据归一化之前必须要把PIL Image转成Tensor，而其他resize或crop操作则不需要。
         transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5)),#归一化
        ])
    #训练集
    #训练集数据的下载
    trainset = tv.datasets.CIFAR10(
    root='D:\\AI\\practise',#设置数据集的根目录  D:\AI\practise
        train=True,#训练集所以是True
        download=True,
        transform=transform
    )
    #训练集数据的加载方式
    trainloader = t.utils.data.DataLoader(
        trainset,#设置为训练集
        batch_size=4,#设置每个batch有4个样本数据
        shuffle=True,#设置对每个epoch将数据集打乱
        num_workers=2#设置使用2个子进程用来加载数据
    )
    #测试集
    #测试集下载
    testset = tv.datasets.CIFAR10(
    'D:\\AI\\practise',
    train=False,
    download=True,
    transform=transform#用了之前定义的transform
    )
    #测试集加载方式
    testloader = t.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=2)
    #数据集10个类的定义
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD（螺旋）随机梯度下降法，

    for epoch in range(10):  # 设置训练的迭代次数
        running_loss = 0.0
        # 在测试集中迭代数据
        for i, data in enumerate(trainloader, 0):  # enumerate枚举数据并从下标0开始

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

            loss.backward()
            # loss.backward(),有时候，我们并不想求所有Variable的梯度。那就要考虑如何在Backward过程中排除子图（ie.排除没必要的梯度计算）。
            # 可以通过Variable的两个参数（requires_grad和volatile）与电路的连接有关啊这样记住吧哈哈哈

            # 更新参数

            # 回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，optimizer.step()就是用来更新参数的。optimizer.step()后，
            # 你就可以从optimizer.param_groups[0][‘params’]里面看到各个层的梯度和权值信息。
            optimizer.step()  # 利用计算的得到的梯度对参数进行更新

            # 打印log信息
            running_loss += loss.item()  # #用于从tensor中获取python数字
            if i % 2000 == 1999:  # 每2000个batch打印一次训练状态
                print('[%d, %5d] loss: %.3f' \
                      % (epoch + 1, i + 1, running_loss / 2000))

                running_loss = 0.0
        print('Finished Training')
        # 此处仅仅训练了两个epoch（遍历完一遍数据集），我们来看看网络有没有效果


class Net(nn.Module):
    def __init__(self):
         # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()# 调用父类的初始化函数 #super是比较两个类谁更高，谁是父类，然后执行父类的函数
        self.conv1 = nn.Conv2d(3, 6, 5) # 卷积层3表示输入图片为RGB型，6表示输出通道数，5表示卷积核为5*5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120) # 全连接层输入是16*5*5的，输出是120的
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 定义输出为10，因为10各类
    def forward(self, x):
        x =self.pool(F.relu(self.conv1(x)))  # 输入x经过卷积conv1之后，经过激活函数ReLU，然后更新到x。
        x =self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) #用来将x展平成16 * 5 * 5，然后就可以进行下面的全连接层操作
        x = F.relu(self.fc1(x))# 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))# 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x =self.fc3(x) # 输入x经过全连接3，然后更新x
        return x


if __name__ == '__main__':
    main()

