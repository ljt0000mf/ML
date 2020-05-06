# _*_ coding:utf-8 _*_
import os
import matplotlib.pyplot as plt
import csv
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sb
import time


# 设置数据目录
train_dir = 'train/'
valid_dir = 'valid/'
test_dir = 'test/'

train_txt = 'train.txt'
valid_txt = 'val.txt'
test_txt = 'test.txt'

# 进行图像预处理参数设置
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])


# 自定义读写数据方式1
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, flag='', target_transform=None):
        """
        tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.flag = flag
        fh = open(txt_path, 'r')  # 读取文件
        imgs = []  # 用来存储路径与标签
        for line in fh:
            line = line.strip('\n')
            if self.flag == 'train' or self.flag == 'val':
                words = line.split(' ')
                imgs.append((words[0], int(words[1])))  # 路径和标签添加到列表中
            elif self.flag == 'test':
                imgs.append(line)
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        if self.flag == 'train' or self.flag == 'val':
            fn, label = self.imgs[index]  # 通过index索引返回一个图像路径fn 与 标签label
            img = Image.open(fn).convert('RGB')  # 把图像转成RGB
            if self.transform is not None:
                img = self.transform(img)
            return img, label  # 这就返回一个样本
        elif self.flag == 'test':
            fn = self.imgs[index]  # 通过index索引返回一个图像路径fn
            img = Image.open(fn).convert('RGB')  # 把图像转成RGB
            if self.transform is not None:
                img = self.transform(img)
            return img  # 这就返回一个样本

    def __len__(self):
        return len(self.imgs)  # 返回长度，index就会自动的指导读取多少


trainObj = MyDataset(train_txt, train_transforms, 'train')
valObj = MyDataset(valid_txt, test_valid_transforms, 'val')
testObj = MyDataset(test_txt, test_valid_transforms, 'test')

# 加载图像
trainloader = torch.utils.data.DataLoader(trainObj, batch_size=3, shuffle=True)
validloader = torch.utils.data.DataLoader(valObj, batch_size=3, shuffle=True)
testloader = torch.utils.data.DataLoader(testObj, batch_size=1, shuffle=False)

# 构建分类器网络
# 使用ResNet152
# 使用resnet152的网络结构，最后一层全连接重写输出102
start_epoch = 0


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # 可以选择冻结卷积层
        # for p in self.parameters():
        #     p.requires_grad = False
        self.fc = nn.Linear(in_features=2048, out_features=102)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


resnet152 = models.resnet152(pretrained=True)
net = Net(resnet152)
net = net.to(device)

# 参数设定
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9,
                            weight_decay=5e-4)

best_acc = 0


# 训练模型
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = 20
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total


# 验证模型
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    if True:
        for idx, (inputs, labels) in enumerate(validloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

            print("Testing ...")
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(validloader), end - start, test_loss / len(validloader), correct, total,
                100. * correct / total
            ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')
    return test_loss / len(validloader), 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


# 在比赛的测试数据集上测试效果
def get_result():
    # 加载模型
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict, strict=False)
    net.eval()
    net.to(device)

    test_pic_name_list = []
    predicted_label_list = []
    for idx, inputs in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        class_index = outputs.max(dim=1)[1]
        class_index = class_index.cuda().data.cpu().numpy()[0]
        predicted_label_list.append(str(class_index))

    for i in range(1434):
        test_pic_name_list.append(str(i))

    # DataFrame_test = pd.DataFrame({'name':test_pic_name_list,'label':predicted_label_list})
    # DataFrame_test = pd.DataFrame({test_pic_name_list,predicted_label_list})
    # DataFrame_test.to_csv('key.csv',index = None, encoding = 'utf-8')
    with open('key.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(test_pic_name_list, predicted_label_list))


def main():
    for epoch in range(start_epoch, start_epoch+40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%20==0:
            lr_decay()
    # get_result()


if __name__ == '__main__':
    main()