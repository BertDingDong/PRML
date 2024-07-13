#本代码大部分基于助教实验课提供的代码完成，在此感谢张瑞松老师提供的代码
import argparse
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
#数据预处理跟AlexNet的一样，直接看它的就行了
traintransform = transforms.Compose([
    transforms.RandomResizedCrop(227),  # 随机裁剪图像
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
    transforms.Normalize(mean=[0.473, 0.4486, 0.4028], std=[0.2795, 0.2713, 0.2849])  # 归一化
])
valtransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),  # 调整图像大小
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
    transforms.Normalize(mean=[0.473, 0.4486, 0.4028], std=[0.2795, 0.2713, 0.2849])  # 归一化
])
#初始化也是一样的，之前试过对批归一化层进行初始化，但是效果不好就删了
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
#ResNet使用的Bottleneck结构，基本按照相关资料中的实现来的，就不多赘述了
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
#残差神经网络模型，也是按照相关资料实现的
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    #搭建层函数，因为残差神经网络其实可以分成四个部分，每个部分里面的卷积层有一定的差距，因此用一个函数以及他的参数来表示
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    #前向传播函数
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
#训练函数也一样
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#测试函数也一样
def val(dataloader, model, loss_fn, device, scheduler):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    correct *= 100
    print(f"Test Error: \t Accuracy: {correct:>0.1f}%, Avg loss: {test_loss:>8f}")
    scheduler.step(test_loss)
    return test_loss, correct
#参数也一样
class Args:
    data_path = r"D:\专业课\大二下\模式识别\代码\第三次作业\usps.h5"
    hidden_size = 512
    learning_rate = 0.005
    num_epoch = 30
    batch_size = 16
    interval_val = 2
    interval_save = 100
#这里采用的是50层的残差神经网络，也是按照资料中的配比实现的
def resnet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def main():
    #以下部分基本跟AlexNet的一样，没加注释的就默认跟AlexNet一样
    args = Args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    train_set = torchvision.datasets.ImageFolder(r"D:\专业课\大二下\模式识别\数据库\mini_imagenet\mini-imagenet\mini-imagenet\train" ,transform=traintransform)
    test_set = torchvision.datasets.ImageFolder(r"D:\专业课\大二下\模式识别\数据库\mini_imagenet\mini-imagenet\mini-imagenet\val", transform=valtransform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    model = resnet50(num_classes=100)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,patience=2)
    loss_history, acc_history = [], []
    for epoch in range(args.num_epoch):
        print(epoch)
        train(train_loader, model, loss_fn, optimizer, device)
        if (epoch+1) % args.interval_val == 0:
            test_loss, correct = val(test_loader, model, loss_fn, device, scheduler)
            loss_history.append(test_loss)
            acc_history.append(correct)
        if (epoch+1) % args.interval_save == 0:
            torch.save(model.state_dict(), 'ckpt_{}.pth'.format(epoch + 1))
    torch.save(model.state_dict(), 'ckpt_last.pth')

    plt.plot(loss_history)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(acc_history)
    plt.title('Acc Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    main()
