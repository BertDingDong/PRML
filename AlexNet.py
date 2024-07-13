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
#由于本人是第一次接触pytorch平台，所以对以上的库的引用并不是很了解，只是一口气引用进来
#以下是对数据进行预处理的过程
traintransform = transforms.Compose([
    transforms.RandomResizedCrop(227),  # 随机裁剪图像
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
    transforms.Normalize(mean=[0.473, 0.4486, 0.4028], std=[0.2795, 0.2713, 0.2849])  # 归一化，这里的参数要根据对应数据集调整
])
# 注意训练集和测试集的预处理不同，训练集需要进行数据增强处理来加强学习效果，测试集只要保证规模一致即可
valtransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),  # 调整图像大小
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
    transforms.Normalize(mean=[0.473, 0.4486, 0.4028], std=[0.2795, 0.2713, 0.2849])  # 归一化
])
#参数初始化，卷积层采用Kaiming初始化，全连接层采用正态分布初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
#AlexNet的网络结构，基本参考书中的结构实现，出了输出层改成了100分类，因此不多赘述
class AlexNet(nn.Module):
    def __init__(self,inputsize,outputsize):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=3, stride=2),
                      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=3, stride=2),
                      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                      nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                      nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv6 = nn.Sequential(nn.Linear(256*6*6, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, outputsize)
                    )
    #前向传播
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.conv6(x)
        return x
#训练函数
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()#将模型进入训练状态
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        #以上是将数据放入GPU中
        pred = model(x)
        #pred是输出
        loss = loss_fn(pred, y)
        #损失函数
        #反向传播过程
        loss.backward()
        optimizer.step()
        #清空梯度，避免梯度积累
        optimizer.zero_grad()


def val(dataloader, model, loss_fn, device, scheduler):
    model.eval()
    #模型进入测试状态，模型参数不再改变
    test_loss, correct = 0, 0
    #以下是计算损失和正确率的过程
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
    #输出结果
    print(f"Test Error: \t Accuracy: {correct:>0.1f}%, Avg loss: {test_loss:>8f}")
    #检测是否需要调整学习率
    scheduler.step(test_loss)
    return test_loss, correct
#由于实验过程中尝试使用colab进行训练，colab环境中没法使用Args传参，因此采用Args类来代替命令行参数
class Args:
    data_path = r"D:\专业课\大二下\模式识别\代码\第三次作业\usps.h5"
    hidden_size = 512#这个其实没有用到
    learning_rate = 0.05#学习率
    num_epoch = 30#总训练轮数
    batch_size = 64#这个其实也没啥用，后面直接在dataloader中设置了
    interval_val = 2#间隔，每训练2轮输出一次
    interval_save = 100#这个好像也没用到

def main():
    args = Args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'#判断是否有GPU
    print(device)#这个是前期测试CUDA能不能用
    #加载数据集，因为这里是图像数据集，直接下载好在本地了
    train_set = torchvision.datasets.ImageFolder(r"D:\专业课\大二下\模式识别\数据库\mini_imagenet\mini-imagenet\mini-imagenet\train" ,transform=traintransform)
    test_set = torchvision.datasets.ImageFolder(r"D:\专业课\大二下\模式识别\数据库\mini_imagenet\mini-imagenet\mini-imagenet\val", transform=valtransform)
    #创建dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    #创建模型，inputsize其中没用上，因为按书中的结构直接写了
    model = AlexNet(inputsize=256, outputsize=100)
    #把模型放入GPU
    model.to(device)
    #优化器，SGD优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    #损失函数：交叉熵
    loss_fn = nn.CrossEntropyLoss()
    #学习率调整器，当损失上升两次时，调整学习率
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,patience=2)
    #训练过程，记录过程中的损失函数和正确率
    loss_history, acc_history = [], []
    for epoch in range(args.num_epoch):
        #输出轮数方便计时
        print(epoch)
        train(train_loader, model, loss_fn, optimizer, device)
        if (epoch+1) % args.interval_val == 0:
            test_loss, correct = val(test_loader, model, loss_fn, device, scheduler)
            loss_history.append(test_loss)
            acc_history.append(correct)
        if (epoch+1) % args.interval_save == 0:
            torch.save(model.state_dict(), 'ckpt_{}.pth'.format(epoch + 1))
    torch.save(model.state_dict(), 'ckpt_last.pth')
    #绘制损失和正确率曲线
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
