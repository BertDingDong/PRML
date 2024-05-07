import numpy as np
import matplotlib.pyplot as plt
##本代码对应第三大题第一小题
##下为对数据的初始化
x=np.array([[0.68,1.34],[0.93,0.89],[0.9,1.66],[1.08,0.65],[1.26,0.57],
           [0.91,1.51],[1.5,1.26],[1.26,1.31],[0.92,1.26],[1.04,0.99],
           [-0.99,-1.54],[-1.16,-1.23],[-0.77,-1.01],[-1.3,-1.28],[-1.27,-0.96],
           [-0.99,-1.13],[-0.94,-0.93],[-1.21,-0.99],[-1.71,-0.82],[-0.99,-0.92],
           [0.99,-0.6],[1.03,-1.08],[1.56,-0.98],[1.26,-1.16],[1.14,-0.8],
           [0.9,-1.25],[1.18,-0.7],[1.3,-0.9],[1.42,-0.82],[0.54,-1.2],
           [-0.78,0.68],[-1.02,1.29],[-0.84,1.12],[-0.73,1.06],[-0.54,1.0],
           [-1.14,0.58],[1.09,0.51],[-1.12,1.33],[-0.14,0.82],[-1.45,1.31]])
t=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
           [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
           [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
           [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
##建立三层神经网络对象
class net:
    ##初始化：将节点数转化为类实例的特征，方便后续在类实例中调用
    def __init__(self,inputsize,unseensize,outputsize):
        self.inputsize=inputsize
        self.unseensize=unseensize
        self.outputsize=outputsize
        ##初始化权值向量，这里直接将权值向量合并到一个矩阵中
        self.w1=np.random.uniform(low=-1,high=1,size=(inputsize,unseensize))
        self.w2=np.random.uniform(low=-1,high=1,size=(unseensize,outputsize))
    ##前向传播过程：返回值为最后的输出矩阵
    def frontspread(self,x):
        self.x1=np.dot(x,self.w1)
        self.y1=self.sigmoid(self.x1)
        self.x2=np.dot(self.y1,self.w2)
        self.y2=self.sigmoid(self.x2)
        return self.y2
    ##反向传播过程，利用前向传播过程中记录下的变量进行反向传播，基本是按照课件中的过程来的，虽然经过一堆乱七八糟的矩阵运算我已经不确定是不是很严格遵守过程了
    ##以下过程大量采用矩阵简化向量操作，本人撰写过程中由于线代水平不高，基本只能通过矩阵乘法规则来保证需要进行操作的矩阵的行秩和列秩相等来推测下一步采取的行动
    ##具体为什么这个是对的我也不是很懂，但是从最后运行结果上来看应该是对的
    def adjust(self,x,t,rate):
        loss=np.sum((self.y2-t)**2)/2
        self.det1=np.multiply(np.multiply((self.y2-t),self.y2),(np.ones(np.shape(self.y2)))-self.y2)
        dw2=np.dot(self.y1.T,self.det1)
        self.det2=np.multiply(np.multiply(np.dot(self.det1,self.w2.T),self.y1),(np.ones(np.shape(self.y1)))-self.y1)
        dw1=np.dot(x.T,self.det2)
        self.w2-=dw2*rate
        self.w1-=dw1*rate
        return loss
    ##sigmoid函数
    def sigmoid(self,x):
        return 1/(1+np.exp((-1)*x))
    ##重置模型，方便多次实验
    def reset(self):
        self.w1=np.random.uniform(low=-1,high=1,size=(self.inputsize,self.unseensize))
        self.w2=np.random.uniform(low=-1,high=1,size=(self.unseensize,self.outputsize))
##类实例化
module=net(3,5,2)
##将向量变成增广向量
ones=np.ones((x.shape[0],1))
broadx=np.hstack((x,ones))
##学习率
learningrate=1
##以下两个数组用于存储过程中的学习率与损失函数值，方便后续绘图
x=[]
y=[]
for j in range(1,100):
    ##初始化学习率
    learningrate=j/100
    x.append(learningrate)
    for i in range(10000):
        ##调用类函数对module模型进行训练
        module.frontspread(broadx)
        loss=module.adjust(broadx,t,learningrate)
        if(i%100==0):
            print(loss)
    ##记录损失函数并重启模型
    y.append(loss)
    module.reset()
##制图
plt.plot(x,y,'blue')
plt.xlim(0,1)
plt.xlabel("learningrate")
plt.ylabel("loss")
plt.show()