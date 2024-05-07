import numpy as np
import matplotlib.pyplot as plt
import h5py
##本代码对应第4大题
##下为平台指示的读取数据过程
with h5py.File(r"D:\专业课\模式识别\代码\第三次作业\usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
##三层神经网络类跟第三大题第一小题是一样的，就不多赘述了，想看注释参考hw31.py
class net:
    def __init__(self,inputsize,unseensize,outputsize):
        self.inputsize=inputsize
        self.unseensize=unseensize
        self.outputsize=outputsize
        self.w1=np.random.uniform(low=-1,high=1,size=(inputsize,unseensize))
        self.w2=np.random.uniform(low=-1,high=1,size=(unseensize,outputsize))

    def frontspread(self,x):
        self.x1=np.dot(x,self.w1)
        self.y1=self.sigmoid(self.x1)
        self.x2=np.dot(self.y1,self.w2)
        self.y2=self.sigmoid(self.x2)
        return self.y2
    
    def adjust(self,x,t,rate):
        loss=np.sum((self.y2-t)**2)/2
        self.det1=np.multiply(np.multiply((self.y2-t),self.y2),(np.ones(np.shape(self.y2)))-self.y2)
        dw2=np.dot(self.y1.T,self.det1)
        self.det2=np.multiply(np.multiply(np.dot(self.det1,self.w2.T),self.y1),(np.ones(np.shape(self.y1)))-self.y1)
        dw1=np.dot(x.T,self.det2)
        self.w2-=dw2*rate
        self.w1-=dw1*rate
        return loss

    def sigmoid(self,x):
        return 1/(1+np.exp((-1)*x))
    
    def reset(self):
        self.w1=np.random.uniform(low=-1,high=1,size=(self.inputsize,self.unseensize))
        self.w2=np.random.uniform(low=-1,high=1,size=(self.unseensize,self.outputsize))
##经测试隐层256个节点会很慢，16个节点效果不大好，最后选取100个节点作为最终结果
module=net(257,100,10)
##下为增广向量过程，只需要引入常数项还是很通俗易懂的
ones=np.ones((X_tr.shape[0],1))
broadxtr=np.hstack((X_tr,ones))
ones=np.ones((X_te.shape[0],1))
broadxte=np.hstack((X_te,ones))
##将提供数据中的数据转化为标准的向量形式
Y_tr=np.zeros((y_tr.shape[0],10))
for i in range(len(y_tr)):
    Y_tr[i][y_tr[i]]=1
Y_te=np.zeros((y_te.shape[0],10))
for i in range(len(y_te)):
    Y_te[i][y_te[i]]=1
##下为记录数组，用于作图
x=[]
y=[]
##由于每次跑这个程序实在太慢，因此我只跑了10次，可能结果不是很精确
for k in range(10):
    ##下为训练过程
    learningrate=k/10
    for i in range(1000):
        module.frontspread(broadxtr)
        module.adjust(broadxtr,Y_tr,learningrate)
    ##下为测试过程，首先将增广的测试x向量加入传播过程得到输出向量
    out=np.zeros((y_te.shape[0],1))
    Y_out=module.frontspread(broadxte)
    wrong=0
    ##这个reset在这好像不是很美观，可以拿到后面去，无所谓了
    module.reset()
    ##下为将输出向量转化为具体结果的过程，原理就是取最大值嘛
    for j in range(Y_out.shape[0]):
        max=Y_out[j][0]-1
        for i in range(Y_out.shape[1]):
            if(Y_out[j][i]>max):
                max=Y_out[j][i]
                out[j]=i
    ##比较输出结果与标准结果，计算错误个数
    for i in range(len(y_te)):
        if(y_te[i]!=out[i]):
            wrong+=1
    x.append(learningrate)
    y.append(wrong)

plt.plot(x,y,'blue')
plt.xlim(0,1)
plt.xlabel("learningrate")
plt.ylabel("wrong")
plt.show()