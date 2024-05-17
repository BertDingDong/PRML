#本代码对应第三部分第三大题的第二小题中的用double-fault方法来挑选模型的部分
import numpy as np
import matplotlib.pyplot as plt
import h5py
#这份代码基本是照着第一份代码来的，因此很多地方注释都是一样的，不多赘述
with h5py.File(r"D:\专业课\模式识别\代码\第四次作业\usps.h5", 'r') as hf:
        train = hf.get('train')
        x_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        x_te = test.get('data')[:]
        y_te = test.get('target')[:]

xcolumntr=np.ones((x_tr.shape[0], 1))
appendx_tr=np.c_[x_tr,xcolumntr]

xcolumnte=np.ones((x_te.shape[0], 1))
appendx_te=np.c_[x_te,xcolumnte]

netry=np.zeros((x_tr.shape[0],10))
netey=np.zeros((x_te.shape[0],10))

for i in range(x_tr.shape[0]):
    netry[i][y_tr[i]]=1
for i in range(x_te.shape[0]):
    netey[i][y_te[i]]=1

x_tr2=np.empty((0,x_tr.shape[1]+16*15+15*16))
x_te2=np.empty((0,x_te.shape[1]+16*15+15*16))

for i in range(x_tr.shape[0]):
    twoD=[]
    for j in range(16):
        for k in range(15):
            twoD.append(x_tr[i][j*16+k]*x_tr[i][j*16+k+1])
    for j in range(15):
        for k in range(16):
            twoD.append(x_tr[i][j*16+k]*x_tr[i][j*16+k+16])
    append=np.hstack((x_tr[i],twoD))
    x_tr2=np.vstack((x_tr2,append))

for i in range(x_te.shape[0]):
    twoD=[]
    for j in range(16):
        for k in range(15):
            twoD.append(x_te[i][j*16+k]*x_te[i][j*16+k+1])
    for j in range(15):
        for k in range(16):
            twoD.append(x_te[i][j*16+k]*x_te[i][j*16+k+16])
    append=np.hstack((x_te[i],twoD))
    x_te2=np.vstack((x_te2,append))

xcolumntr2=np.ones((x_tr2.shape[0], 1))
appendx_tr2=np.c_[x_tr2,xcolumntr2]

xcolumnte2=np.ones((x_te2.shape[0], 1))
appendx_te2=np.c_[x_te2,xcolumnte2]

class logistic:
    def __init__(self,x,y,learning_rate=0.01):
        self.x=x
        self.y=y
        self.w=np.zeros((x,y))
        self.learning_rate = learning_rate

    def yinx(self,x):#条件概率
        scores = np.dot(x,self.w)
        scores -= np.max(scores)
        prob = np.exp(scores)/np.sum(np.exp(scores))
        return prob

    def train(self,trainx,trainy,cycle=50):
        for time in range(cycle):
            for i in range(trainx.shape[0]):
                prob = self.yinx(trainx[i])
                for j in range(self.y):
                    if j == trainy[i]:
                        delta = prob[j] - 1
                    else:
                        delta = prob[j]
                    self.w[:,j] -= self.learning_rate * delta * trainx[i]

    def test(self,testx,testy):
        right=0
        for i in range(testx.shape[0]):
            outcome=np.dot(testx[i],self.w)
            if(np.argmax(outcome)==testy[i]):
                right+=1
        print("the right rate is",(right/testx.shape[0]))

    def vote(self,testx):
        outcome=np.zeros((testx.shape[0],10))
        for i in range(testx.shape[0]):
            onevote=np.dot(testx[i],self.w)
            outcome[i][np.argmax(onevote)]=1
        return outcome

        
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
                                  
    def test(self,testx,testy):
        out=np.zeros((testy.shape[0],1))
        Y_out=self.frontspread(testx)
        right=0
        ##下为将输出向量转化为具体结果的过程，原理就是取最大值嘛
        for i in range(Y_out.shape[0]):
            out[i]=np.argmax(Y_out[i])
        ##比较输出结果与标准结果，计算错误个数
        for i in range(len(testy)):
            if(testy[i]==out[i]):
                right+=1
        print("the right rate is",(right/len(testy)))
    
    def vote(self,testx):
        outcome=np.zeros((testx.shape[0],10))
        out=self.frontspread(testx)
        for i in range(out.shape[0]):
            outcome[i][np.argmax(out[i])]=1
        return outcome


class together:
    def __init__(self,m1,m2,m3,m4,m5):
        self.m1=m1
        self.m2=m2
        self.m3=m3
        self.m4=m4
        self.m5=m5
    
    def test(self,testx1,testx2,testy):
        out1=self.m1.vote(testx1)
        out2=self.m2.vote(testx2)
        out3=self.m3.vote(testx1)
        out4=self.m4.vote(testx1)
        out5=self.m5.vote(testx1)
        out=out1+out2+out3+out4+out5
        right=0
        for i in range(out.shape[0]):
            if(np.argmax(out[i])==testy[i]):
                right+=1
        print("the right rate is",(right/out.shape[0]))
#定义doublefoult函数，计算两个模型的双错率
def doublefoult(modle1,modle2,appendx_te,y_te,appendx_te2,num1,num2):
    #由于模型2的输入数据不一样，这里要特判一下，如果用一个字典包含一个模型和他对应的训练数据或许会更好些，但是我已经写完了
    if num1==1:
        out1=modle1.vote(appendx_te2)
    else:
        out1=modle1.vote(appendx_te)
    if num2==1:
        out2=modle2.vote(appendx_te2)
    else:
        out2=modle2.vote(appendx_te)
    df=0
    #计算双错率，就是两个都出错的情况
    for i in range(out1.shape[0]):
        if (np.argmax(out1[i])!=y_te[i]) and (np.argmax(out2[i])!=y_te[i]):
            df+=1
    return df

modle1=logistic(x_tr.shape[1]+1,10)
modle2=logistic(x_tr2.shape[1]+1,10)
modle3=net(x_tr.shape[1]+1,100,10)
modle4=net(x_tr.shape[1]+1,50,10)
modle5=net(x_tr.shape[1]+1,25,10)
modle1.train(appendx_tr,y_tr,50)
modle1.test(appendx_te,y_te)
modle2.train(appendx_tr2,y_tr,50)
modle2.test(appendx_te2,y_te)
for i in range(10000):
        ##调用类函数对module模型进行训练
        modle3.frontspread(appendx_tr)
        modle3.adjust(appendx_tr,netry,0.0001)
        modle4.frontspread(appendx_tr)
        modle4.adjust(appendx_tr,netry,0.00005)
        modle5.frontspread(appendx_tr)
        modle5.adjust(appendx_tr,netry,0.00005)
modle3.test(appendx_te,y_te)
modle4.test(appendx_te,y_te)
modle5.test(appendx_te,y_te)
#这里发现可以用一个模型列表把模型放到一块，这样后面的使用就会方便很多了
modlegather=[modle1,modle2,modle3,modle4,modle5]
#计算模型两两双错率，最大限度减少计算量
for m1 in range(len(modlegather)):
    for m2 in range(m1+1,len(modlegather)):
        print(m1,m2,doublefoult(modlegather[m1],modlegather[m2],appendx_te,y_te,appendx_te2,m1,m2)/len(y_te))