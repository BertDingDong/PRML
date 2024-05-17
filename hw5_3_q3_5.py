#本代码对应第三部分第三大题的第二小题中的stacking方法
import numpy as np
import matplotlib.pyplot as plt
import h5py
#本代码也是从第二份代码里改过来的，大部分地方是一样的，有不一样的地方我会标明注释，没标的就默认跟第二份一样吧
with h5py.File(r"D:\专业课\模式识别\代码\第四次作业\usps.h5", 'r') as hf:
        train = hf.get('train')
        x_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        x_te = test.get('data')[:]
        y_te = test.get('target')[:]

x_train=x_tr[:int(0.8*x_tr.shape[0])]
y_train=y_tr[:int(0.8*x_tr.shape[0])]
x_exam=x_tr[int(0.8*x_tr.shape[0]):]
y_exam=y_tr[int(0.8*x_tr.shape[0]):]

xcolumntr=np.ones((x_train.shape[0], 1))
appendx_tr=np.c_[x_train,xcolumntr]

xcolumnex=np.ones((x_exam.shape[0], 1))
appendx_ex=np.c_[x_exam,xcolumnex]

xcolumnte=np.ones((x_te.shape[0], 1))
appendx_te=np.c_[x_te,xcolumnte]

netry=np.zeros((y_train.shape[0],10))
netex=np.zeros((y_exam.shape[0],10))
netey=np.zeros((x_te.shape[0],10))

for i in range(netry.shape[0]):
    netry[i][y_train[i]]=1
for i in range(netey.shape[0]):
    netey[i][y_te[i]]=1
for i in range(netex.shape[0]):
    netex[i][y_exam[i]]=1

x_tr2=np.empty((0,x_train.shape[1]+16*15+15*16))
x_te2=np.empty((0,x_te.shape[1]+16*15+15*16))
x_ex2=np.empty((0,x_exam.shape[1]+16*15+15*16))

for i in range(x_train.shape[0]):
    twoD=[]
    for j in range(16):
        for k in range(15):
            twoD.append(x_train[i][j*16+k]*x_train[i][j*16+k+1])
    for j in range(15):
        for k in range(16):
            twoD.append(x_train[i][j*16+k]*x_train[i][j*16+k+16])
    append=np.hstack((x_train[i],twoD))
    x_tr2=np.vstack((x_tr2,append))

for i in range(x_exam.shape[0]):
    twoD=[]
    for j in range(16):
        for k in range(15):
            twoD.append(x_exam[i][j*16+k]*x_exam[i][j*16+k+1])
    for j in range(15):
        for k in range(16):
            twoD.append(x_exam[i][j*16+k]*x_exam[i][j*16+k+16])
    append=np.hstack((x_exam[i],twoD))
    x_ex2=np.vstack((x_ex2,append))

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

xcolumnex2=np.ones((x_ex2.shape[0], 1))
appendx_ex2=np.c_[x_ex2,xcolumnex2]

xcolumnte2=np.ones((x_te2.shape[0], 1))
appendx_te2=np.c_[x_te2,xcolumnte2]

class logistic:
    def __init__(self, x, y, learning_rate=0.01):
        self.x = x
        self.y = y
        self.w = np.zeros((x, y))
        self.learning_rate = learning_rate

    def yinx(self, x):  # 条件概率
        scores = np.dot(x, self.w)
        scores -= np.max(scores)
        prob = np.exp(scores) / np.sum(np.exp(scores))
        return prob

    def train(self, trainx, trainy, cycle=50):
        for time in range(cycle):
            for i in range(trainx.shape[0]):
                prob = self.yinx(trainx[i])
                for j in range(self.y):
                    if j == trainy[i]:
                        delta = prob[j] - 1
                    else:
                        delta = prob[j]
                    self.w[:, j] -= self.learning_rate * delta * trainx[i]

    def test(self, testx, testy):
        right = 0
        for i in range(testx.shape[0]):
            outcome = np.dot(testx[i], self.w)
            if np.argmax(outcome) == testy[i]:
                right += 1
        print("the right rate is", (right / testx.shape[0]))

    def vote(self, testx):
        outcome = np.zeros((testx.shape[0], 10))
        for i in range(testx.shape[0]):
            onevote = np.dot(testx[i], self.w)
            outcome[i][np.argmax(onevote)] = 1
        return outcome

    def predict(self, testx):
        return np.dot(testx, self.w)

        
class net:
    def __init__(self, inputsize, unseensize, outputsize):
        self.inputsize = inputsize
        self.unseensize = unseensize
        self.outputsize = outputsize
        self.w1 = np.random.uniform(low=-1, high=1, size=(inputsize, unseensize))
        self.w2 = np.random.uniform(low=-1, high=1, size=(unseensize, outputsize))

    def frontspread(self, x):
        self.x1 = np.dot(x, self.w1)
        self.y1 = self.sigmoid(self.x1)
        self.x2 = np.dot(self.y1, self.w2)
        self.y2 = self.sigmoid(self.x2)
        return self.y2

    def adjust(self, x, t, rate):
        loss = np.sum((self.y2 - t) ** 2) / 2
        self.det1 = np.multiply(np.multiply((self.y2 - t), self.y2), (np.ones(np.shape(self.y2))) - self.y2)
        dw2 = np.dot(self.y1.T, self.det1)
        self.det2 = np.multiply(np.multiply(np.dot(self.det1, self.w2.T), self.y1), (np.ones(np.shape(self.y1))) - self.y1)
        dw1 = np.dot(x.T, self.det2)
        self.w2 -= dw2 * rate
        self.w1 -= dw1 * rate
        return loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reset(self):
        self.w1 = np.random.uniform(low=-1, high=1, size=(self.inputsize, self.unseensize))
        self.w2 = np.random.uniform(low=-1, high=1, size=(self.unseensize, self.outputsize))

    def test(self, testx, testy):
        out = np.zeros((testy.shape[0], 1))
        Y_out = self.frontspread(testx)
        right = 0
        for i in range(Y_out.shape[0]):
            out[i] = np.argmax(Y_out[i])
        for i in range(len(testy)):
            if testy[i] == out[i]:
                right += 1
        print("the right rate is", (right / len(testy)))

    def vote(self, testx):
        outcome = np.zeros((testx.shape[0], 10))
        out = self.frontspread(testx)
        for i in range(out.shape[0]):
            outcome[i][np.argmax(out[i])] = 1
        return outcome

    def predict(self, testx):
        return self.frontspread(testx)


#这里也是改了一下让他变成3个模型的集成，这里罗杰斯特回归模型的输入31之前忘记改了，白跑了一遍，更加说明避免代码中出现魔数的重要性
class together:
    def __init__(self,m1,m2,m3):
        self.m1=m1
        self.m2=m2
        self.m3=m3
        self.logistic=logistic(31,10)
    
    def train(self,trainx1,trainx2,trainy,cycle=50):
        importx=self.vote(trainx1,trainx2)
        self.logistic.train(importx,trainy,cycle)

    def vote(self,importx1,importx2):
        self.out1=self.m1.predict(importx2)
        self.out2=self.m2.predict(importx1)
        self.out3=self.m3.predict(importx1)
        importx = np.hstack((self.out1, self.out2, self.out3))
        ones = np.ones((importx.shape[0], 1))
        importx = np.hstack((importx, ones))

        return importx

    def test(self,testx1,testx2,testy):
        importx=self.vote(testx1,testx2)
        self.logistic.test(importx,testy)
#以下部分除了删掉了1,4两个模型的戏份外其它都跟第二份代码一样
modle2=logistic(x_tr2.shape[1]+1,10)
modle3=net(x_tr.shape[1]+1,100,10)
modle5=net(x_tr.shape[1]+1,25,10)
modle2.train(appendx_tr2,y_train,50)
modle2.test(appendx_te2,y_te)
for i in range(10000):
        ##调用类函数对module模型进行训练
        modle3.frontspread(appendx_tr)
        modle3.adjust(appendx_tr,netry,0.0001)
        modle5.frontspread(appendx_tr)
        modle5.adjust(appendx_tr,netry,0.00005)
modle3.test(appendx_te,y_te)
modle5.test(appendx_te,y_te)
gether_modle=together(modle2,modle3,modle5)
for i in range(1,50):
    gether_modle.train(appendx_ex,appendx_ex2,y_exam,i)
    gether_modle.test(appendx_te,appendx_te2,y_te)