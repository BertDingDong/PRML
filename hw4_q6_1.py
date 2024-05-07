import numpy as np
import matplotlib.pyplot as plt
import h5py
##本代码对应第六大题第一小题
##下为提取数据过程，直接搬的上次作业的代码
with h5py.File(r"D:\专业课\模式识别\代码\第四次作业\usps.h5", 'r') as hf:
        train = hf.get('train')
        x_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        x_te = test.get('data')[:]
        y_te = test.get('target')[:]
#输入一些空矩阵，用于后续存储我们需要的向量
train1 = np.empty((0, 256))
train2 = np.empty((0, 256))

test1 = np.empty((0, 256))
test2 = np.empty((0, 256))
#把y中值为0的点对应的x向量归入第一类，值为1的点归入第二类，也就是我们要分类的对象
for i in range(len(y_tr)):
    if y_tr[i] == 0:
        train1 = np.vstack((train1, x_tr[i]))
    if y_tr[i] == 1:
        train2 = np.vstack((train2, x_tr[i]))

for i in range(len(y_te)):
    if y_te[i] == 0:
        test1 = np.vstack((test1, x_te[i]))
    if y_te[i] == 1:
        test2 = np.vstack((test2, x_te[i]))
#下面代码与第五大题的第二小问的第二种情况相似，不多赘述，详情可见hw4_q5_3
u1=np.mean(train1,axis=0)
u2=np.mean(train2,axis=0)

train=np.concatenate((train1,train2),axis=0)
sigma=np.cov(train,rowvar=False)+0.00001*np.eye(train.shape[1])

def possible(x,u,sigma):
    return 2*np.dot(np.dot(u,np.linalg.inv(sigma)),x.T)-np.dot(np.dot(u,np.linalg.inv(sigma)),u.T)

wrong=0
for i in range(len(test1)):
    test1possible=possible(test1[i],u1,sigma)
    if(test1possible<=possible(test1[i],u2,sigma)):
        wrong+=1

for i in range(len(test2)):
    test2possible=possible(test2[i],u2,sigma)
    if(test2possible<=possible(test2[i],u1,sigma)):
        wrong+=1

print("the right rate of bayes is:%f" % (1.0-wrong/(len(test1)+len(test2))))
#下为基于fisher判别准则的线性分类器，也是基本依照第二次作业的代码来实现的，详情可见hw2_q3_2
s1=np.zeros((256,256))
s2=np.zeros((256,256))

for i in range(train1.shape[0]):
    s1+=np.dot((train1[i]-u1).reshape(-1,1),(train1[i]-u1).reshape(1,-1))
for i in range(train2.shape[0]):
    s2+=np.dot((train2[i]-u2).reshape(-1,1),(train2[i]-u2).reshape(1,-1))

s=np.dot((u1-u2).T,(u1-u2))
#这里可能是由于样本量过大，导致sw如果不进行正则化的话，直接取s1+s2会出现sw变成奇异矩阵的情况，因此给它加上一点扰动进行正则化，确保后续操作能够正常进行
sw=s1+s2+0.000001*np.eye(256)
w=np.dot(np.linalg.inv(sw),(u1-u2))
w0=(np.dot(w,u1)+np.dot(w,u2))*(-0.5)
wrong=0
for i in range(test1.shape[0]):#计算错分的数量
        if(np.dot(w,test1[i])+w0<0):
            wrong+=1
for i in range(test2.shape[0]):#计算错分的数量
        if(np.dot(w,test2[i])+w0>0):
            wrong+=1
print("the right rate of fisher is:%f" %(1-wrong/(test1.shape[0]+test2.shape[0])))