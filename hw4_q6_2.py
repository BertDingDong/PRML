import numpy as np
import matplotlib.pyplot as plt
import h5py
##本代码对应第六大题第二小题
##读取数据，应该跟上一个小题是一样的
with h5py.File(r"D:\专业课\模式识别\代码\第四次作业\usps.h5", 'r') as hf:
        train = hf.get('train')
        x_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        x_te = test.get('data')[:]
        y_te = test.get('target')[:]

train1 = np.empty((0, 256))
train2 = np.empty((0, 256))

test1 = np.empty((0, 256))
test2 = np.empty((0, 256))

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
#由于实际运行过程中发现sigma矩阵的值过小导致其取行列式并取对数时，会出现超出机器计算范围的情况，导致后续的计算出现问题
#因此为每一个样本进行了一定的放大，并且可以证明这种放大是不会影响最后分类的结果的
train1*=10
train2*=10
test1*=10
test2*=10
#下面都是很俗套的内容了，没有什么有意思的地方就不多赘述了
u1=np.mean(train1,axis=0)
u2=np.mean(train2,axis=0)
#同样要加上扰动后才能保证矩阵可逆
sigma1=np.cov(train1,rowvar=False)+0.00001*np.eye(train1.shape[1])*np.mean(u1)
sigma2=np.cov(train2,rowvar=False)+0.00001*np.eye(train2.shape[1])*np.mean(u2)

def possible(x,u,sigma):
    temp=2*np.dot(np.linalg.inv(sigma),u.T)
    return (-1)*(np.dot(np.dot(x,np.linalg.inv(sigma)),x.T))+np.dot(temp,x.T)-np.dot(np.dot(u,np.linalg.det(sigma)),u.T)-np.log(np.linalg.det(sigma))

wrong=0
for i in range(len(test1)):
    test1possible=possible(test1[i],u1,sigma1)
    if(test1possible<=possible(test1[i],u2,sigma2)):
        wrong+=1
for i in range(len(test2)):
    test2possible=possible(test2[i],u2,sigma2)
    if(test2possible<=possible(test2[i],u1,sigma1)):
        wrong+=1

print("the right rate of bayes is:%f" % (1.0-wrong/(len(test1)+len(test2))))