import numpy as np
import matplotlib.pyplot as plt
#该代码对应第五大题第二小题，各类协方差矩阵相同的情况
data = np.load(r'D:\专业课\模式识别\代码\第四次作业\data.npz')
train1=data['x1_train']
train2=data['x2_train']
train3=data['x3_train']

test1=data['x1_test']
test2=data['x2_test']
test3=data['x3_test']

u1=np.mean(train1,axis=0)
u2=np.mean(train2,axis=0)
u3=np.mean(train3,axis=0)
#以上内容与上一代码相同，不多赘述
#下为将所有训练样本合并到一处后，计算出协方差矩阵
train=np.concatenate((train1,train2,train3),axis=0)
u=np.mean(train,axis=0)
center=train-u
sigma=np.cov(center,rowvar=False)
#引入协方差矩阵后，计算后验概率
def possible(x,u,sigma):
    return 2*np.dot(np.dot(u,np.linalg.inv(sigma)),x.T)-np.dot(np.dot(u,np.linalg.inv(sigma)),u.T)

wrong=0
#同样的对每一类进行后验概率的计算和错误判断
for i in range(50):
    test1possible=possible(test1[i],u1,sigma)
    if(test1possible<=possible(test1[i],u2,sigma)):
        wrong+=1
        continue
    if(test1possible<=possible(test1[i],u3,sigma)):
        wrong+=1

for i in range(50):
    test2possible=possible(test2[i],u2,sigma)
    if(test2possible<=possible(test2[i],u1,sigma)):
        wrong+=1
        continue
    if(test2possible<=possible(test2[i],u3,sigma)):
        wrong+=1

for i in range(50):
    test3possible=possible(test3[i],u3,sigma)
    if(test3possible<=possible(test3[i],u2,sigma)):
        wrong+=1
        continue
    if(test3possible<=possible(test3[i],u1,sigma)):
        wrong+=1

print("the right rate is",1.0-(wrong/150.0))
#usigma是u和sigma逆两个矩阵的相乘，由于dot后直接取【0】会报错，就在这里先完成计算后再去取它的值
usigma1=np.dot(u1,np.linalg.inv(sigma))
usigma2=np.dot(u2,np.linalg.inv(sigma))
usigma3=np.dot(u3,np.linalg.inv(sigma))
print("the decision function of type1 is:",2*usigma1[0],"*x0+",2*usigma1[1],"*x1",-1*np.dot(usigma1,u1.T))
print("the decision function of type2 is:",2*usigma2[0],"*x0+",2*usigma2[1],"*x1",-1*np.dot(usigma2,u2.T))
print("the decision function of type3 is:",2*usigma3[0],"*x0+",2*usigma3[1],"*x1",-1*np.dot(usigma3,u3.T))