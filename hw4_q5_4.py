import numpy as np
import matplotlib.pyplot as plt
#该代码对应第五大题第二小题各类协方差矩阵不同的情况
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
#以上代码和前面的都是一样的
#唯一有区别的就是下面分别计算每一类样本的协方差矩阵的过程，相当于将每一类样本的协方差矩阵作为类协方差矩阵的最大似然估计
center1=train1-u1
center2=train2-u2
center3=train3-u3
sigma1=np.cov(center1,rowvar=False)
sigma2=np.cov(center2,rowvar=False)
sigma3=np.cov(center3,rowvar=False)
#由于引入了各类的协方差矩阵，后验概率的计算也变得更复杂了
def possible(x,u,sigma):
    temp=2*np.dot(np.linalg.inv(sigma),u.T)
    return (-1)*(np.dot(np.dot(x,np.linalg.inv(sigma)),x.T))+np.dot(temp,x.T)-np.dot(np.dot(u,np.linalg.det(sigma)),u.T)-np.log(np.linalg.det(sigma))

wrong=0
#同样的后验概率计算环节
for i in range(50):
    test1possible=possible(test1[i],u1,sigma1)
    if(test1possible<=possible(test1[i],u2,sigma2)):
        wrong+=1
        continue
    if(test1possible<=possible(test1[i],u3,sigma3)):
        wrong+=1

for i in range(50):
    test2possible=possible(test2[i],u2,sigma2)
    if(test2possible<=possible(test2[i],u1,sigma1)):
        wrong+=1
        continue
    if(test2possible<=possible(test2[i],u3,sigma3)):
        wrong+=1

for i in range(50):
    test3possible=possible(test3[i],u3,sigma3)
    if(test3possible<=possible(test3[i],u2,sigma2)):
        wrong+=1
        continue
    if(test3possible<=possible(test3[i],u1,sigma1)):
        wrong+=1

print("the right rate is",1.0-(wrong/150.0))
#同样的判别函数计算环节
usigma1=np.dot(u1,np.linalg.inv(sigma1))
usigma2=np.dot(u2,np.linalg.inv(sigma2))
usigma3=np.dot(u3,np.linalg.inv(sigma3))
print("the decision function of type1 is:%fx0%fx1%f" % (2*usigma1[0],2*usigma1[1],-1*np.dot(usigma1,u1.T)-np.log(np.linalg.det(sigma1))))
print("the decision function of type2 is:%fx0+%fx1%f" % (2*usigma2[0],2*usigma2[1],-1*np.dot(usigma2,u2.T)-np.log(np.linalg.det(sigma2))))
print("the decision function of type3 is:%fx0+%fx1%f" % (2*usigma3[0],2*usigma3[1],-1*np.dot(usigma3,u3.T)-np.log(np.linalg.det(sigma3))))