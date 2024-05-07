import numpy as np
import matplotlib.pyplot as plt
#该代码对应第五大题第二小题的协方差矩阵均相等且为对角阵的情况
#下为读取数据过程，将三类数据分别读取
data = np.load(r'D:\专业课\模式识别\代码\第四次作业\data.npz')
train1=data['x1_train']
train2=data['x2_train']
train3=data['x3_train']

test1=data['x1_test']
test2=data['x2_test']
test3=data['x3_test']
#下为计算向量均值，将均值作为最大似然估计
u1=np.mean(train1,axis=0)
u2=np.mean(train2,axis=0)
u3=np.mean(train3,axis=0)
#由于协方差矩阵相等且为对角阵，只需要考虑每一类的均值即可计算出后验概率
def possible(x,u):
    return (-1)*(np.dot((x-u).T,(x-u)))

wrong=0
#下为分别对每一类进行计算后验概率分类，如果出现了其它类的后验概率大于正确分类的情况，则错误点+1
for i in range(50):
    test1possible=possible(test1[i],u1)
    if(test1possible<=possible(test1[i],u2)):
        wrong+=1
        continue
    if(test1possible<=possible(test1[i],u3)):
        wrong+=1

for i in range(50):
    test2possible=possible(test2[i],u2)
    if(test2possible<=possible(test2[i],u1)):
        wrong+=1
        continue
    if(test2possible<=possible(test2[i],u3)):
        wrong+=1

for i in range(50):
    test3possible=possible(test3[i],u3)
    if(test3possible<=possible(test3[i],u2)):
        wrong+=1
        continue
    if(test3possible<=possible(test3[i],u1)):
        wrong+=1
#输出错误率
print("the right rate is",1.0-(wrong/150.0))
#输出判别函数
print("the decision function of type1 is:",2*u1[0],"*x0+",2*u1[1],"*x1",-1*np.dot(u1.T,u1))
print("the decision function of type2 is:",2*u2[0],"*x0+",2*u2[1],"*x1",-1*np.dot(u2.T,u2))
print("the decision function of type3 is:",2*u3[0],"*x0+",2*u3[1],"*x1",-1*np.dot(u3.T,u3))