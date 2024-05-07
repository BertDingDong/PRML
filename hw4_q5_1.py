import numpy as np
import matplotlib.pyplot as plt
#该代码对应第五大题第一小题，展示训练样本点
#相信本代码不需要太多说明
data = np.load(r'D:\专业课\模式识别\代码\第四次作业\data.npz')
train1=data['x1_train']
train2=data['x2_train']
train3=data['x3_train']
plt.scatter(train1[:,0],train1[:,1],color='blue')
plt.scatter(train2[:,0],train2[:,1],color='red')
plt.scatter(train3[:,0],train3[:,1],color='yellow')
plt.show()