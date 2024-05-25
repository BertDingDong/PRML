#该代码对应第四题的生成随机样本的过程，生成四组样本并且存储在四个npy文件中
#其实在这里直接把四组样本合并成一个样本，后面直接用合并后的样本就行了，因为判断具体属于哪一组正是后面要做的事
import numpy as np
import matplotlib.pyplot as plt
#这里均值采用-10到10的随机点
mean1=np.random.uniform(low=-10,high=10,size=2)
mean2=np.random.uniform(low=-10,high=10,size=2)
mean3=np.random.uniform(low=-10,high=10,size=2)
mean4=np.random.uniform(low=-10,high=10,size=2)
#方差也是取随机值，可能是这里取值不是很好，导致样本重叠程度还挺高的
cov1=np.random.uniform(low=-5,high=5,size=(2,2))
cov2=np.random.uniform(low=-5,high=5,size=(2,2))
cov3=np.random.uniform(low=-5,high=5,size=(2,2))
cov4=np.random.uniform(low=-5,high=5,size=(2,2))
#用作业里给的np.random.multivariate_normal生成随机样本
sample1=np.random.multivariate_normal(mean1,cov1,50)
sample2=np.random.multivariate_normal(mean2,cov2,50)
sample3=np.random.multivariate_normal(mean3,cov3,50)
sample4=np.random.multivariate_normal(mean4,cov4,50)
#保存样本
np.save('sample1.npy',sample1)
np.save('sample2.npy',sample2)
np.save('sample3.npy',sample3)
np.save('sample4.npy',sample4)
#这个代码没怎么用for循环，因为前面试了一下好像报错了，干脆直接重复四次