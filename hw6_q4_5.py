import numpy as np
import matplotlib.pyplot as plt
#本代码对应第四题第二小问
#K均值类的内容大部分跟上一份代码类似，因此重复之处不多赘述
class K:
    def __init__(self,k,dimension):
        self.k=k
        self.dimension=dimension
        self.means=[np.random.uniform(low=-10,high=10,size=dimension) for i in range(k)]

    def distance(self,a,b):
        return np.linalg.norm(a-b)
    
    def into(self,sample):
        outcome=np.empty(sample.shape[0])
        for i in range(sample.shape[0]):
            outcome[i]=np.argmin([self.distance(sample[i],mean) for mean in self.means])
        return outcome
    
    def update(self,sample,outcome):
        #由于类别数较多时，会出现某些类别里面没有样本的情况，这里就要加一个判断
        for i in range(self.k):
            if np.any(outcome == i):  
                self.means[i] = np.mean(sample[outcome==i], axis=0)
            else:  
                pass

    def classify(self,sample,time):
        for i in range(time):
            out=self.into(sample)
            self.update(sample,out)
        return out
    #由于要计算平方误差，这里添加了一个计算平方误差的函数
    def wrong(self,sample,outcome):
        return np.sum([self.distance(sample[i],self.means[int(outcome[i])])**2 for i in range(sample.shape[0])])

sample1=np.load('sample1.npy')
sample2=np.load('sample2.npy')
sample3=np.load('sample3.npy')
sample4=np.load('sample4.npy')
sample=np.concatenate((sample1,sample2,sample3,sample4),axis=0)
#这里把不同类别的K均值分类用一个列表存起来
ks=[K(i,2)for i in range(2,11)]
#后面的操作就可以直接对这个列表里面的元素进行循环操作了，会简洁很多
outcomes=[ks[i].classify(sample,100)for i in range(0,9)]
wrongs=[ks[i].wrong(sample,outcomes[i])for i in range(0,9)]
plt.plot(range(2,11),wrongs)
plt.show()