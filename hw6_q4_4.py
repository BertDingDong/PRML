import numpy as np
import matplotlib.pyplot as plt
#本代码对应第四题第一小问的K均值算法
class K:
    #初始化依然是引入需要分的类别数跟维度
    def __init__(self,k,dimension):
        self.k=k
        self.dimension=dimension
        self.means=[np.random.uniform(low=-10,high=10,size=dimension) for i in range(k)]
    #由于K均值算法需要大量计算距离，这里直接把距离作为一个函数留在这了
    def distance(self,a,b):
        return np.linalg.norm(a-b)
    #into指的是把样本进行分类，计算每个样本和每个聚类中心的距离，选取最小的作为分类结果
    def into(self,sample):
        outcome=np.empty(sample.shape[0])
        for i in range(sample.shape[0]):
            #这里直接用列表+for循环的方式来实现计算多个距离，避免复制粘贴
            outcome[i]=np.argmin([self.distance(sample[i],mean) for mean in self.means])
        return outcome
    #更新聚类中心
    def update(self,sample,outcome):
        for i in range(self.k):
            self.means[i]=np.mean(sample[outcome==i],axis=0)
    #分类，也是迭代time次
    def classify(self,sample,time):
        for i in range(time):
            out=self.into(sample)
            self.update(sample,out)
        return out
#下面跟之前的差不多
sample1=np.load('sample1.npy')
sample2=np.load('sample2.npy')
sample3=np.load('sample3.npy')
sample4=np.load('sample4.npy')
sample=np.concatenate((sample1,sample2,sample3,sample4),axis=0)
k=K(4,2)
outcome=k.classify(sample,100)
print(outcome)
plt.scatter(sample[:,0],sample[:,1],c=outcome)
plt.show()