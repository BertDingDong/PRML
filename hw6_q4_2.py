import numpy as np
import matplotlib.pyplot as plt
#本代码对应第四题第一小问的EM算法（协方差矩阵随机）的情况
#将EM算法封装成类，方便后续要用的时候直接搬过去
class EM:
    #初始化：要分类的组数，维度
    def __init__(self,k,dimension):
        self.k=k
        self.dimension=dimension
        #初始化先验概率，均值，协方差矩阵
        self.preposi=np.ones(k)/k
        #这里对均值和协方差矩阵的初始化采用列表和for循环的方式，这样能对不同的k值做到兼容，避免了重复复制粘贴
        self.means=[np.random.uniform(low=-10,high=10,size=dimension) for i in range(k)]
        self.covs=[np.eye(dimension) for i in range(k)]  
    #计算后验概率，pos就是possible，把一个样本属于每一类的后验概率存在一个数组里
    def possible(self,one):
        pos=np.empty(self.k)
        for i in range(self.k):
            pos[i]=self.preposi[i]*(1/((2*np.pi)**(self.dimension/2)*np.linalg.det(self.covs[i])**0.5))*np.exp(-0.5*(one-self.means[i]).dot(np.linalg.inv(self.covs[i])).dot((one-self.means[i]).T))
        return pos/np.sum(pos)
    #再把所有样本的后验概率做成一个矩阵的形式方便后续操作，也就是EM算法的E
    def excepetion(self,sample):
        lateposi=np.empty((sample.shape[0],self.k))
        for i in range(sample.shape[0]):
            lateposi[i]=self.possible(sample[i])
        return lateposi
    #EM算法的M操作，迭代更新参数
    def maximization(self,sample):
        lateposi=self.excepetion(sample)
        for i in range(self.k):
            #因为要把好几个协方差矩阵算加权平均，直接用np.sum的话会出问题，这里直接用土办法，初始化零矩阵后一个个加起来
            cov=np.zeros((self.dimension,self.dimension))
            #更新均值
            self.means[i]=np.sum(lateposi[:,i].reshape(-1,1)*sample,axis=0)/np.sum(lateposi[:,i])
            #更新协方差矩阵
            diff = sample - self.means[i]
            for j in range(sample.shape[0]):
                cov += lateposi[j,i] * np.outer(diff[j], diff[j])
            self.covs[i] = cov / np.sum(lateposi[:,i])
            #更新先验概率
            self.preposi[i] = np.sum(lateposi[:,i]) / sample.shape[0]
    #分类，迭代time次，实测100次完全足够收敛了
    def classify(self,sample,time):
        for i in range(time):
            self.maximization(sample)
        #最后返回的是分类结果，取期望最大的
        return np.argmax(self.excepetion(sample),axis=1)
#读取数据
sample1=np.load('sample1.npy')
sample2=np.load('sample2.npy')
sample3=np.load('sample3.npy')
sample4=np.load('sample4.npy')
#把数据组合在一块
sample=np.concatenate((sample1,sample2,sample3,sample4),axis=0)
#实例化
em=EM(4,2)
#计算结果
outcome=em.classify(sample,100)
#作图
plt.scatter(sample[:,0],sample[:,1],c=outcome)
plt.show()