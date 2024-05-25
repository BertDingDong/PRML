import numpy as np
import matplotlib.pyplot as plt
#本代码对应第四题第一小问的EM算法（协方差矩阵为对角阵）的情况
#注释大部分跟上一份代码是一样的，没有注释的地方直接参考上一份代码就行了
class EM:
    def __init__(self,k,dimension):
        self.k=k
        self.dimension=dimension
        self.preposi=np.ones(k)/k
        self.means=[np.random.uniform(low=-10,high=10,size=dimension) for i in range(k)]
        self.covs=[np.eye(dimension) for i in range(k)]  
    
    def possible(self,one):
        pos=np.empty(self.k)
        for i in range(self.k):
            pos[i]=self.preposi[i]*(1/((2*np.pi)**(self.dimension/2)*np.linalg.det(self.covs[i])**0.5))*np.exp(-0.5*(one-self.means[i]).dot(np.linalg.inv(self.covs[i])).dot((one-self.means[i]).T))
        return pos/np.sum(pos)
    
    def excepetion(self,sample):
        lateposi=np.empty((sample.shape[0],self.k))
        for i in range(sample.shape[0]):
            lateposi[i]=self.possible(sample[i])
        return lateposi
    
    def maximization(self,sample):
        lateposi=self.excepetion(sample)
        for i in range(self.k):
            cov=np.zeros((self.dimension,self.dimension))
            self.means[i]=np.sum(lateposi[:,i].reshape(-1,1)*sample,axis=0)/np.sum(lateposi[:,i])
            diff = sample - self.means[i]
            #唯一有改动的地方，这里只计算对角元素，保证协方差矩阵经过迭代后还是对角阵
            for j in range(sample.shape[0]):
                cov += lateposi[j,i] * np.eye(self.dimension) * diff[j]* diff[j]
            self.covs[i] = cov / np.sum(lateposi[:,i])
            self.preposi[i] = np.sum(lateposi[:,i]) / sample.shape[0]
            print(self.covs)
    def classify(self,sample,time):
        for i in range(time):
            self.maximization(sample)
        return np.argmax(self.excepetion(sample),axis=1)

sample1=np.load('sample1.npy')
sample2=np.load('sample2.npy')
sample3=np.load('sample3.npy')
sample4=np.load('sample4.npy')
sample=np.concatenate((sample1,sample2,sample3,sample4),axis=0)
em=EM(4,2)
outcome=em.classify(sample,100)
print(outcome)
plt.scatter(sample[:,0],sample[:,1],c=outcome)
plt.show()