import numpy as np
import matplotlib.pyplot as plt
#本代码对应第五大题的分步最优层次聚类算法的情况
#也是除了距离其他跟hw6_q5_1那份一样
class together:
    def __init__(self,k):
        self.k=k
    #这里距离计算就是取中心点的距离然后乘上两类样本数量的调和平均数
    def distance(self, groupa, groupb):
        groupa = np.array(groupa)
        groupb = np.array(groupb)
        meana=np.mean(groupa,axis=0)
        meanb=np.mean(groupb,axis=0)
        distance = np.linalg.norm(meana-meanb)
        return distance*np.sqrt(len(groupa)*len(groupb)/(len(groupa)+len(groupb)))
    
    def wrongdistance(self,a,b):
        return np.linalg.norm(a-b)**2
    
    def get(self,groups):
        newgroups=[]
        min=np.inf
        a=0
        b=0
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                if(self.distance(groups[i],groups[j])<min):
                    min=self.distance(groups[i],groups[j])
                    a=i
                    b=j
        for i in range(len(groups)):
            if(i!=a and i!=b):
                newgroups.append(groups[i])
            elif (i==a):
                newgroups.append(groups[a]+groups[b])
        return newgroups

    def classify(self,sample):
        self.groups=[[sample[i]] for i in range(sample.shape[0])]
        for i in range(sample.shape[0]-self.k):
            self.groups=self.get(self.groups)
        return self.groups
    
    def wrong(self,groups):
        wrong=0
        for group in groups:
            mean=np.mean(group,axis=0)
            wrong+=np.sum([self.wrongdistance(one,mean) for one in group])
        return wrong

sample1=np.load('sample1.npy')
sample2=np.load('sample2.npy')
sample3=np.load('sample3.npy')
sample4=np.load('sample4.npy')
sample=np.concatenate((sample1,sample2,sample3,sample4),axis=0)
together=together(4)
groups=together.classify(sample)
colors=['red','blue','green','black']
for i,group in enumerate(groups):
    x_values = [row[0] for row in group]
    y_values = [row[1] for row in group]
    plt.scatter(x_values, y_values, c=colors[i])
plt.show()
print(together.wrong(groups))