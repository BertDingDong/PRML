import numpy as np
import matplotlib.pyplot as plt
#本代码对应第五大题中采用平均距离合并层次聚类算法的情况
#除了类之间距离的计算方式之外跟上一个代码都是一样的，注释参考上一份代码
class together:
    def __init__(self,k):
        self.k=k

    def distance(self, groupa, groupb):
        groupa = np.array(groupa)
        groupb = np.array(groupb)
        distances = np.linalg.norm(groupa[:, np.newaxis] - groupb, axis=2)
        #修改了返回的距离，这里用平均距离
        return np.sum(distances)/groupa.shape[0]/groupb.shape[0]
    
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