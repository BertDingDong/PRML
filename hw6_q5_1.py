import numpy as np
import matplotlib.pyplot as plt
#本代码对应第五大题中采用最近邻聚类算法的情况
class together:
    #初始化需要分的类别数
    def __init__(self,k):
        self.k=k
    #计算类与类的距离远，由于是最近邻聚类，因此选取距离最小的两个点作为聚类结果
    def distance(self, groupa, groupb):
        groupa = np.array(groupa)
        groupb = np.array(groupb)
        distances = np.linalg.norm(groupa[:, np.newaxis] - groupb, axis=2)
        return np.min(distances)
    #这个是计算平方误差
    def wrongdistance(self,a,b):
        return np.linalg.norm(a-b)**2
    #每次get相当于把最近的两个类合并
    def get(self,groups):
        #本来想在原来的groups上面操作的，但是一直索引报错，python的索引有点奇怪，干脆直接键一个新的列表了
        #新建一个列表再操作会导致这个程序很慢，也是一个可以改进的点
        newgroups=[]
        min=np.inf
        #本来这里是用一个列表把两个值存起来的，之前debug的时候改成分开存了，虽然不是很美观但是我也不想再改了
        a=0
        b=0
        #比较所有类之间的距离后把最小的两个给记住
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                if(self.distance(groups[i],groups[j])<min):
                    min=self.distance(groups[i],groups[j])
                    a=i
                    b=j
        #把最近的两类合并，其它正常加入新列表里就行了
        for i in range(len(groups)):
            if(i!=a and i!=b):
                newgroups.append(groups[i])
            elif (i==a):
                newgroups.append(groups[a]+groups[b])
        return newgroups
    #分类，真正实现这个功能的入口
    def classify(self,sample):
        #初始化先将所有样本归为一类
        self.groups=[[sample[i]] for i in range(sample.shape[0])]
        #迭代sample.shape[0]-self.k次后就能让最后的类别数为k
        for i in range(sample.shape[0]-self.k):
            self.groups=self.get(self.groups)
        return self.groups
    #计算平方误差的函数
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
#这里由于我希望用for循环直接对groups进行操作，因此把颜色也做成一个数组，然后不同的group和颜色之间就能有对应关系了
colors=['red','blue','green','black']
for i,group in enumerate(groups):
    x_values = [row[0] for row in group]
    y_values = [row[1] for row in group]
    plt.scatter(x_values, y_values, c=colors[i])
#展示分类结果，计算错误率
plt.show()
print(together.wrong(groups))