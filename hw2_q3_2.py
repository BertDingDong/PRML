#该代码对应第三大题第二小题
import numpy as np
import matplotlib.pyplot as plt

w1=np.array([[0.1,1.1],[6.8,7.1],[-3.5,-4.1],[2.0,2.7],[4.1,2.8],[3.1,5.0],[-0.8,-1.3],[0.9,1.2],[5.0,6.4],[3.9,4.0]])
w2=np.array([[7.1,4.2],[-1.4,-4.3],[4.5,0.0],[6.3,1.6],[4.2,1.9],[1.4,-3.2],[2.4,-4.0],[2.5,-6.1],[8.4,3.7],[4.1,-2.2]])
w3=np.array([[-3.0,-2.9],[0.5,8.7],[2.9,2.1],[-0.1,5.2],[-4.0,2.2],[-1.3,3.7],[-3.4,6.2],[-4.1,3.4],[-5.1,1.6],[1.9,5.1]])
w4=np.array([[-2.0,-8.4],[-8.9,0.2],[-4.2,-7.7],[-8.5,-3.2],[-6.7,-4.0],[-0.5,-9.2],[-5.3,-6.7],[-8.7,-6.4],[-7.1,-9.7],[-8.0,-6.3]])
def oldfisher():#这是采用两组样本中间值作为阈值的Fisher判别法
    #以下过程严格按照书中定义实现，符号也是一样的
    wrong=0
    m1=np.array([0.0,0.0])
    m2=np.array([0.0,0.0])
    for i in range(0,10):
        m1+=w2[i]
    for i in range(0,10):
        m2+=w3[i]
    m1/=10
    m2/=10
    s1=np.array([[0.0,0.0],[0.0,0.0]])
    s2=np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(0,10):
        s1+=np.dot((w2[i]-m1).reshape(-1,1),(w2[i]-m1).reshape(1,-1))
    for i in range(0,10):
        s2+=np.dot((w3[i]-m2).reshape(-1,1),(w3[i]-m2).reshape(1,-1))
    sw=s1+s2
    w=np.dot(np.linalg.inv(sw),(m1-m2))
    plt.plot([-w[0]*100,w[0]*100],[-w[1]*100,w[1]*100],'green')#做出投影方向
    w0=(np.dot(w,m1)+np.dot(w,m2))*(-0.5)
    plt.plot([w[1]*100,-w[1]*100],[-w0/w[1]-w[0]*100,-w0/w[1]+w[0]*100],'green')#做出分离面
    for i in range(0,10):#计算错分的数量
        if(np.dot(w,w2[i])+w0<0):
            wrong+=1
        if(np.dot(w,w3[i])+w0>0):
            wrong+=1
    print(wrong*0.05)

def newfisher():#除了修改阈值的取值之外其他都是一样的
    wrong=0
    m1=np.array([0.0,0.0])
    m2=np.array([0.0,0.0])
    for i in range(0,10):
        m1+=w2[i]
    for i in range(0,10):
        m2+=w3[i]
    m1/=10
    m2/=10
    s1=np.array([[0.0,0.0],[0.0,0.0]])
    s2=np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(0,10):
        s1+=np.dot((w2[i]-m1).reshape(-1,1),(w2[i]-m1).reshape(1,-1))
    for i in range(0,10):
        s2+=np.dot((w3[i]-m2).reshape(-1,1),(w3[i]-m2).reshape(1,-1))
    sw=s1+s2
    w=np.dot(np.linalg.inv(sw),(m1-m2))
    plt.plot([-w[0]*100,w[0]*100],[-w[1]*100,w[1]*100],'yellow')
    count=0
    w0=0
    m=np.array([0.0,0.0])
    for i in range(0,10):
        m+=w2[i]
        m+=w3[i]
    m/=20
    w0=np.dot(w,m1)*(-0.65)+np.dot(w,m2)*(-0.35)
    plt.plot([w[1]*100,-w[1]*100],[-w0/w[1]-w[0]*100,-w0/w[1]+w[0]*100],'yellow')
    for i in range(0,10):
        if(np.dot(w,w2[i])+w0<0):
            wrong+=1
        if(np.dot(w,w3[i])+w0>0):
            wrong+=1
    print(wrong*0.05)
    
oldfisher()
newfisher()

x2=w2[:,0]
y2=w2[:,1]
x3=w3[:,0]
y3=w3[:,1]
plt.scatter(x2, y2, color='blue')
plt.scatter(x3, y3, color='red')
plt.show()