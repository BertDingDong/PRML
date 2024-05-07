#本代码对应第三大题第四小题中OVA分类器
import numpy as np
import matplotlib.pyplot as plt

w1=np.array([[0.1,1.1],[6.8,7.1],[-3.5,-4.1],[2.0,2.7],[4.1,2.8],[3.1,5.0],[-0.8,-1.3],[0.9,1.2],[5.0,6.4],[3.9,4.0]])
w2=np.array([[7.1,4.2],[-1.4,-4.3],[4.5,0.0],[6.3,1.6],[4.2,1.9],[1.4,-3.2],[2.4,-4.0],[2.5,-6.1],[8.4,3.7],[4.1,-2.2]])
w3=np.array([[-3.0,-2.9],[0.5,8.7],[2.9,2.1],[-0.1,5.2],[-4.0,2.2],[-1.3,3.7],[-3.4,6.2],[-4.1,3.4],[-5.1,1.6],[1.9,5.1]])
w4=np.array([[-2.0,-8.4],[-8.9,0.2],[-4.2,-7.7],[-8.5,-3.2],[-6.7,-4.0],[-0.5,-9.2],[-5.3,-6.7],[-8.7,-6.4],[-7.1,-9.7],[-8.0,-6.3]])
#前半部分与罗杰斯特回归部分类似
def sigmoid(x):
    return 1/(1+np.exp(-x))
def yinx(w,x):
    return sigmoid(np.dot(x,w))
def logistic(x1,x2,x3,x4):
    x01=np.array([[0,0.1,1.1],[0,6.8,7.1],[0,-3.5,-4.1],[0,2.0,2.7],[0,4.1,2.8],[0,3.1,5.0],[0,-0.8,-1.3],[0,0.9,1.2],[0,5.0,6.4],[0,3.9,4.0]])
    x02=np.array([[0,0.1,1.1],[0,6.8,7.1],[0,-3.5,-4.1],[0,2.0,2.7],[0,4.1,2.8],[0,3.1,5.0],[0,-0.8,-1.3],[0,0.9,1.2],[0,5.0,6.4],[0,3.9,4.0]])
    x03=np.array([[0,0.1,1.1],[0,6.8,7.1],[0,-3.5,-4.1],[0,2.0,2.7],[0,4.1,2.8],[0,3.1,5.0],[0,-0.8,-1.3],[0,0.9,1.2],[0,5.0,6.4],[0,3.9,4.0]])
    x04=np.array([[0,0.1,1.1],[0,6.8,7.1],[0,-3.5,-4.1],[0,2.0,2.7],[0,4.1,2.8],[0,3.1,5.0],[0,-0.8,-1.3],[0,0.9,1.2],[0,5.0,6.4],[0,3.9,4.0]])
    for i in range(0,10):
        x01[i]=np.insert(x1[i],2,1)
    for i in range(0,10):
        x02[i]=np.insert(x2[i],2,1)
    for i in range(0,10):
        x03[i]=np.insert(x3[i],2,1)
    for i in range(0,10):
        x04[i]=np.insert(x4[i],2,1)
    #相当于将罗杰斯特回归的过程重复了四遍
    w1=np.array([0.0,0.0,0.0])
    w2=np.array([0.0,0.0,0.0])
    w3=np.array([0.0,0.0,0.0])
    w4=np.array([0.0,0.0,0.0])
    #以下为对每个判别向量进行计算
    for time in range(0,10000):
        count=np.array([0.0,0.0,0.0])
        for i in range(0,8):
            count+=x01[i]*(yinx(w1,x01[i])-1)
            count+=x02[i]*(yinx(w1,x02[i]))
            count+=x03[i]*(yinx(w1,x03[i]))
            count+=x04[i]*(yinx(w1,x04[i]))
        w1-=count
    for time in range(0,10000):
        count=np.array([0.0,0.0,0.0])
        for i in range(0,8):
            count+=x02[i]*(yinx(w2,x02[i])-1)
            count+=x01[i]*(yinx(w2,x01[i]))
            count+=x03[i]*(yinx(w2,x03[i]))
            count+=x04[i]*(yinx(w2,x04[i]))
        w2-=count
    for time in range(0,10000):
        count=np.array([0.0,0.0,0.0])
        for i in range(0,8):
            count+=x03[i]*(yinx(w3,x03[i])-1)
            count+=x02[i]*(yinx(w3,x02[i]))
            count+=x01[i]*(yinx(w3,x01[i]))
            count+=x04[i]*(yinx(w3,x04[i]))
        w3-=count
    for time in range(0,10000):
        count=np.array([0.0,0.0,0.0])
        for i in range(0,8):
            count+=x04[i]*(yinx(w4,x04[i])-1)
            count+=x02[i]*(yinx(w4,x02[i]))
            count+=x03[i]*(yinx(w4,x03[i]))
            count+=x01[i]*(yinx(w4,x01[i]))
        w4-=count
    wrong=0
    #以下为对测试样本求解错误率
    for i in range(8,10):
        if(np.dot(w1,x01[i])<0):
            wrong+=1
        if(np.dot(w1,x02[i])>0):
            wrong+=1
        if(np.dot(w1,x03[i])>0):
            wrong+=1
        if(np.dot(w1,x04[i])>0):
            wrong+=1
    print('wrong rate of w1:',wrong*12.5)
    wrong=0
    for i in range(8,10):
        if(np.dot(w2,x02[i])<0):
            wrong+=1
        if(np.dot(w2,x01[i])>0):
            wrong+=1
        if(np.dot(w2,x03[i])>0):
            wrong+=1
        if(np.dot(w2,x04[i])>0):
            wrong+=1
    print('wrong rate of w2:',wrong*12.5)
    wrong=0
    for i in range(8,10):
        if(np.dot(w3,x03[i])<0):
            wrong+=1
        if(np.dot(w3,x02[i])>0):
            wrong+=1
        if(np.dot(w3,x01[i])>0):
            wrong+=1
        if(np.dot(w3,x04[i])>0):
            wrong+=1
    print('wrong rate of w3:',wrong*12.5)
    wrong=0
    for i in range(8,10):
        if(np.dot(w4,x04[i])<0):
            wrong+=1
        if(np.dot(w4,x02[i])>0):
            wrong+=1
        if(np.dot(w4,x03[i])>0):
            wrong+=1
        if(np.dot(w4,x01[i])>0):
            wrong+=1
    print('wrong rate of w4:',wrong*12.5)
    print(w1)
    print(w2)
    print(w3)
    print(w4)
    recordx1=x1[:,0]
    recordy1=x1[:,1]
    recordx2=x2[:,0]
    recordy2=x2[:,1]
    recordx3=x3[:,0]
    recordy3=x3[:,1]
    recordx4=x4[:,0]
    recordy4=x4[:,1]
    plt.scatter(recordx1, recordy1, color='blue')
    plt.scatter(recordx2, recordy2, color='red')
    plt.scatter(recordx3, recordy3, color='yellow')
    plt.scatter(recordx4, recordy4, color='black')
    plt.plot([-w1[1],w1[1]],[-w1[2]/w1[1]+w1[0],-w1[2]/w1[1]-w1[0]],'blue')
    plt.plot([-w2[1],w2[1]],[-w2[2]/w2[1]+w2[0],-w2[2]/w2[1]-w2[0]],'red')
    plt.plot([-w3[1],w3[1]],[-w3[2]/w3[1]+w3[0],-w3[2]/w3[1]-w3[0]],'yellow')
    plt.plot([-w4[1],w4[1]],[-w4[2]/w4[1]+w4[0],-w4[2]/w4[1]-w4[0]],'black')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.show()

logistic(w1,w2,w3,w4)