import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Initialization
xs=[(0.0, 0.0), (0.0, 1.0), (1.0, 8.0), (1.0, 1.0)]

#列印輸入矩陣
for i in range(4):
    for j in range(2):
        print()
    print(float(xs[i][j]),end='')

#目標矩陣
yd=[0.0,0.0,0.0,0.0]

#W13,W23,W24,W24,W35,W45
w= [0.5,0.4,0.9,1.0,-1.2,1.1]

#Y3,Y4,Y5 三個神經元的輸出
ya=[0.0,0.0,0.0]

#T3, T4, TS 三個神經元的門檻值
theta=[0.8,-0.1,0.3]

#Learning rate
alpha=0.2

#繪圖範圍
x=np.arange(-1,3,0.01)

#循環次數
epouch =0

#訓練次數
training_times=0

# 自行設計直線函数,Y3
MSElist=[]
def line1(A):
    return (-A*w[0]+theta[0])/w[1]

#自行設計直線函數,Y4
def line2(A):
    return (-A*w[2]+theta[1])/w[3]

#設計軸線函數
def axis (A):
    return 0*A

#Weight 取亂數值

for i in range(6):
    w[i]=random.random()
    print('Weight %d=%f'%(i,w[i]))

#使用者資料輸入
print("***This is a Multi-Layer Perceptron learning program****")
select=eval(input("請入: <AND OF ZOR> or <XOR>:"))
if select == 1:
    yd=[0.0,0.0,0.0,1.0]
    plt.plot([0],[0], 'ro') #繪製四點用紅色區分
    plt.plot([0],[1], 'bo') #繪製四點用藍色區分
    plt.plot([1],[0], 'bo') #繪製四點用藍色區分
    plt.plot([1],[1], 'ro') #繪製四點用紅色區分
if select == 2:
    yd=[0.0,1.0,1.0,1.0]
    plt.plot([0],[0], 'ro') #繪製四點用紅色區分
    plt.plot([0],[1], 'bo') #繪製四點用藍色區分
    plt.plot([1],[0], 'bo') #繪製四點用藍色區分
    plt.plot([1],[1], 'bo') #繪製四點用藍色區分
if select== 3:
    yd=[0.0,1.0,1.0,0.0]
    plt.plot([0],[0], 'ro') #繪製四點用紅色區分
    plt.plot([0],[1], 'bo') #繪製四點用藍色區分
    plt.plot([1],[0], 'bo') #繪製四點用藍色區分
    plt.plot([1],[1], 'ro') #繪製四點用紅色區分
print("Your learning target is\n",yd)
plt.plot(x,line1(x), 'c-')
plt.plot(x,line2(x), 'c--')#最初的設定RY4
plt.plot(x,axis(x), 'k-')#edraw x
plt.plot(axis(x),x,'k-')#draw Y


#trainingLoop

#square error
SE=0.0

#Mean Square error
Accuracy=0.0

MSE=1.0

Accuracy=eval(input("What is the MSEPEX: 0.01 от 0.001 or 0.0001: "))
while (MSE > Accuracy and epouch <100000): #step 4, Iteration
    epouch + 1

    print("epouch",epouch)
    SE=0.0
    for i in range(4):
        #step 2a
        #****feed forward #colculate 13 and 14

        ya[0]=1.0/(1.0+math.exp(-(xs[i][0]*w[0]+xs[i][1]*w[1]-theta[0]))) 
        ya[1]=1.0/(1.0+math.exp(-(xs[i][0]*w[2]+xs[i][1]*w[3]-theta[1])))

        #step 2b 
        #colculate YS 
        ya[2]=1.0/(1.0+math.exp(-(ya[0]*w[4]+ya[1]*w[5]-theta[2])))
        #vs 
        print('Y3NF, Y4 %f, YS=%f '%(ya[0],ya[1],ya[2]))
        #step 30
        e=yd[i]-ya[2]
        #calculate real error 
        ge5=ya [2] (1.0-ya[2])*e
        #calculate gradient error of 13
        #change weights (Learning)
        #******backward propagation
        w[4]= w[4]+alpha*ya[0]*ge5 
        w[5]= w[5]+alpha*ya[1]*ge5
        theta[2]= theta[2]+ (-alpha)*ge5 #change thetas#change 
        #35 
        #change 45
        #step 3b
        #...... error backward calculation this is most important p
        ge3=ya[0]*(1.0-ya[0])*ge5*w[4]#gradient error of VI with error-5]


        ge4=ya[1]*(1.0-ya[1])*ge5*w[5]#gradient error of 12 with error-
        #change weights (Learning)
        w[0]= w[0]+alpha*xs[i][0]*ge3#change 13
        w[1]= w[1]+alpha*xs[i][1]*ge3
        w[2]= w[2]+alpha*xs[i][0]*ge4#change 14
        w[3]= w[3]+alpha*xs[i][1]*ge4
        theta[0]= theta[0]+ (-alpha)*ge3 #change thetas#change 13
        theta[1]= theta[1]+ (-alpha)*ge4 #change thetas#change 14
        SE=SE+e*e
    MSE=SE/4.0  #mean square error
    MSElist.append(MSE)
plt.plot(x,line1(x), 'r-')
plt.plot(x,line2(x), 'r--')#final Y4
plt.xlabel('---X1---')
plt.ylabel('---X2---')
plt.show()

plt.plot((range(epouch)),MSElist)
plt.xlabel('---epouch---')
plt.ylabel('---MSE---')
plt.show()
