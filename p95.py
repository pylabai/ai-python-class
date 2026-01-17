import numpy as np
import matplotlib.pyplot as plt
#建立資料 Initialization
xs=[(0, 0), (0, 1), (1, 0), (1, 1)]
yd=[0,1,1,1]
w=[0.2,0.3]
theta=0.7
alpha=0.2
x=np.arange(-1,3,0.01)
epouch=0
training_times=0
# 自行設計直線函數
def line (A):
    return (-A*w[0]+theta)/w[1]

def axis(A):
    return 0*A

#請使用者輸入資料
print("*****This is a Perceptron learning program****")

#yd-eval (input("請輸入yd,請用细分,AND [0,0,0,1],OR [0,1,2,4]:
select=eval(input("請輸入:1<AND> or 2<OR> or 3<XOR>: "))
if select == 1:
    yd=[0,0,0,1]
    plt.plot([0],[0], 'bo') #繪製四點用藍色區分
    plt.plot([0],[1], 'bo') #繪製四點用藍色區分
    plt.plot([1],[0], 'bo') #繪製四點用藍色區分
    plt.plot([1],[1], 'ro') #繪製四點用紅色區分
if select == 2:
    yd=[0,1,1,1]
    plt.plot([0],[0],'ro')  #繪製四點用紅色區分
    plt.plot([0],[1], 'bo') #繪製四點用藍色區分
    plt.plot([1],[0], 'bo') #繪製四點用藍色區分
    plt.plot([1],[1], 'bo') #繪製四點用藍色區分
if select == 3:
    yd=[0,1,1,0]
    plt.plot([0], [0], 'ro') #繪製四點用紅色區分
    plt.plot([0], [1], 'bo') #繪製四點用藍色區分
    plt.plot([1], [0], 'bo') #繪製四點用藍色區分
    plt.plot([1], [1], 'ro') #繪製四點用紅色區分
print('Your learning target is\n', yd)
theta=float(input('請輸入門檻值(0.1~0.9):'))
print('w1=%f,w2=%f'%(w[0], w[1])) #列印初始weight
plt.plot(x, line(x), 'c--') #畫初始設定直線
plt.plot(x, axis(x), 'k--') #draw X
plt.plot(axis(x),x,'k--')#draw Y

#*******Activation
all_correct=0
while(all_correct < 4 and epouch < 50):
    epouch += 1
    print('epouch=',epouch)
    #**讀四筆資料唯一循環
    for i in range(0,4):
        if ((xs[i][0]*w[0]+xs[i][1]*w[1]-theta)>= 0):
            ya=1
        else:
            ya=0
        e=yd[i]-ya #calculate error
        #**** change weights (Learning)
        if (e!=0):
            training_times +=1
            all_correct=0
            w[0]=w[0]+alpha*xs[i][0]*e
            w[1]=w[1]+alpha*xs[i][1]*e
            print('training_times', training_times)
            print('w1-%4.2f,w2=%4.2f'%(w[0],w[1]))
            plt.plot(x, line(x))
        else:
            all_correct += 1
print("******劃出調整後的直線,鏈線為原始設定*******")
plt.xlabel('---X1---')
plt.ylabel('---X2---')
plt.show()
# Program End
