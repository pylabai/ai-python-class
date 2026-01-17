import numpy as np
import matplotlib.pyplot as plt
w=[0.2,0.1]
theta=0.7
x=np.arange(-1,3,0.01)
def line(A):
    return (-A*w[0]+theta)/w[1]
print("這是plot()指令練習....")
theta=float(input("請輸入門檻值(0.1-0.9):"))
w[0]=eval(input("請體入w1,請用0.1~0.9:"))
w[1]=eval(input("請輸入w2,請用0.1~0.9:"))
print("w1=%f,w2=%f",(w[0],w[1])) #weights
print("....分割直線....")
#πρίτ.ακίς (10,2,0,2))***XY軸範圍
plt.plot([0], [0], 'ro') #製四點用紅色區分
plt.plot([0], [1], 'bo') #製四點用紅色區分
plt.plot([1], [0], 'bo') #繪製四點用红色區分
plt.plot([1], [1], 'bo') #製四點用紅色區分
plt.plot(x,line(x))
plt.xlabel("---x1---")
plt.ylabel("---x2---")
plt.show()
