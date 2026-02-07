"""
Edge Finding (Canny)
This is a program for edge detection
Created on Thu Nov 18 14:10:02 2021
@author: 李天明
"""
import cv2

# 讀取原始影像
img = cv2.imread('pepper.jpg', -1)
cv2.imshow('Original', img)
cv2.waitKey(0)

# 讀取為灰階影像
img_grey = cv2.imread('pepper.jpg', 0)
cv2.imshow('Greyscale', img_grey)
cv2.waitKey(0)

# 執行 Canny 邊緣偵測 
# threshold1 與 threshold2 設定為 120
edges = cv2.Canny(image=img_grey, threshold1=120, threshold2=120)

# 顯示結果
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
