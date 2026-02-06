"""
Histogram Contrast Testing for OpenCV
Using histogram method to draw a histogram diagram for a greyscale image.
@author: 李天明
Date: 2022/03/21
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 初始化直方圖統計陣列 [cite: 74, 76, 77, 82]
Histogram = []
Histogram_new = []
for i in range(256):
    Histogram.append(0)
    Histogram_new.append(0)

# 設定 x 軸範圍 [cite: 84]
x = np.arange(0, 256, 1)

# 讀取影像 [cite: 86]
picName = input('picture name is: ') # 例如: ABC.jpg
image = cv2.imread(picName, -1)
cv2.imshow('original image', image)
cv2.waitKey(0)

# 獲取影像尺寸 [cite: 89, 90]
high = image.shape[0]
width = image.shape[1]

# 讀取灰階影像進行處理 [cite: 91, 106]
greyimage = cv2.imread(picName, 0)
cv2.imshow('grey image', greyimage)
cv2.waitKey(0)

# 統計原始直方圖 [cite: 111, 112, 114, 116]
for i in range(high):
    for j in range(width):
        grey = greyimage[i, j]
        Histogram[grey] += 1

# 找出最小及最大像素值 [cite: 118, 119, 127, 129, 135]
first = 0
min_pixel = 0
max_pixel = 255
for i in range(256):
    if (Histogram[i] != 0):
        if first == 0:
            min_pixel = i
            first = 1
        max_pixel = i

print("minimum pixel=", min_pixel) [cite: 136]
print("maximum pixel=", max_pixel) [cite: 137]

# 顯示原始直方圖 [cite: 139, 140, 143]
plt.bar(x, Histogram, color='red')
plt.title("Before")
plt.xlabel('pixel value')
plt.ylabel('Frequency')
plt.show()

# 直方圖拉伸處理 [cite: 167, 168, 169, 174, 176]
for i in range(high):
    for j in range(width):
        # 公式: Pout = (Pin - min) * (255 / (max - min))
        temp = (greyimage[i, j] - min_pixel) * (255 / (max_pixel - min_pixel))
        greyimage[i, j] = np.clip(temp, 0, 255)
        new_grey = greyimage[i, j]
        Histogram_new[new_grey] += 1

# 顯示調整後的直方圖與影像 [cite: 177, 178, 181, 183]
plt.bar(x, Histogram_new, color='blue')
plt.title("After")
plt.xlabel('pixel value')
plt.ylabel('Frequency')
plt.show()

cv2.imshow('Contrast image', greyimage)
cv2.waitKey(0)
print("press enter to close windows") [cite: 186]
cv2.destroyAllWindows() [cite: 188]
