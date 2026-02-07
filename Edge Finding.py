import cv2

# 11 讀取影像並輸入檔名
picName = input('picture name is: ') # enter picture name with attached file
image = cv2.imread(picName, -1)      # 0:grayscale image, 1:color image, -1: unchanged
# 13 影像縮放
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
cv2.imshow('Original image', image)
cv2.waitKey(0)

# 16 讀取灰階影像
image = cv2.imread(picName, 0)       # 0:grayscale image, 1:color image
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
cv2.imshow('Greyscale image', image)
cv2.waitKey(0)

# --- 雜訊去除 (Noise Reduction) ---
# 21 中值濾波 (Median Blur)
Gimage = cv2.medianBlur(image, 3)    # 中值濾波，數字為遮罩大小
cv2.imshow('MedianBlur image', Gimage)
cv2.waitKey(0)

# 25 高斯濾波 (Gaussian Blur)
Gimage = cv2.GaussianBlur(image, (7, 7), 1) # 高斯濾波遮罩只能用奇數:3,5,7
cv2.imshow('GaussianBlur image', Gimage)
cv2.waitKey(0)

# --- Sobel Edge Detection ---
# 30 Sobel X軸偵測
sobelx = cv2.Sobel(src=Gimage, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)

# 34 Sobel Y軸偵測
sobely = cv2.Sobel(src=Gimage, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)

# 38 Sobel X+Y軸偵測
sobelxy = cv2.Sobel(src=Gimage, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)
cv2.imshow('Sobel XY', sobelxy)
cv2.waitKey(0)

# --- Canny Edge Detection ---
"""
44 高於 high_threshold: 為 strong edge，我們直接保留
45 介於 low_threshold 與 high_threshold: 為 weak edge
46 Canny會檢測 weak edge 是否能與 strong edge 相連，如果會相連的才會被保留
47 低於 low_threshold: 我們都不當作 edge
"""

# 49 使用較高門檻值的 Canny 偵測
edges = cv2.Canny(image=image, threshold1=200, threshold2=200) # Canny
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

# 53 使用不同門檻組合的 Canny 偵測 (Low=100, High=200)
edges = cv2.Canny(image=image, threshold1=100, threshold2=200) # Canny
cv2.imshow('Canny Edge L100 H200', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
