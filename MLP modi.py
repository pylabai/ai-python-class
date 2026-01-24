import numpy as np
import math
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. 基礎資料與超參數初始化
# ==========================================

# 訓練輸入資料：(x1, x2) 座標，代表邏輯閘的四種輸入組合
xs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

# 權重初始化 (w0~w5)：
# w[0], w[1] 是從輸入層連到隱藏層神經元 Y3 的權重
# w[2], w[3] 是從輸入層連到隱藏層神經元 Y4 的權重
# w[4], w[5] 是從隱藏層連到輸出層神經元 Y5 的權重
w = [random.random() for _ in range(6)]

# 門檻值 (Bias/Theta)：神經元活化的偏移量，分別對應 Y3, Y4, Y5
theta = [random.random() for _ in range(3)]

# ya：儲存神經元的實際輸出 [Y3, Y4, Y5]
ya = [0.0, 0.0, 0.0]

# alpha：學習率 (Learning Rate)，決定每次修正權重的步長
alpha = 0.2

# epoch：訓練代數計數器；MSElist：紀錄每一代的誤差，供後續繪圖
epoch = 0
MSElist = []

# ==========================================
# 2. 定義輔助函數 (決策邊界與繪圖)
# ==========================================

# 將神經元的判斷準則轉化為直線方程式 (w1*x1 + w2*x2 - theta = 0)
# 轉換後為：x2 = (-x1*w1 + theta) / w2
def line1(A): return (-A * w[0] + theta[0]) / w[1] # 神經元 Y3 的邊界線
def line2(A): return (-A * w[2] + theta[1]) / w[3] # 神經元 Y4 的邊界線

# ==========================================
# 3. 訓練目標設定
# ==========================================

print("=== MLP 邏輯閘學習程式 ===")
select = int(input("請選擇目標 (1:AND, 2:OR, 3:XOR): "))

if select == 1:
    yd = [0.0, 0.0, 0.0, 1.0] # AND 的標準答案
    colors = ['ro', 'ro', 'ro', 'bo'] # 前三點紅(0)，最後一點藍(1)
elif select == 2:
    yd = [0.0, 1.0, 1.0, 1.0] # OR 的標準答案
    colors = ['ro', 'bo', 'bo', 'bo']
else:
    yd = [0.0, 1.0, 1.0, 0.0] # XOR 的標準答案
    colors = ['ro', 'bo', 'bo', 'ro']

# ==========================================
# 4. 訓練核心迴圈
# ==========================================

target_accuracy = float(input("請輸入停止誤差門檻 (例如 0.001): "))
mse = 1.0 # 初始誤差設為 1

while (mse > target_accuracy and epoch < 100000):
    epoch += 1
    se = 0.0 # 本代的總平方誤差 (Sum of Errors)
    
    # 遍歷四組訓練資料
    for i in range(4):
        
        # --- (A) 前向傳播 (Forward Pass) ---
        # 計算隱藏層神經元 Y3 與 Y4 的輸出 (使用 Sigmoid 函數)
        # 公式: Output = 1 / (1 + exp(-(Σwx - theta)))
        ya[0] = 1.0 / (1.0 + math.exp(-(xs[i][0] * w[0] + xs[i][1] * w[1] - theta[0])))
        ya[1] = 1.0 / (1.0 + math.exp(-(xs[i][0] * w[2] + xs[i][1] * w[3] - theta[1])))
        
        # 計算輸出層神經元 Y5 的輸出
        ya[2] = 1.0 / (1.0 + math.exp(-(ya[0] * w[4] + ya[1] * w[5] - theta[2])))
        
        # --- (B) 誤差計算 (Error Calculation) ---
        e = yd[i] - ya[2] # 期望目標與實際輸出的差距
        
        # --- (C) 倒傳遞 (Backpropagation) ---
        
        # 1. 計算輸出層 (Y5) 的梯度誤差 ge5
        # 公式: 梯度 = 輸出 * (1 - 輸出) * 誤差  <-- 這是 Sigmoid 的導數應用
        ge5 = ya[2] * (1.0 - ya[2]) * e
        
        # 2. 更新輸出層權重 (w4, w5) 與門檻 (theta2)
        # 新權重 = 舊權重 + 學習率 * 輸入值 * 梯度
        w[4] += alpha * ya[0] * ge5
        w[5] += alpha * ya[1] * ge5
        theta[2] += (-alpha) * ge5 # 門檻值的更新方向與權重相反
        
        # 3. 計算隱藏層 (Y3, Y4) 的梯度誤差 (將輸出層誤差依權重比例回傳)
        ge3 = ya[0] * (1.0 - ya[0]) * ge5 * w[4]
        ge4 = ya[1] * (1.0 - ya[1]) * ge5 * w[5]
        
        # 4. 更新隱藏層權重 (w0~w3) 與門檻 (theta0, theta1)
        w[0] += alpha * xs[i][0] * ge3
        w[1] += alpha * xs[i][1] * ge3
        w[2] += alpha * xs[i][0] * ge4
        w[3] += alpha * xs[i][1] * ge4
        theta[0] += (-alpha) * ge3
        theta[1] += (-alpha) * ge4
        
        # 累加平方誤差
        se += e**2
    
    # 計算平均平方誤差 (Mean Square Error)
    mse = se / 4.0
    MSElist.append(mse)
    
    if epoch % 5000 == 0:
        print(f"Epoch: {epoch}, MSE: {mse:.6f}")

# ==========================================
# 5. 結果視覺化
# ==========================================

# 繪製分類結果與決策邊界
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i in range(4):
    plt.plot(xs[i][0], xs[i][1], colors[i], markersize=12) # 畫出原始點

x_axis = np.arange(-0.5, 1.5, 0.01)
plt.plot(x_axis, line1(x_axis), 'g--', label='Hidden Neuron Y3 boundary')
plt.plot(x_axis, line2(x_axis), 'm--', label='Hidden Neuron Y4 boundary')
plt.title(f"Result after {epoch} epochs")
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.legend()

# 繪製學習曲線
plt.subplot(1, 2, 2)
plt.plot(MSElist)
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error (MSE)")
plt.title("Learning Curve")

plt.tight_layout()
plt.show()
