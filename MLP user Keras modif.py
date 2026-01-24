import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 訓練資料初始化
# ---------------------------------------------------------

# 定義輸入資料 x_train，包含 4 組座標點，代表邏輯閘的所有可能輸入。
# 在二維空間中，這些點分別位於 (0,0), (0,1), (1,0) 與 (1,1)，是測試非線性分類的經典數據。
x_train = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

# y_train 為預期的標籤值，初始設定為 XOR 邏輯（相同為 0，不同為 1）。
# 標籤的順序必須嚴格對應 x_train 的輸入順序，否則模型將無法正確學習特徵。
y_train = [0.0, 1.0, 1.0, 0.0]

# 提供使用者互動介面，根據輸入數字切換不同的邏輯運算目標。
# 這段程式碼展示了同一個模型架構，透過更換標籤（Label）就能學習完全不同的邏輯規則。
select = eval(input('Please select training target: AND==> 1, OR==> 2, XOR==> 3\n'))

# 根據使用者的選擇，動態重寫 y_train 的內容。
# AND 要求兩者皆為 1 才輸出 1；OR 要求任一者為 1 即輸出 1；XOR 則是最難的「互斥或」邏輯。
if select == 1:
    y_train = [0.0, 0.0, 0.0, 1.0] # AND 標籤
if select == 2:
    y_train = [0.0, 1.0, 1.0, 1.0] # OR 標籤
if select == 3:
    y_train = [0.0, 1.0, 1.0, 0.0] # XOR 標籤

# ---------------------------------------------------------
# 2. 神經網路超參數與架構設定
# ---------------------------------------------------------

# 定義各層神經元的數量：In 代表輸入特徵數，Hidden 代表隱藏層中負責特徵提取的單元數。
# 這裡將隱藏層設為 10 個神經元，遠超理論所需的數量，是為了確保能快速收斂並應付 XOR 的非線性。
In_neuron = 2
Hiden1_neuron = 10
Hiden2_neuron = 10

# Out_type 定義為 2，代表模型最終會輸出兩個機率值（屬於類別 0 的機率與類別 1 的機率）。
# 這種「一對多」的分類方式，能比單一數值輸出提供更穩定的學習梯度。
Out_type = 2

# 接收訓練次數 (Epochs)，這代表神經網路要完整閱讀並訓練這四筆資料幾次。
# 訓練次數太少會導致「欠擬合 (Underfitting)」，太多則可能浪費計算資源。
TC = eval(input('How many training cycles: '))

# 使用 Sequential API 建立模型，這是一種線性的層堆疊方式，非常適合中小型網路。
# 每一層 Dense 代表全連接層，前一層的每個神經元都會與後一層的每個神經元相連。
model = tf.keras.models.Sequential([
    # 第一層隱藏層：接收 2 個輸入，並使用 tanh 作為激活函數。
    # tanh (雙曲正切) 能將輸入壓縮至 -1 到 1，其梯度在 0 附近比 sigmoid 更強，收斂較快。
    tf.keras.layers.Dense(Hiden1_neuron, activation=tf.nn.tanh, input_dim=In_neuron),
    
    # 第二層隱藏層：進一步將第一層提取的非線性特徵進行組合。
    # 增加隱藏層能讓模型學習更高維度的空間變換，這是解決 XOR 問題的關鍵。
    tf.keras.layers.Dense(Hiden2_neuron, activation=tf.nn.tanh),
    
    # 輸出層：使用 Softmax 激活函數，將結果轉化為總和為 1 的機率分佈。
    # 這層的神經元數為 Out_type (2)，對應分類任務中的兩個不同類別。
    tf.keras.layers.Dense(Out_type, activation=tf.nn.softmax)
])

# ---------------------------------------------------------
# 3. 編譯與模型訓練
# ---------------------------------------------------------

# 設定編譯參數：optimizer 使用 'adam'，它能根據訓練進度自動調整學習率（Adaptive LR）。
# loss 選用 'sparse_categorical_crossentropy'，這專門用於整數標籤的多分類損失計算。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 執行 fit 訓練：將訓練資料與標籤餵入模型，並設定 batch_size 為 4。
# batch_size=4 意味著每次更新權重前，會同時參考所有四筆訓練資料的誤差總和。
history = model.fit(x_train, y_train,
                    epochs=TC,
                    batch_size=4)

# ---------------------------------------------------------
# 4. 預測與結果顯示
# ---------------------------------------------------------

# 準備測試資料：將原本的輸入轉換為 NumPy 陣列，以符合 TensorFlow 的張量運算格式。
# predict 函數會回傳模型對於每個輸入點預測的「分類機率值」。
x_test = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_test = model.predict(x_test)

# 使用 np.argmax 找出機率最高的索引值，這就是神經網路最終判斷的類別。
# 例如 [0.1, 0.9] 的最大索引是 1，代表預測結果為「真 (True)」。
print("學習後之答案為:", 
      np.argmax(y_test[0]), np.argmax(y_test[1]),
      np.argmax(y_test[2]), np.argmax(y_test[3]))

# ---------------------------------------------------------
# 5. 視覺化分析
# ---------------------------------------------------------

# 繪製 Loss (損失) 曲線，這是一張判斷訓練是否成功的「成績單」。
# 橫軸為訓練次數，縱軸為誤差值；理想的曲線應從高處迅速下降並趨於平緩。
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
