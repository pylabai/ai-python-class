import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定義輸入資料（訓練資料）
x_train = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# 預設訓練目標
y_train = [0.0, 1.0, 1.0, 0.0]

# 選擇訓練目標
select = eval(input('Please select training target: AND==> 1, OR==> 2, XOR==> 3\n'))

if select == 1:
    y_train = [0.0, 0.0, 0.0, 1.0] # AND
if select == 2:
    y_train = [0.0, 1.0, 1.0, 1.0] # OR
if select == 3:
    y_train = [0.0, 1.0, 1.0, 0.0] # XOR

# 神經網路參數設定
In_neuron = 2
Hiden1_neuron = 10
Hiden2_neuron = 10
Out_neuron = 1 # 註：原始碼 30-31 行同時定義了 Out_neuron 與 Out_type
Out_type = 2

# 設定訓練次數
TC = eval(input('How many training cycles: '))

# 設定神經網路架構
# 包含兩個隱藏層，激活函數使用 tanh
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(Hiden1_neuron, activation=tf.nn.tanh, input_dim=In_neuron),
    tf.keras.layers.Dense(Hiden2_neuron, activation=tf.nn.tanh),
    # 輸出層：使用 softmax 進行分類
    tf.keras.layers.Dense(Out_type, activation=tf.nn.softmax)
])

# 最佳化設定
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型 [cite: 37, 38]
history = model.fit(x_train, y_train,
                    epochs=TC,
                    batch_size=4)

# 結果預測 [cite: 43]
x_test = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
print(x_test)
y_test = model.predict(x_test)
print('y_test=', y_test)

# 透過 numpy.argmax 顯示分類結果
print("學習後之答案為:", 
      np.argmax(y_test[0]), np.argmax(y_test[1]),
      np.argmax(y_test[2]), np.argmax(y_test[3]))

# 繪製誤差曲線
print('****誤差曲線****')
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss/accuracy')
# plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()
