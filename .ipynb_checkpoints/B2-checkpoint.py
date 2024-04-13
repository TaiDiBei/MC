# 导入必要的库
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

# 加载训练数据集
train_data = np.load("train_data.npy")
train_labels = np.load("train_labels.npy")

# 数据预处理
train_data = train_data.reshape(-1, 100, 100, 1)
train_data = train_data.astype('float32') / 255.
train_labels = train_labels.reshape(-1, 100, 100, 1)
train_labels = train_labels.astype('float32') / 255.

# 定义卷积神经网络模型
model = Sequential()
model.add(Conv2D(64, 3, activation='relu', padding='same', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(1, 3, activation='sigmoid', padding='same'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy')

# 训练模型
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# 加载测试数据集
test_data = np.load("test_data.npy")

# 数据预处理
test_data = test_data.reshape(-1, 100, 100, 1)
test_data = test_data.astype('float32') / 255.

# 对测试数据进行预测
prediction = model.predict(test_data)

# 显示测试结果
plt.imshow(prediction[0].reshape(100, 100))
plt.show()

