# 导入需要的库
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

# 加载训练好的甲骨文图像分割模型
model = load_model('甲骨文图像分割模型.h5')

# 加载甲骨文文字形的训练集
train_data = pd.read_csv('甲骨文文字形.csv')

# 定义一个函数用于对甲骨文图像进行预处理
def preprocess(image):
    # 将图像转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对图像进行二值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # 对二值化图像进行腐蚀操作，去除图像中的噪声
    kernel = np.ones((3,3),np.uint8)
    eroded = cv2.erode(thresh,kernel,iterations = 1)
    # 对腐蚀后的图像进行闭运算，填补图像中的空洞
    kernel = np.ones((3,3),np.uint8)
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    # 对闭运算后的图像进行开运算，去除图像中的小细节
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    # 返回预处理后的图像
    return opened

# 定义一个函数用于对甲骨文图像进行单字分割
def segment(image):
    # 对图像进行预处理
    image = preprocess(image)
    # 将图像转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对图像进行二值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # 使用甲骨文图像分割模型进行分割
    segments = model.predict(thresh.reshape(1, 64, 64, 1))
    # 将分割结果转为二值化图像
    segments[segments < 0.5] = 0
    segments[segments >= 0.5] = 1
    # 根据分割结果进行图像分割
    result = cv2.bitwise_and(thresh, thresh, mask=segments.reshape(64, 64))
    # 返回分割后的图像
    return result

# 定义一个函数用于识别甲骨文文字
def recognize(image):
    # 对图像进行单字分割
    result = segment(image)
    # 将分割后的图像转为灰度图
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # 将灰度图像转为数组
    data = np.array(gray).reshape(64, 64, 1)
    # 使用训练好的模型进行预测
    prediction = model.predict(data.reshape(1, 64, 64, 1))
    # 将预测结果转为字母
    result = train_data['label'][np.argmax(prediction)]
    # 返回识别结果
    return result

# 定义一个函数用于识别测试集中的图像
def recognize_test(image):
    # 对图像进行单字分割
    result = segment(image)
    # 将分割后的图像转为灰度图
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # 将灰度图像转为数组
    data = np.array(gray).reshape(64, 64, 1)
    # 使用训练好的模型进行预测
    prediction = model.predict(data.reshape(1, 64, 64, 1))
    # 将预测结果转为字母
    result = train_data['label'][np.argmax(prediction)]
    # 返回识别结果
    return result

# 对测试集中的图像进行识别
test_results = pd.DataFrame(columns=['img_name', 'recognize_result'])
for i in range(1, 51):
    # 读取图像
    image = cv2.imread('Test/' + str(i) + '.jpg')
    # 对图像进行识别
    result = recognize_test(image)
    # 添加识别结果到test_results中
    test_results = test_results.append({'img_name': str(i), 'recognize_result': result}, ignore_index=True)

# 将识别结果保存到Excel中
test_results.to_excel('Test_results.xlsx', index=False)

