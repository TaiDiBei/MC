import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
def load_data(path):
    '''
    加载训练集和测试集数据
    :param path: 数据所在文件夹路径
    :return: 训练集数据和标签，测试集数据和文件名列表
    '''
    train_data = []
    train_label = []
    test_data = []
    test_filenames = []
    for filename in os.listdir(path):
        # 训练集
        if 'train' in filename:
            with open(os.path.join(path, filename), 'rb') as f:
                data = np.load(f)
                train_data.append(data['arr_0'])
                train_label.append(data['arr_1'])
        # 测试集
        elif 'test' in filename:
            with open(os.path.join(path, filename), 'rb') as f:
                data = np.load(f)
                test_data.append(data['arr_0'])
                test_filenames.append(data['arr_1'])
    return np.concatenate(train_data), np.concatenate(train_label), np.concatenate(test_data), test_filenames
def preprocess_feature(img):
    '''
    图像预处理和特征提取
    :param img: 输入图像
    :return: 预处理后的图像和特征
    '''
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应二值化
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # 提取轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 计算图像面积
    img_area = img.shape[0] * img.shape[1]
    # 只保留面积大于0.5%图像面积的轮廓
    contours = [c for c in contours if cv2.contourArea(c) > 0.005 * img_area]
    # 计算每个轮廓的最小外接矩形
    rects = [cv2.boundingRect(c) for c in contours]
    # 提取特征
    features = []
    for rect in rects:
        x, y, w, h = rect
        roi = img[y:y + h, x:x + w]
        # 缩放大小为32x32
        roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        # 将图像转换为一维向量
        roi = roi.reshape(1, -1)
        # 归一化
        roi = roi / 255.0
        # 添加到特征列表中
        features.append(roi)
    return thresh, features
def train_and_evaluate(train_data, train_label, test_data, test_filenames):
    '''
    训练和评估模型
    :param train_data: 训练集数据
    :param train_label: 训练集标签
    :param test_data: 测试集数据
    :param test_filenames: 测试集文件名列表
    :return: 预测结果和评估报告
    '''
    # 拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
    # 定义模型，使用SVM分类器
    model = SVC(C=1.0, kernel='linear', gamma='auto')
    # 训练模型
    model.fit(X_train, y_train)
    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    # 在测试集上进行预测
    y_pred = model.predict(test_data)
    # 将预测结果保存到Excel文件中
    df = pd.DataFrame({'file_name': test_filenames, 'predict': y_pred})
    df.to_excel('Test_results.xlsx', index=False)
    return y_pred, classification_report(y_val, y_pred)

