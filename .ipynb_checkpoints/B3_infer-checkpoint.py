# 加载数据
train_data, train_label, test_data, test_filenames = load_data('./Recognize')
# 预处理和提取特征
test_data_preprocessed = []
test_features = []
for i in range(test_data.shape[0]):
    img = test_data[i]
    thresh, features = preprocess_feature(img)
    test_data_preprocessed.append(thresh)
    test_features.append(features)
test_data_preprocessed = np.array(test_data_preprocessed)
test_features = np.concatenate(test_features)
# 训练和评估模型
y_pred, report = train_and_evaluate(train_data, train_label, test_features, test_filenames)
print(report)

