# 使用第二问训练好的yolo，我们将附件四的五十张图像进行甲骨文的自动识别与分割
# 将自动识别、分割后的结果保存下来进行文字识别
# 文字识别代码如下：
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm

# 检查CUDA是否可用，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 构建自定义数据集
class CustomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# 转换图像
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_file_dir = "test_image_2"# 将需要预测的图像放在此文件夹
test_file_path = os.listdir(test_file_dir)
# 给每个文件路径添加文件夹路径
test_file_paths = [os.path.join(test_file_dir, file_name) for file_name in test_file_path]
# 数据加载
test_dataset = CustomDataset(test_file_paths,transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载预训练的InceptionV3模型
model = models.inception_v3(pretrained=False)
model.aux_logits = False  # 禁用辅助输出

# 修改最后一层全连接层以适应多标签分类任务
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 76),
    nn.Sigmoid()  # 多标签分类使用Sigmoid激活函数
)
# 加载微调后的模型权重
model_path = 'ckpt/inceptionv3_ft_10_f1_0.9962_acc_0.9940.pth'  # 替换为你的模型权重文件路径
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()
with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        predicted = outputs > 0.5  # 使用阈值 0.5 来确定标签
        # 获取值为 True 的元素的索引
        indices = torch.nonzero(predicted, as_tuple=False)
    print("predicted", predicted)
    print(type(predicted))
    print(predicted.shape)
    print(indices)
