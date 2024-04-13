import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# 检查CUDA是否可用，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 构建自定义数据集
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# 超参数
batch_size = 32
num_epochs = 50
learning_rate = 0.001
num_classes = 76  # 假设有10个类别

# 转换图像
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_file_dir = "q4_data/train/image"
train_file_paths = os.listdir(train_file_dir)
# 给每个文件路径添加文件夹路径
train_file_paths = [os.path.join(train_file_dir, file_name) for file_name in train_file_paths]
# 从JSON文件中读取数据
with open("q4_train_inception.json", "r") as json_file:
    train_labels = json.load(json_file)
# 数据加载
train_dataset = CustomDataset(train_file_paths, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载预训练的InceptionV3模型
model = models.inception_v3(pretrained=True)
model.aux_logits = False  # 禁用辅助输出
# # 冻结模型参数
# for param in model.parameters():
#     param.requires_grad = False

# 修改最后一层全连接层以适应多标签分类任务
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, num_classes),
    nn.Sigmoid()  # 多标签分类使用Sigmoid激活函数
)

model = model.to(device)
# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 训练模型
total_step = len(train_loader)
for epoch in tqdm(range(num_epochs)):
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 调整学习率
    scheduler.step()

    # 评估循环
    model.eval()
    with torch.no_grad():
        # 初始化预测和标签列表
        all_preds = []
        all_labels = []
        correct_predictions = 0
        total_predictions = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images)
            predicted = outputs > 0.5
            # 计算准确率
            correct_predictions += (predicted == labels.byte()).all(1).sum().item()
            total_predictions += labels.size(0)
            # 收集预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        accuracy = correct_predictions / total_predictions
        f1 = f1_score(all_labels, all_preds, average='micro')
        print('Epoch [{}/{}], Accuracy: {:.4f}, F1 Score: {:.4f}'.format(epoch + 1, num_epochs, accuracy, f1))

    # 保存模型
    torch.save(model.state_dict(), f'ckpt/inceptionv3_ft_{epoch}_f1_{f1:.4f}_acc_{accuracy:.4f}.pth')

