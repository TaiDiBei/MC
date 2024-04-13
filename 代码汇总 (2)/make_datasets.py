import os
from shutil import copyfile
from tqdm import tqdm

source_folder = "cup_data/train_yolo5/labels/train"  # 源文件夹路径
destination_folder = "cup_data/train_yolo5/images/train"  # 目标文件夹路径

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取源文件夹中的所有文件
file_list = os.listdir(source_folder)

for json_file in tqdm(file_list, desc="Copying files"):
    jpg_file = json_file[:-4] + ".jpg"
    source_file_path = os.path.join("cup_data/2_Train", jpg_file)
    destination_file_path = os.path.join(destination_folder, jpg_file)
    # 检查源文件是否存在，如果存在则复制
    if os.path.exists(source_file_path):
        copyfile(source_file_path, destination_file_path)
