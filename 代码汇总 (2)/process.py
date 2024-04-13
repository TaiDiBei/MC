import os
import json

from tqdm import tqdm

folder_path = "data/2_Train"  # 替换为你的文件夹路径

# 获取文件夹中的所有文件
file_list = os.listdir(folder_path)

# 筛选出json文件
json_files = [file for file in file_list if file.endswith('.json')]

# 读取每个json文件并处理内容
for json_file in tqdm(json_files):
    json_file_path = os.path.join(folder_path, json_file)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        img_name = data['img_name']
        annotations = data['ann']

        # 创建并写入txt文件
        txt_file_path = os.path.join("test001", f"{img_name}.txt")
        with open(txt_file_path, 'w') as txt_file:
            for ann in annotations:
                # 将annotation列表的最后一个值去除，并转换为字符串
                ann_str = ' '.join(map(str, ann[:-1]))
                # 将最后的1放在第一个位置
                ann_str = f"1 {ann_str}"
                # 将字符串写入txt文件
                txt_file.write(ann_str + '\n')
