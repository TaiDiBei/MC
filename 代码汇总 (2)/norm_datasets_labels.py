import os

from PIL import Image
from tqdm import tqdm


def normalize_bbox(width, height, x1, y1, w, h):
    """
    Normalize bounding box coordinates.
    Returns:
        tuple: 归一化后的矩形框坐标 (xmin, ymin, xmax, ymax)。
    """
    xmin = round(x1 / width, 4)
    ymin = round(y1 / height, 4)
    xmax = round(w / width, 4)
    ymax = round(h / height, 4)

    return [xmin, ymin, xmax, ymax]


path = "cup_data/train_yolo5/labels2/train"
image_list = os.listdir(path)
for il in tqdm(image_list):
    txt_path = os.path.join(path, il)
    # 打开txt文件
    with open(txt_path, 'r') as file:
        # 逐行读取文件内容
        rus = []
        for line in file:
            # 移除行末的换行符，并将每行内容按空格分割成列表
            line_data = line.strip().split()
            # 获取每行的后四个数字（假设每行都至少有五个数字）
            # last_four_numbers = line_data[1:]
            rus.append(line_data)
            # 打印后四个数字
    # 示例用法
    # 打开图像文件
    image_path = "cup_data/train_yolo5/images/train/" + il[:-4] + ".jpg"
    image = Image.open(image_path)

    # 获取图像的长和宽
    width, height = image.size
    tem = []
    for rs in range(len(rus)):
        x1 = abs(float(rus[rs][1])-float(rus[rs][3])) / 2
        y1 = abs(float(rus[rs][2])-float(rus[rs][4])) / 2
        w = abs(float(rus[rs][1])-float(rus[rs][3]))
        h = abs(float(rus[rs][2])-float(rus[rs][4]))

        # 归一化矩形框坐标
        normalized_bbox = normalize_bbox(width, height, x1, y1, w, h)
        normalized_bbox = [int(0)] + normalized_bbox
        tem = tem + normalized_bbox
    # 将变量写入txt文件，每五个一行，用空格分隔
    write_path = os.path.join("cup_data/train_yolo5/labels/train", f"{il}")
    with open(write_path, 'w') as file:
        for i in range(0, len(tem), 5):
            line = ' '.join(map(str, tem[i:i + 5])) + '\n'
            file.write(line)


