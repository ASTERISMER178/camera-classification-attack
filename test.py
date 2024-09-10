import os
import shutil
import random

# 定义路径
images_dir = 'dataset/myDataset/images'
labels_dir = 'dataset/myDataset/labels'
train_img_dir = 'mmsegmentation/data/img_dir/train'
val_img_dir = 'mmsegmentation/data/img_dir/val'
train_label_dir = 'mmsegmentation/data/ann_dir/train'
val_label_dir = 'mmsegmentation/data/ann_dir/val'

# 创建目标目录
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有图片文件名
all_images = os.listdir(images_dir)
random.shuffle(all_images)

# 计算训练集和测试集的数量
train_size = int(len(all_images) * 0.7)
train_images = all_images[:train_size]
val_images = all_images[train_size:]

# 将文件复制到相应的目录
for img_name in train_images:
    img_path = os.path.join(images_dir, img_name)
    label_name = img_name.replace('.jpg', '_label.png')
    label_path = os.path.join(labels_dir, label_name)
    
    shutil.copy(img_path, train_img_dir)
    shutil.copy(label_path, train_label_dir)

for img_name in val_images:
    img_path = os.path.join(images_dir, img_name)
    label_name = img_name.replace('.jpg', '_label.png')
    label_path = os.path.join(labels_dir, label_name)
    
    shutil.copy(img_path, val_img_dir)
    shutil.copy(label_path, val_label_dir)

print('数据集划分完成。')
