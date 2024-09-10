# _*_ coding : UTF-8 _*_
# @Time : 2024/7/25 20:46
# @Author : GYH
# @File : writeFile_train
# @Project :
import os

def create_file_list(data_dir, output_file):
    with open(output_file, 'w') as f:
        # 遍历数据集文件夹中的所有子文件夹
        for class_index, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                # 遍历子文件夹中的所有图片文件
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):  # 只处理图片文件
                        img_path = os.path.join(data_dir, class_name, img_name).replace("\\", "/")
                        # 写入相对路径和类别索引
                        f.write(f"{img_path} {class_index}\n")

# 定义数据集文件夹路径
train_data_dir = 'dataset/archive/train'
val_data_dir = 'dataset/archive/test'

# 定义输出文件路径
train_output_file = 'train_files'
val_output_file = 'val_files'

# 创建文件列表
create_file_list(train_data_dir, train_output_file)
create_file_list(val_data_dir, val_output_file)

print("文件列表已创建完毕。")
