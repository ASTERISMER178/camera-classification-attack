import os
import random
import numpy as np
from PIL import Image

# 函数：生成与原始颜色不同的随机颜色
def generate_random_color(exclude_color):
    while True:
        random_color = np.random.randint(0, 256, size=3)
        if not np.array_equal(random_color, exclude_color):
            return random_color

# 设置文件路径
image_dir = 'dataset/images'
train_dir = 'dataset/myDataset/images'
label_dir = 'dataset/myDataset/labels'

# 创建输出文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
total_images = len(image_files)
selected_images = random.sample(image_files, int(total_images * 0.8))
print(total_images,int(total_images * 0.8))
# 处理每一张图片
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path)
    image_array = np.array(image)
    
    # 初始化二值标签图像
    label_array = np.zeros(image_array.shape[:2], dtype=np.uint8)
    
    if img_name in selected_images:
        # 生成10%-50%的随机百分数
        modify_percentage = random.uniform(0.1, 0.5)
        num_pixels = int(modify_percentage * image_array.size / 3)  # 3代表RGB三个通道
        
        # 随机选择要修改的像素位置
        height, width = image_array.shape[:2]
        y_indices = np.random.randint(0, height, num_pixels)
        x_indices = np.random.randint(0, width, num_pixels)
        
        # 修改随机选择的像素
        for y, x in zip(y_indices, x_indices):
            original_color = image_array[y, x]
            new_color = generate_random_color(original_color)
            image_array[y, x] = new_color  # 将像素修改为不同的随机颜色
            label_array[y, x] = 1  # 设置对应的二值标签
        
    # 保存修改后的图片
    modified_image = Image.fromarray(image_array)
    modified_image.save(os.path.join(train_dir, img_name))
    
    # 保存二值标签
    label_image = Image.fromarray(label_array * 255)  # 乘以255以便显示为白色
    label_image.save(os.path.join(label_dir, img_name.replace('.jpg', '_label.png')))

print("数据集创建完成！")
