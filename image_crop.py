import os
import numpy as np
from PIL import Image

# 设置文件路径
image_dir = 'dataset/images'
output_dir = 'dataset/crop_images'

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 裁剪函数
def limited_non_overlapping_crops(image, crop_size, max_crops=5):
    width, height = image.size
    crop_width, crop_height = crop_size
    
    # 计算可以裁剪的列数和行数
    num_crops_x = width // crop_width
    num_crops_y = height // crop_height
    
    cropped_images = []
    
    for i in range(num_crops_x):
        for j in range(num_crops_y):
            if len(cropped_images) >= max_crops:
                return cropped_images
            
            x = i * crop_width
            y = j * crop_height
            cropped_image = image.crop((x, y, x + crop_width, y + crop_height))
            cropped_images.append(cropped_image)
    
    return cropped_images

# 处理每一张图片
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path)
    
    # 裁剪成最多 5 张 224x224 像素的子图
    cropped_images = limited_non_overlapping_crops(image, (224, 224), max_crops=5)
    
    # 保存每个裁剪后的图片
    for idx, cropped_image in enumerate(cropped_images):
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_crop_{idx}.jpg")
        cropped_image.save(output_path)

print("图片裁剪完成！")
