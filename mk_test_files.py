import os

# 定义test文件夹路径
test_folder = 'test'

# 定义要保存路径的文件名
output_file = 'test_files'

# 支持的图像文件扩展名
image_extensions = ('.tif', '.jpg', '.jpeg', '.png', '.bmp')

# 打开输出文件准备写入
with open(output_file, 'w') as f:
    # 递归遍历test文件夹
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(image_extensions):  # 检查文件扩展名
                # 获取相对路径
                relative_path = os.path.join(root, file)
                # 将路径写入文件
                f.write(relative_path + '\n')

print(f"图片路径已保存到 {output_file} 文件中")
