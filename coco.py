import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
from sklearn.model_selection import train_test_split

# 设置文件路径
dataset_dir = 'dataset/myDataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
annotation_dir=os.path.join(dataset_dir, 'annotations')
os.makedirs(annotation_dir, exist_ok=True)
# 获取所有的图像文件名
image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# 7:3 划分训练集和测试集
train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.3, random_state=42)

def create_coco_annotations(image_filenames, dataset_type):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "modified", "supercategory": "none"}]
    }

    image_id = 0
    annotation_id = 0

    for img_filename in image_filenames:
        # 图像信息
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path)
        width, height = img.size  # 这应该是224, 224
        
        image_info = {
            "id": image_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        }
        coco_format["images"].append(image_info)

        # 标签信息（处理文件名不同的情况）
        label_filename = img_filename.replace('.jpg', '_label.png')
        label_path = os.path.join(labels_dir, label_filename)
        
        # 初始化标注信息
        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": {"counts": "", "size": [height, width]},  # 初始化为空的字典
            "area": 0,
            "bbox": [0, 0, 0, 0],
            "iscrowd": 0
        }

        # 检查标签文件是否存在
        if os.path.exists(label_path):
            label_img = Image.open(label_path)
            label_array = np.array(label_img)

            # 寻找非零区域 (即修改过的像素区域)
            if np.any(label_array == 255):  # 如果有修改过的像素
                fortran_mask = np.asfortranarray(label_array == 255)  # 255表示修改过的像素
                encoded_mask = mask_utils.encode(fortran_mask)
                area = int(mask_utils.area(encoded_mask))
                bbox = mask_utils.toBbox(encoded_mask).tolist()

                # 使用 counts 字段将掩码转换为字符串
                segmentation = encoded_mask['counts'].decode('utf-8') if isinstance(encoded_mask['counts'], bytes) else encoded_mask['counts']

                annotation_info.update({
                    "segmentation": {
                        "size": encoded_mask["size"],  # 保留 size
                        "counts": segmentation  # 确保 counts 是字符串或整数
                    },
                    "area": area,
                    "bbox": bbox
                })

        # 将标注信息添加到COCO格式的字典中
        coco_format["annotations"].append(annotation_info)
        annotation_id += 1
        image_id += 1

    # 保存为 COCO 格式的 JSON 文件
    output_json_path = os.path.join(annotation_dir, f'{dataset_type}.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)
    
    print(f"COCO 格式的 {dataset_type} JSON 文件已生成！")

# 生成训练集和测试集的 COCO JSON 文件
create_coco_annotations(train_filenames, 'train')
create_coco_annotations(test_filenames, 'valid')
