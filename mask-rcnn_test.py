import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2
import os

# 设置模型配置
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 设置为你的类别数量
cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")  # 加载训练好的模型
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测阈值
cfg.MODEL.DEVICE = "cuda"  # 如果有GPU, 使用GPU

# 创建推理器
predictor = DefaultPredictor(cfg)

# 推理的图片文件夹
test_images_dir = 'dataset/test'

# 保存推理结果的文件夹
output_dir = "output/inference_results"
os.makedirs(output_dir, exist_ok=True)

# 对每张图片进行推理
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    image = cv2.imread(image_path)
    
    # 推理
    outputs = predictor(image)
    
    # 可视化并保存结果
    v = Visualizer(image[:, :, ::-1], metadata={}, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # 保存推理结果图片
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, v.get_image()[:, :, ::-1])

print("推理完成，结果已保存到 output/inference_results 文件夹。")
