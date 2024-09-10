import torch
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
import os
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for training.")
else:
    print("CUDA is not available. Using CPU for training.")

# 设置数据集路径
dataset_dir = 'dataset/myDataset'
images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')

train_annotations_file = os.path.join(annotations_dir, 'train.json')
valid_annotations_file = os.path.join(annotations_dir, 'valid.json')

# 注册COCO格式的数据集
register_coco_instances("my_dataset_train", {}, train_annotations_file, images_dir)
register_coco_instances("my_dataset_valid", {}, valid_annotations_file, images_dir)

# 获取配置文件并设置参数
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_valid",)
cfg.DATALOADER.NUM_WORKERS = 32
cfg.MODEL.WEIGHTS = "detectron2/weights/model_final_f10217.pkl"  # 使用预训练模型
cfg.SOLVER.IMS_PER_BATCH = 32  # 根据显存调整 batch size
cfg.SOLVER.BASE_LR = 0.00025  # 初始学习率
cfg.SOLVER.MAX_ITER = 3000  # 训练迭代次数，根据需求调整
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 每张图片的ROIs数量
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别 (modified)
cfg.INPUT.MASK_FORMAT = 'bitmask'

# 设置模型设备为cuda
cfg.MODEL.DEVICE = "cuda"  # 使用GPU进行训练

# 输出模型保存路径
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 训练模型
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("模型训练完成！")

# 评估模型
evaluator = detectron2.evaluation.COCOEvaluator("my_dataset_valid", cfg, False, output_dir="./output/")
val_loader = detectron2.data.build_detection_test_loader(cfg, "my_dataset_valid")
metrics = detectron2.evaluation.inference_on_dataset(trainer.model, val_loader, evaluator)


# 从训练存储中获取 loss 历史
history = trainer.storage.history("total_loss")
loss_history = history.values()  # 获取所有历史记录的值

# 提取损失值和迭代次数
loss_values = [x[0] for x in loss_history]  # 假设每个记录是一个包含损失值的列表
iterations = list(range(len(loss_values)))  # 创建迭代次数的列表

# 创建存放图像的文件夹
plots_dir = "results/plots"
os.makedirs(plots_dir, exist_ok=True)

# 绘制损失图
plt.figure()
plt.plot(iterations, loss_values, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.legend()

# 保存损失图
plt.savefig(os.path.join(plots_dir, "mask-rcnn_loss.png"))

print("Loss 图已经生成并保存在 results/plots 文件夹中！")
