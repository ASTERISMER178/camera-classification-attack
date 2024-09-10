### 相机分类攻击
我们希望通过白盒攻击的方法实现可控地让设备分类模型分类错误并分类到我们预期地类别上去。以往的分类攻击是在原图像上添加一个噪音，以此来干扰模型判断，但是这种方法无法控制模型分类地结果。我们希望设计这样一个网络结构，他拥有两个loss，一个loss随着伪造后的图像与原来的图像内容差别越小而越小，另一个loss随着伪造后的图像经过模型的分类结果置信度与我们预期的结果越接近越小，目前正在讨论网络结构的设计和训练的策略。

### 相机分类codebase说明：

#### 文件结构说明

首先codebase目录如下：

```
-- 项目路径
	-- camera(文件夹)
	-- train(文件夹)
	-- test(文件夹)
	-- model(文件夹)
	-- res_model(文件夹)
	-- train.py
	-- predict.py
```

camera文件夹下保存模型的各种py文件包括数据集定义、数据处理定义、模型方法定义、运行时定义、训练方法等等

train文件夹下保存训练的图片，对于真假相机分类任务，训练集应该就两类，所以train文件夹下应该只有real和fake两个文件夹，分别存放真实图片和造假图片。

test文件夹下保存需要推理的图片，图片格式为tif。

model文件夹下存放预训练的模型

res_model文件夹下存放预训练的模型



#### 运行说明

***请注意想要跑通该模型，请确保自己的python版本为3.9，torch版本为2.1，torchvision版本为0.16，transformers版本为4.28***

train.py是启动训练的代码，如果你想要进行训练，直接call下面代码

```cmd
python train.py --train_files train_files --val_files val_files --pretrained_weights_path model/resnet50-19c8e357.pth --batch_size 128 --model_save_path model.pth
```

其中train_files这个文件存放训练集图片的路径和类别索引，val_files这个文件存放验证集图片的路径和类别索引。

predict.py是启动推理的代码，如果你想要进行推理，请直接call下面代码

```
python predict.py --test_files test_files --batch_size 128 --model_path res_model/model.pth --submit_path results/plots/submit.csv
```

其中test_files这个文件存放训练集图片的路径和类别索引



#### 代码说明

```
-- camera
	-- augmentation.py
	-- dataset.py
	-- model.py
	-- postprocessing.py
	-- scheduler.py
	-- train_utils.py
```



**augmentation.py**是对训练集进行预处理和数据增强，需要改动的只有裁剪尺寸***CROP_SIZE***。

**dataset.py**是对数据集的格式进行定义，不需要改动。

**model.py**定义模型的训练和预测方法以及loss和acc的计算。需要改动的只有分类数量***NUM_CLASSES***。

**postprocessing.py**定义数据集的详细情况，需要改动的有类别名称列表***CLASSES***和分类数量***NUM_CLASSES***。

**scheduler.py**定义训练时的日志等运行情况，不需要改动。

**train_utils.py**定义训练的过程，你可在这里画出loss和acc的图。需要改动的只有训练批次***NUM_EPOCH***



#### 资料说明

伪造工具Targeted adversarial attacks with Keras and TensorFlow仓库地址：[https://github.com/bcmi/libcom?tab=readme-ov-file](https://pyimagesearch.com/2020/10/26/targeted-adversarial-attacks-with-keras-and-tensorflow/)


