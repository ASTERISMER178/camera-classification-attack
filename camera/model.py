import os

from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image

from .scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, log_loss
from torch.autograd import Variable
from torchvision.models import resnet50
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import matplotlib as mpl

from captum.attr import LayerConductance, LayerIntegratedGradients, LayerDeepLift, LayerGradientShap, visualization as viz 

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

mpl.use('Agg')
NUM_CLASSES = 10

class SerializableModule(nn.Module):
    def __init__(self):
        super(SerializableModule, self).__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class Model(SerializableModule):
    def __init__(self, weights_path=None):
        super(Model, self).__init__()

        model = resnet50()
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)

        num_features = model.fc.in_features
        model.fc = nn.Dropout(0.0)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

        self._model = model

    def forward(self, x):
        x = self._model(x)
        x = self.fc(x)
        return x


class CameraModel(object):
    def __init__(self, model=None):
        if model is None:
            model = Model()
        self._torch_single_model = model
        self._torch_model = nn.DataParallel(self._torch_single_model).cuda()

        self._optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=0.0001)
        self._scheduler = ReduceLROnPlateau(self._optimizer, factor=0.5, patience=5,
                                            min_lr=1e-6, epsilon=1e-5, verbose=1, mode='min')
        self._optimizer.zero_grad()
        self._criterion = nn.CrossEntropyLoss()

        # 用于存储 conv1 层输出的特征向量
        self.activations = []

    def scheduler_step(self, loss, epoch):
        self._scheduler.step(loss, epoch)

    def enable_train_mode(self):
        self._torch_model.train()

    def enable_predict_mode(self):
        self._torch_model.eval()

    def train_on_batch(self, X, y):
        X = X.cuda()#async=True
        y = y.cuda()#async=True
        X = Variable(X, requires_grad=False)
        y = Variable(y, requires_grad=False)

        y_pred = self._torch_model(X)

        loss = self._criterion(y_pred, y)
        loss.backward()

        self._optimizer.step()
        self._optimizer.zero_grad()

        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred.cpu().data.numpy()

    def predict_on_batch(self, X):
        X = X.cuda()##async=True
        X = Variable(X, requires_grad=False, volatile=True)
        y_pred = self._torch_model(X)
        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred.cpu().data.numpy()

    # def predict_on_batch(self, X):
    #     # Register hooks only for specific layers
    #     activations = {}
        
    #     def hook_fn(name):
    #         def hook(module, input, output):
    #             print(f"Hook executed for {name}, output shape: {output.shape}")
    #             activations[name] = output
    #         return hook
        
    #     # Layers of interest
    #     target_layers = {
    #         'conv1': self._torch_model.module._model.conv1,
    #         'maxpool': self._torch_model.module._model.maxpool,
    #         'layer1': self._torch_model.module._model.layer1,
    #         'layer2': self._torch_model.module._model.layer2,
    #         'layer3': self._torch_model.module._model.layer3,
    #         'layer4': self._torch_model.module._model.layer4,
    #         'avgpool': self._torch_model.module._model.avgpool,
    #         'fc': self._torch_model.module.fc
    #     }
        
    #     # Register hooks for target layers
    #     hooks = []
    #     for name, layer in target_layers.items():
    #         hooks.append(layer.register_forward_hook(hook_fn(name)))
        
    #     # Forward pass
    #     X = X.cuda()##async=True
    #     X = Variable(X, requires_grad=False, volatile=True)
    #     print("Start forward pass")
    #     y_pred = self._torch_model(X)
    #     print("End forward pass")
    #     y_pred = F.softmax(y_pred, dim=-1)
        
    #     # Remove hooks
    #     for hook in hooks:
    #         hook.remove()
        
    #     # Return activations from target layers and predictions
    #     return activations, y_pred.cpu().data.numpy()

    def _register_hooks(self):
        """注册钩子以获取 conv1 层的输出"""
        def hook_fn(module, input, output):
            # 将每次的输出存储起来
            self.activations.append(output.cpu().data.numpy())

        # 获取模型的conv1层并注册钩子
        self._torch_single_model._model.fc.register_forward_hook(hook_fn)
    

    def fit_generator(self, generator):
        self.enable_train_mode()
        mean_loss = None
        mean_accuracy = None
        start_time = time.time()
        for step_no, (X, y) in enumerate(generator):
            y_pred = self.train_on_batch(X, y)

            y = y.cpu().numpy()

            accuracy = accuracy_score(y, y_pred.argmax(axis=-1))
            loss = log_loss(y, y_pred, labels=list(range(10)))

            if mean_loss is None:
                mean_loss = loss

            if mean_accuracy is None:
                mean_accuracy = accuracy

            mean_loss = 0.9 * mean_loss + 0.1 * loss
            mean_accuracy = 0.9 * mean_accuracy + 0.1 * accuracy

            cur_time = time.time() - start_time
            print("[{3} s] Train step {0}. Loss {1}. Accuracy {2}".format(step_no, mean_loss, mean_accuracy, cur_time))

    # def predict_generator(self, generator,labels):
    #     self.enable_predict_mode()
    #     result = []
    #     # activation_diffs = []
    #     # activation_stats = []
    #     start_time = time.time()
    #     # cam_extractor = SmoothGradCAMpp(self._torch_model)

    #     for step_no, X in enumerate(generator):
    #         print(type(X))
    #         if isinstance(X, (tuple, list)):
    #             X = X[0]

    #         y_pred = self.predict_on_batch(X)
    #         print(type(y_pred))
    #         result.append(y_pred)
    #         print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

    #     return np.concatenate(result)


    def save(self, filename):
        self._torch_single_model.save(filename)

    @staticmethod
    def load(filename):
        model = Model()
        model.load(filename)
        return CameraModel(model=model)


    def plot_activation_pca(self, labels):
        """对 conv1 特征进行降维并绘制点云图"""
        activations = np.concatenate(self.activations, axis=0)  # 将所有批次的 activations 合并
        activations_flat = activations.reshape(activations.shape[0], -1)  # 将每个 feature map 拉平成一维

        # 使用 PCA 或者 t-SNE 进行降维
        pca = PCA(n_components=2)  # 可以换成 TSNE(n_components=2) 来使用 t-SNE
        reduced_features = pca.fit_transform(activations_flat)

        # 绘制二维点云图
        plt.figure(figsize=(10, 8))
        for i in range(NUM_CLASSES):
            indices = np.where(np.array(labels) == i)
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Class {i}')

        plt.legend()
        plt.title('PCA of fc features')
        plt.savefig("results/plots/fc_pca_scatter.png")
        plt.close
    

    # def plot_activation_pca(self, labels):
    #     """对 conv1 特征进行降维并绘制 3D 点云图"""
    #     activations = np.concatenate(self.activations, axis=0)  # 将所有批次的 activations 合并
    #     activations_flat = activations.reshape(activations.shape[0], -1)  # 将每个 feature map 拉平成一维

    #     # 使用 PCA 进行降维到 3 维
    #     pca = PCA(n_components=3)  # 将 PCA 降维到 3 维
    #     reduced_features = pca.fit_transform(activations_flat)

    #     # 绘制 3D 点云图
    #     fig = plt.figure(figsize=(12, 10))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # 绘制每个类别的点
    #     for i in range(NUM_CLASSES):
    #         indices = np.where(np.array(labels) == i)
    #         ax.scatter(reduced_features[indices, 0], 
    #                 reduced_features[indices, 1], 
    #                 reduced_features[indices, 2], 
    #                 label=f'Class {i}', 
    #                 s=10)
        
    #     ax.set_xlabel('PCA Component 1')
    #     ax.set_ylabel('PCA Component 2')
    #     ax.set_zlabel('PCA Component 3')
    #     ax.set_title('PCA of fc features in 3D')
    #     ax.legend()
        
    #     plt.savefig("fc_pca_scatter_3d.png")
    #     plt.close()
    #     print("PCA 3D scatter plot saved as 'fc_pca_scatter_3d.png'")


    def predict_generator(self, generator, labels):
        self.enable_predict_mode()
        result = []
        self.activations = []  # 清空之前的激活值
        self._register_hooks()  # 注册钩子

        start_time = time.time()
        # print(labels)
        # 遍历生成器，获取预测和 conv1 层特征向量
        for step_no, X in enumerate(generator):
            if isinstance(X, (tuple, list)):
                X = X[0]
            print(type(X))
            y_pred = self.predict_on_batch(X)
            result.append(y_pred)
            print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

        self.plot_activation_pca(labels)

        # 合并预测结果
        return np.concatenate(result)
        



    # def predict_generator(self, generator, labels):
    #     self.enable_predict_mode()
    #     result = []
    #     start_time = time.time()

    #     layer_outputs = {layer: [] for layer in ['conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']}
    #     # print(layer_outputs)
    #     print(labels)
    #     for step_no, X in enumerate(generator):
    #         if isinstance(X, (tuple, list)):
    #              X = X[0]

    #         activations, y_pred = self.predict_on_batch(X)
    #         print(type(activations['conv1']))
    #         result.append(y_pred)
    #         # Append layer outputs and labels
    #         for layer_name, output in activations.items():
    #             # print(layer_name,output)
    #             if layer_name not in layer_outputs:
    #                 layer_outputs[layer_name] = []
    #             layer_outputs[layer_name].append(output.cpu().detach().numpy())
            
    #         print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))
        
    #     # Convert lists to numpy arrays
    #     for layer_name in layer_outputs:
    #         layer_outputs[layer_name] = np.concatenate(layer_outputs[layer_name], axis=0)
        
    #     # Plot PCA results
    #     for layer_name, output in layer_outputs.items():
    #         # Flatten the output if necessary
    #         output_flat = output.reshape(output.shape[0], -1)
            
    #         # Apply PCA
    #         pca = PCA(n_components=2)
    #         reduced_output = pca.fit_transform(output_flat)
            
    #         # Plot
    #         plt.figure(figsize=(10, 8))
    #         print(reduced_output.shape)
    #         scatter = plt.scatter(reduced_output[:, 0], reduced_output[:, 1], c=labels, cmap='tab10', s=10)
    #         plt.colorbar(scatter, ticks=range(NUM_CLASSES), label='Class')
    #         plt.title(f'PCA of {layer_name}')
    #         plt.xlabel('PCA 1')
    #         plt.ylabel('PCA 2')
    #         plt.savefig(f'results/plots/{layer_name}_pca.png')
    #         plt.close()

    #     print("PCA and plotting completed.")
    #     return np.concatenate(result)


    # def predict_generator(self, generator):
    #     self.enable_predict_mode()
    #     result = []
    #     activation_diffs = []
    #     activation_stats = []
    #     start_time = time.time()
    #     # cam_extractor = SmoothGradCAMpp(self._torch_model)



        #创建类激活图
        #     # Generate and save class activation maps for each image in the batch
        #     for i, img_tensor in enumerate(X):
        #         if i % 11 == 0:
        #             idx = (i + 1) // 11
        #             img_tensor = img_tensor.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        #             # Get model output for the single image
        #             out = self._torch_model(img_tensor)
                    
        #             class_idx = out.argmax(dim=-1).item()  # Get predicted class index

        #             # Generate CAM
        #             activation_map = cam_extractor(class_idx, out)

        #             # Resize the CAM and overlay it
        #             original_image = to_pil_image(img_tensor.squeeze(0).cpu())
        #             activation_map_resized = transforms.Resize(original_image.size)(activation_map[0].unsqueeze(0)).squeeze(0)
        #             result_image = overlay_mask(original_image, to_pil_image(activation_map_resized.cpu(), mode='F'), alpha=0.3)

        #             # Save the result image
        #             os.makedirs('results/plots', exist_ok=True)
        #             save_path = os.path.join('results', 'plots', f'cam_{step_no}_{idx}.png')
        #             result_image.save(save_path)

        #             # Convert activation map to binary map
        #             binary_map = (activation_map_resized > activation_map_resized.mean()).float()

        #             # Calculate mean of high activation area
        #             high_activation_mean = (activation_map_resized * binary_map).sum() / binary_map.sum()

        #             # Calculate mean of low activation area
        #             low_activation_mean = (activation_map_resized * (1 - binary_map)).sum() / (1 - binary_map).sum()

        #             # Calculate difference
        #             activation_diff = high_activation_mean - low_activation_mean
        #             activation_diffs.append(activation_diff.item())

        #             # Calculate statistics of the activation map
        #             activation_map_np = activation_map_resized.cpu().numpy()
        #             max_value = activation_map_np.max()
        #             min_value = activation_map_np.min()
        #             mean_value = activation_map_np.mean()
        #             std_value = activation_map_np.std()

        #             activation_stats.append((max_value, min_value, mean_value, std_value))

        #             print(f"Activation difference for image {idx} in step {step_no}: {activation_diff}")
        #             print(f"Activation map stats for image {idx} in step {step_no}: max={max_value}, min={min_value}, mean={mean_value}, std={std_value}")

        # # Save activation differences to a text file
        # activation_diff_path = os.path.join('results', 'plots', 'activation_diff.txt')
        # with open(activation_diff_path, 'w') as f:
        #     for diff in activation_diffs:
        #         f.write(f"{diff}\n")

        # # Calculate and print overall statistics for activation differences
        # activation_diffs = np.array(activation_diffs)
        # print(f"Activation differences - Max: {activation_diffs.max()}, Min: {activation_diffs.min()}, Mean: {activation_diffs.mean()}, Std: {activation_diffs.std()}")

        # # Save activation stats to a text file
        # stats_save_path = os.path.join('results', 'plots', 'activation_stats.txt')
        # with open(stats_save_path, 'w') as f:
        #     for stats in activation_stats:
        #         f.write(f"Max: {stats[0]}, Min: {stats[1]}, Mean: {stats[2]}, Std: {stats[3]}\n")





        #
                # print(layer_cond.ndim)

                # # 处理维度
                # if layer_cond.ndim == 2:  # 对于全连接层 (fc)
                #     layer_cond_np = layer_cond.squeeze().cpu().detach().numpy()
                #     # 在可视化时对二维数据特殊处理，例如添加一个空的维度
                #     layer_cond_np = np.expand_dims(layer_cond_np, axis=-1)
                # else: 
                #     layer_cond_np = np.transpose(layer_cond.squeeze().cpu().detach().numpy(), (1, 2, 0))

                # # Visualization of conductance
                # fig, ax = plt.subplots(figsize=(10, 5))
                # viz.visualize_image_attr(layer_cond_np,
                #                         np.transpose(X_single.squeeze().cpu().numpy(), (1, 2, 0)),
                #                         method="blended_heat_map",
                #                         sign="all",
                #                         show_colorbar=True,
                #                         title=f"Layer Conductance for {layer_name}")
                # plt.savefig(f'results/plots/layer_conductance_{layer_name}_step_{step_no}.png')
                # plt.close(fig)


    # #Layer Integrated Gradients
    # def predict_generator(self, generator):
    #     self.enable_predict_mode()
    #     result = []
    #     start_time = time.time()
        
    #     # Initialize LayerIntegratedGradients for specific layers
    #     integrated_gradients_layers = {
    #         'conv1': LayerIntegratedGradients(self._torch_model.module, self._torch_model.module._model.conv1),
    #         'layer1': LayerIntegratedGradients(self._torch_model.module, self._torch_model.module._model.layer1),
    #         'layer2': LayerIntegratedGradients(self._torch_model.module, self._torch_model.module._model.layer2),
    #         'layer3': LayerIntegratedGradients(self._torch_model.module, self._torch_model.module._model.layer3),
    #         'layer4': LayerIntegratedGradients(self._torch_model.module, self._torch_model.module._model.layer4),
    #         'fc': LayerIntegratedGradients(self._torch_model.module, self._torch_model.module.fc)
    #     }
        
    #     # Initialize dictionaries to accumulate contributions
    #     layer_contributions = {layer: [] for layer in integrated_gradients_layers.keys()}
        
    #     for step_no, X in enumerate(generator):
    #         if isinstance(X, (tuple, list)):
    #             X = X[0]

    #         y_pred = self.predict_on_batch(X)
    #         result.append(y_pred)
    #         print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

    #         # Get the top prediction class for each image in the batch
    #         for i in range(X.shape[0]):  # Iterate over each image in the batch
    #             pred_class = torch.tensor(y_pred[i].argmax(axis=-1))  # Convert to PyTorch tensor
    #             X_single = X[i].unsqueeze(0)  # Select the image and add batch dimension
                
    #             # Calculate integrated gradients for each layer
    #             for layer_name, integrated_gradients in integrated_gradients_layers.items():
    #                 layer_ig = integrated_gradients.attribute(X_single.cuda(), target=pred_class.cuda())  # Ensure target is on GPU if necessary
    #                 layer_contributions[layer_name].append(layer_ig.cpu().detach().numpy())
    #                 print(f"{i}: Layer: {layer_name}, Integrated Gradients: {layer_ig.sum().item()}")

    #     # Calculate average contributions
    #     average_contributions = {}
    #     for layer_name, contributions in layer_contributions.items():
    #         # Convert list to numpy array and compute mean
    #         contributions_array = np.array(contributions)
    #         average_contributions[layer_name] = np.mean(contributions_array, axis=0)
            
    #         # Print or log average contributions
    #         avg_contrib_value = np.sum(average_contributions[layer_name])
    #         print(f"Layer: {layer_name}, Average Integrated Gradients: {avg_contrib_value}")

    #     return np.concatenate(result)


    # #Layer DeepLift
    # def predict_generator(self, generator):
    #     self.enable_predict_mode()
    #     result = []
    #     start_time = time.time()
        
    #     # Initialize LayerDeepLift for specific layers
    #     deeplift_layers = {
    #         'conv1': LayerDeepLift(self._torch_model.module, self._torch_model.module._model.conv1),
    #         'layer1': LayerDeepLift(self._torch_model.module, self._torch_model.module._model.layer1),
    #         'layer2': LayerDeepLift(self._torch_model.module, self._torch_model.module._model.layer2),
    #         'layer3': LayerDeepLift(self._torch_model.module, self._torch_model.module._model.layer3),
    #         'layer4': LayerDeepLift(self._torch_model.module, self._torch_model.module._model.layer4),
    #         'fc': LayerDeepLift(self._torch_model.module, self._torch_model.module.fc)
    #     }
        
    #     # Initialize dictionaries to accumulate contributions
    #     layer_contributions = {layer: [] for layer in deeplift_layers.keys()}
        
    #     for step_no, X in enumerate(generator):
    #         if isinstance(X, (tuple, list)):
    #             X = X[0]

    #         y_pred = self.predict_on_batch(X)
    #         result.append(y_pred)
    #         print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

    #         # Get the top prediction class for each image in the batch
    #         for i in range(X.shape[0]):  # Iterate over each image in the batch
    #             pred_class = torch.tensor(y_pred[i].argmax(axis=-1))  # Convert to PyTorch tensor
    #             X_single = X[i].unsqueeze(0)  # Select the image and add batch dimension
                
    #             # Calculate contributions for each layer using Layer DeepLift
    #             for layer_name, layer_deeplift in deeplift_layers.items():
    #                 layer_cond = layer_deeplift.attribute(X_single.cuda(), target=pred_class.cuda())  # Ensure target is on GPU if necessary
    #                 layer_contributions[layer_name].append(layer_cond.cpu().detach().numpy())
    #                 print(f"{i}: Layer: {layer_name}, DeepLift Contribution: {layer_cond.sum().item()}")

    #     # Calculate average contributions
    #     average_contributions = {}
    #     for layer_name, contributions in layer_contributions.items():
    #         # Convert list to numpy array and compute mean
    #         contributions_array = np.array(contributions)
    #         average_contributions[layer_name] = np.mean(contributions_array, axis=0)
            
    #         # Print or log average contributions
    #         avg_contrib_value = np.sum(average_contributions[layer_name])
    #         print(f"Layer: {layer_name}, Average DeepLift Contribution: {avg_contrib_value}")

    #     return np.concatenate(result)



        
    # #Layer GradientSHAP
    # def predict_generator(self, generator):
    #     self.enable_predict_mode()
    #     result = []
    #     start_time = time.time()
        
    #     # Initialize LayerGradientShap for specific layers
    #     gradientshap_layers = {
    #         'conv1': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.conv1),
    #         'layer1': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer1),
    #         'layer2': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer2),
    #         'layer3': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer3),
    #         'layer4': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer4),
    #         'fc': LayerGradientShap(self._torch_model.module, self._torch_model.module.fc)
    #     }
        
    #     # Initialize dictionaries to accumulate contributions
    #     layer_contributions = {layer: [] for layer in gradientshap_layers.keys()}
        
    #     for step_no, X in enumerate(generator):
    #         if isinstance(X, (tuple, list)):
    #             X = X[0]

    #         y_pred = self.predict_on_batch(X)
    #         result.append(y_pred)
    #         print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

    #         # Get the top prediction class for each image in the batch
    #         for i in range(X.shape[0]):  # Iterate over each image in the batch
    #             pred_class = torch.tensor(y_pred[i].argmax(axis=-1))  # Convert to PyTorch tensor
    #             X_single = X[i].unsqueeze(0)  # Select the image and add batch dimension
                
    #             # Define a baseline (e.g., zero baseline or random noise)
    #             baseline_dist = torch.zeros_like(X_single).cuda()

    #             # Calculate GradientSHAP for each layer
    #             for layer_name, layer_gradientshap in gradientshap_layers.items():
    #                 layer_gradshap = layer_gradientshap.attribute(X_single.cuda(), baselines=baseline_dist, target=pred_class.cuda())
    #                 layer_contributions[layer_name].append(layer_gradshap.cpu().detach().numpy())
    #                 print(f"{i}: Layer: {layer_name}, GradientSHAP: {layer_gradshap.sum().item()}")

    #     # Calculate average contributions
    #     average_contributions = {}
    #     for layer_name, contributions in layer_contributions.items():
    #         # Convert list to numpy array and compute mean
    #         contributions_array = np.array(contributions)
    #         average_contributions[layer_name] = np.mean(contributions_array, axis=0)
            
    #         # Print or log average contributions
    #         avg_contrib_value = np.sum(average_contributions[layer_name])
    #         print(f"Layer: {layer_name}, Average GradientSHAP: {avg_contrib_value}")

    #     return np.concatenate(result)




    # def predict_generator(self, generator):
    #     self.enable_predict_mode()
    #     result = []
    #     start_time = time.time()
        
    #     # Initialize LayerGradientShap for specific layers
    #     gradientshap_layers = {
    #         'conv1': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.conv1),
    #         'layer1': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer1),
    #         'layer2': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer2),
    #         'layer3': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer3),
    #         'layer4': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.layer4),
    #         'avgpool': LayerGradientShap(self._torch_model.module, self._torch_model.module._model.avgpool),
    #         'fc': LayerGradientShap(self._torch_model.module, self._torch_model.module.fc)
    #     }
        
    #     # Initialize dictionaries to accumulate contributions
    #     layer_contributions = {layer: [] for layer in gradientshap_layers.keys()}
        
    #     for step_no, X in enumerate(generator):
    #         if isinstance(X, (tuple, list)):
    #             X = X[0]

    #         y_pred = self.predict_on_batch(X)
    #         result.append(y_pred)
    #         print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))
    #         # print(X[0].shape[0])
    #         # Get the top prediction class for each image in the batch
    #         for i in range(X.shape[0]):  # Iterate over each image in the batch
    #             pred_class = torch.tensor(y_pred[i].argmax(axis=-1))  # Convert to PyTorch tensor
    #             X_single = X[i].unsqueeze(0)  # Select the image and add batch dimension
                
    #             # Define a baseline (e.g., zero baseline or random noise)
    #             baseline_dist = torch.zeros_like(X_single).cuda()

    #             # Calculate GradientSHAP for each layer
    #             for layer_name, layer_gradientshap in gradientshap_layers.items():
    #                 layer_gradshap = layer_gradientshap.attribute(X_single.cuda(), baselines=baseline_dist, target=pred_class.cuda())
    #                 layer_contributions[layer_name].append(layer_gradshap.cpu().detach().numpy())
    #                 print(f"{i}: Layer: {layer_name}, GradientSHAP: {layer_gradshap.sum().item()}")

    #     # Calculate average contributions
    #     average_contributions = {}
    #     for layer_name, contributions in layer_contributions.items():
    #         # Convert list to numpy array and compute mean
    #         contributions_array = np.array(contributions)
    #         average_contributions[layer_name] = np.mean(contributions_array, axis=0)
            
    #         # Print or log average contributions
    #         avg_contrib_value = np.sum(average_contributions[layer_name])
    #         print(f"Layer: {layer_name}, Average GradientSHAP: {avg_contrib_value}")

    #     return np.concatenate(result)



