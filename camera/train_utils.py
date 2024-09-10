import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import os

NUM_EPOCH = 150

def predict_test(model, test_loader,labels):
    return model.predict_generator(test_loader,labels)

def train(model, train_loader, val_loader, val_labels, model_save_path):
    epoch_id = 0
    best_loss = 1000

    # 用于记录每个epoch的损失值和准确率
    losses = []
    accuracies = []

    while True:
        model.fit_generator(train_loader)
        y_pred = model.predict_generator(val_loader)

        loss = log_loss(val_labels, y_pred)
        accuracy = accuracy_score(val_labels, y_pred.argmax(axis=-1))

        print("Epoch {0}. Val accuracy {1}. Val loss {2}".format(epoch_id, accuracy, loss))

        # 记录损失值和准确率
        losses.append(loss)
        accuracies.append(accuracy)

        model.scheduler_step(loss, epoch_id)
        if loss < best_loss:
            best_loss = loss
            model.save(model_save_path)

        epoch_id += 1

        if epoch_id == NUM_EPOCH:
            break

    # 确保results/plots文件夹存在
    os.makedirs('results/plots', exist_ok=True)

    # 绘制损失值和准确率的图表
    epochs = np.arange(NUM_EPOCH)

    # 绘制损失值的图表
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(losses)), losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')

    # 标出最大值和最小值
    min_loss = min(losses)
    max_loss = max(losses)
    min_loss_epoch = losses.index(min_loss)
    max_loss_epoch = losses.index(max_loss)

    plt.plot(min_loss_epoch, min_loss, 'ro')  # 标出最小值
    plt.plot(max_loss_epoch, max_loss, 'bo')  # 标出最大值
    plt.annotate(f'Min: {min_loss:.2f}', (min_loss_epoch, min_loss), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'Max: {max_loss:.2f}', (max_loss_epoch, max_loss), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.legend()
    plt.savefig('results/plots/validation_loss.png')
    plt.close()

    # 绘制准确率的图表
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(accuracies)), accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')

    # 标出最大值和最小值
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    min_accuracy_epoch = accuracies.index(min_accuracy)
    max_accuracy_epoch = accuracies.index(max_accuracy)

    plt.plot(min_accuracy_epoch, min_accuracy, 'ro')  # 标出最小值
    plt.plot(max_accuracy_epoch, max_accuracy, 'bo')  # 标出最大值
    plt.annotate(f'Min: {min_accuracy:.2f}', (min_accuracy_epoch, min_accuracy), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'Max: {max_accuracy:.2f}', (max_accuracy_epoch, max_accuracy), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.legend()
    plt.savefig('results/plots/validation_accuracy.png')
    plt.close()
