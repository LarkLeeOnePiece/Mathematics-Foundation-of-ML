import numpy as np
import matplotlib.pyplot as plt
# Confusion draw function
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch
def plot_confusion_matrix2(true_labels, predicted_labels, class_names=['Class 0', 'Class 1', 'Class 2']):
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 创建热图
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    #plt.show()



def plot_confusion_matrix(cm, classes=['class1,class2'], normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    import itertools
    """
    绘制混淆矩阵
    :param cm: 混淆矩阵
    :param classes: 类别标签
    :param normalize: 是否归一化
    :param title: 图表标题
    :param cmap: 颜色图谱
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plotLossHistory(train_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()

# read MNIST training data
def LoadData(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
    y = df.iloc[:, 0].to_numpy()                # labels of images
    X=torch.tensor(X)
    y=torch.tensor(y)
    X=X.view(len(y),1, 28, 28)
    print("X.shape=",X.size())
    X=X.to(torch.float32)
    #y=y.to(torch.float32)
    return X,y