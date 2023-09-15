import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from K_MEANS import *
from matplotlib.ticker import MaxNLocator
plt.ion()
# read MNIST training data
def LoadData(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
    y = df.iloc[:, 0].to_numpy()                # labels of images
    return X,y
def ShowFigure(X,RepsLabels):
    plt.figure()
    num=X.shape[0]
    print(num)
    row=4
    column=num//row
    for i in range(num):
        ax=plt.subplot(row, column, i + 1, xticks=[], yticks=[])
        image = X[i, :].reshape((28, 28))
        ax.set_title("Repres:%d"%RepsLabels[i],loc='center')
        plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.show()
def Classifier(repres,represlabel,testdata,testlabels):
    accuracy=0
    K=repres.shape[0]# K classes
    N=testdata.shape[0] # N test data
    CorrectNum=0
    for i in range(N):
        prediction=np.argmin([la.norm(testdata[i]-repres[p])for p in range(K)])
        if represlabel[prediction]==testlabels[i]:
            #classified successfully
            CorrectNum+=1
    accuracy=CorrectNum/N
    print("accuracy:",accuracy)
    return accuracy

if __name__ == '__main__':
    path='mnist_train.csv'
    data, labels = LoadData(path)
    testpath='mnist_test.csv'
    testdata, testlabels = LoadData(path)
    class_K = 20
    accuary=[]
    for n in range(10):
        assignmnet, reps, J_List = k_means(data, maxiters=30,K=class_K)
        np.savetxt(f"MNISTAss{n}.cvs",assignmnet,fmt="%d")
        # print(list(enumerate(assignmnet)))
        class_data = []
        class_index = []
        class_labels=[]
        for j in range(class_K):
            class_index.append([i for i, x in enumerate(assignmnet) if x == j])
            G = np.array(data[class_index[j], :])
            Prelabels=np.argmax(np.bincount(labels[class_index[j]]))
            class_labels.append(Prelabels)
            class_data.append(G)
        print(class_labels)
        ShowFigure(reps,class_labels)
        a=Classifier(reps,class_labels,testdata,testlabels)
        accuary.append(a)

    [0.7064666666666667, 0.69035, 0.7011333333333334, 0.6934833333333333, 0.6842, 0.71615, 0.73255, 0.7044333333333334,
     0.7135, 0.7085833333333333]
    plt.figure()
    print("len(accuary)", len(accuary))
    x_axis_data = [i + 1 for i in range(len(accuary))]  # x
    y_axis_data = [accuary[i] for i in range(len(accuary))]  # y
    print(y_axis_data)
    plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1,
             label='Classification Accuracy')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

    plt.legend()  # 显示上面的label
    plt.xlabel('Test Count')  # x_label
    plt.ylabel('Accuracy')  # y_label
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ioff()
    plt.show()
# plot the first dozen images from the data set

