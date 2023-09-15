import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from K_MEANS import *
from matplotlib.ticker import MaxNLocator

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
def test():
    # a function use to test some functions in libs
    group=[i for i in range(10)]
    x=np.array([j for j in range(10,20)])
    print(x)
    print(group)
    A=np.array([[1,2],[3,4],[5,6]])
    print(np.sum(A,axis=0))# [9 12] sum of column
    print(np.sum(A, axis=1))# [3 7 11] sum of row
    print(np.sum(x[group]))# to use a list to slice data, we require data type is arrry
    #arr = np.array([1, 2, 3, 4, 5])
    #indices = np.array([1, 2])
    #print(arr[indices.tolist()])

#LoadData
#input: path for data
#output: data(array), features(str)
def LoadData(path):
    df = pd.read_csv(path) # df contains the keys as well as the values
    feature_names = df.keys()
    X = df.iloc[:, 0:4].to_numpy()#get all the rows and column 0,1,2,3 four features
    return X,feature_names
def showfigure(classes,features:"4+1 key",reps,J):

    feature_names = features
    plt.figure()
    class_N=len(classes)
    ax = plt.axes(projection='3d')
    for i in range(class_N):
        data=classes[i]
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.scatter(reps[:,0], reps[:, 1], reps[:, 2],marker='X')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    plt.show()
    plt.figure()
    print("len(J)",len(J))
    x_axis_data = [i+1 for i in range(len(J))]  # x
    print(x_axis_data)
    y_axis_data = [J[i] for i in range(len(J))]  # y
    plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='Clustering Objective')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

    plt.legend()  # 显示上面的label
    plt.xlabel('iter')  # x_label
    plt.ylabel('J_Objective')  # y_label
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


# Press the green button in the gutter to run the script.
def showOriginalData(classes,features:"4+1 key"):
    feature_names = features
    ax = plt.axes(projection='3d')
    class_N=3
    class1=[i for i in range(50)]
    class2 = [i for i in range(50,100)]
    class3 = [i for i in range(100,150)]
    Allclasses=[class1,class2,class3]# save 3 classes indexes
    for i in range(class_N):
        data = classes[Allclasses[i],:]
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],'i')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    plt.show()
    plt.figure()
if __name__ == '__main__':
    trytimes=10
    print_hi('PyCharm')
    path='iris.csv'
    class_K=3
    data,features=LoadData(path)
    showOriginalData(data,features)
    column=2
    row=trytimes
    for n in range(trytimes):
        assignmnet,reps,J_List=k_means(data,K=class_K)
        #print(list(enumerate(assignmnet)))
        class_data = []
        class_index=[]
        for j in range(class_K):
            class_index.append([i for i,x in enumerate(assignmnet) if x==j])
            G=np.array(data[class_index[j],:])
            class_data.append(G)
        showfigure(class_data,features,reps,J_List)




# you may want to write your kmeans routine separately (in a kmeans.py file) and import it here
# from kmeans import kmeans



    # YOUR CODE GOES HERE

    # You should cluster the data to get an assignmemt of the training vectors to group IDs
    # add an argument ``c = avec'' to scatter to color the groups differently. avec is the assignment
    # that your clustering generated


