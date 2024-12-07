import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from K_MEANS import *


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
def showfigure(classes,features:"4+1 key",reps):

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path='data/iris.csv'
    class_K=3
    data,features=LoadData(path)
    assignmnet,reps=k_means(data,K=class_K)
    #print(list(enumerate(assignmnet)))
    class_data = []
    class_index=[]
    for j in range(class_K):
        class_index.append([i for i,x in enumerate(assignmnet) if x==j])
        G=np.array(data[class_index[j],:])
        class_data.append(G)
    showfigure(class_data,features,reps)
    #print(class_data)

# you may want to write your kmeans routine separately (in a kmeans.py file) and import it here
# from kmeans import kmeans



    # YOUR CODE GOES HERE

    # You should cluster the data to get an assignmemt of the training vectors to group IDs
    # add an argument ``c = avec'' to scatter to color the groups differently. avec is the assignment
    # that your clustering generated


