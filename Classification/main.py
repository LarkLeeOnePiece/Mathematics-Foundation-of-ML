'''
Author: Da Li
Processes:
#: Process the labels for learning parameters,like[0,1,0] means it belongs to the second class
#: Learn 3 parameters vectors
#
'''
import torch
import pandas as pd
import numpy as np
class1='Iris-setosa'
class2='Iris-versicolor'
class3='Iris-virginica'
CLASSIFIERS=3#define the num of classifiers
#LoadData
#input: path for data
#output: data(array), features(str)
def LoadData(path):
    df = pd.read_csv(path) # df contains the keys as well as the values
    feature_names = df.keys()
    X = df.iloc[:, 0:4].to_numpy()#get all the rows and column 0,1,2,3 four features
    labels=df.iloc[:, -1]
    return X,feature_names,labels
# Press the green button in the gutter to run the script.

#LoadData
#input: path for data
#output: data(array), features(str)
def LabelsProcess(labels,classNum=3):
    N=len(labels)
    labels_array=np.zeros((N,classNum))#size N*classNum,like 150*3
    #set the value according the labes
    for i in range(N):
        if labels[i]==class1:#first class
            labels_array[i,0]=1
        if labels[i]==class2:#first class
            labels_array[i,1]=1
        if labels[i]==class3:#first class
            labels_array[i,2]=1
    return labels_array
if __name__ == '__main__':
    print(torch.cuda.is_available())
    datapoint,features_key,o_labels=LoadData('.\data\iris.csv')
    #print(datapoint,features_key,o_labels)
    datapoint=torch.tensor(datapoint)
    #print(datapoint)
    DeLabels=LabelsProcess(o_labels)#decoding the labels to arrays
    #print(DeLabels)
    DeLabels=torch.tensor(DeLabels)
    #print(DeLabels)
    #Labels processing finished,Start the process of learning the parameters

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
