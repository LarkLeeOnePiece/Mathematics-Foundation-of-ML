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
classes=['Iris-setosa','Iris-versicolor','Iris-virginica']
CLASSIFIERS=3#define the num of classifiers
FEATURES=4 #four features
BIAS=0 #wx+b  let etha=(b,w) x=(1,x)

#LoadData
#input: path for data
#output: data(array), features(str)
def LabelsProcess(labels,classNum=3):
    N=len(labels)
    labels_array=np.zeros((N,classNum))#size N*classNum,like 150*3
    #set the value according the labes
    for col in range(classNum):
        for i in range(N):
            if labels[i]==classes[col]:#first class
                labels_array[i,col]=1
            else:
                labels_array[i,col]=-1
    return labels_array


# LoadData
# input: path for data
# output: data(array), features(str)
def LoadData(path,classNum=3):
    df = pd.read_csv(path)  # df contains the keys as well as the values
    feature_names = df.keys()
    X = df.iloc[:, 0:4].to_numpy()  # get all the rows and column 0,1,2,3 four features
    labels = df.iloc[:, -1]
    data=torch.tensor(X)
    data=torch.hstack([torch.ones([data.size()[0], 1]), data])
    Labels = LabelsProcess(labels)  # decoding the labels to arrays
    Labels=torch.tensor(Labels)
    return data, feature_names, Labels

def QR_Decomposition(A,y):
    q, r = torch.linalg.qr(A)
    #print("y=",y.unsqueeze(1))
    x = torch.inverse(r).mm(q.t().mm(y.unsqueeze(1)))
    return x
'''
Train:
function: train the paramter
let y=wx+b set W=(1,w),A=(1,x)
'''
def Train(train_data,Train_labels):
    #first decide the size for X(FEATURES+1)*ClassesNUM
    X=[]#tempt save the tensors
    #Add 1 to Feature build A(1,x)
    A= train_data
    for i in range(Train_labels.size()[1]):
        x=QR_Decomposition(A,Train_labels[:, i])
        #print("x=",x)
        X.append(x)
    X_tensor=X[0]
    #print("X=",X)
    for i in range(1,Train_labels.size()[1]):
        X_tensor=torch.hstack([X_tensor,X[i]] )
    #print("X_Tensor=",X_tensor)
    return X_tensor# the learned parameters
def Predixtion(LearnedPs,data,labels):
    Pre_Y=data.mm(LearnedPs)
    #figure the number of the classification
    Prediction = torch.argmax(Pre_Y, dim=1).numpy()
    TrueLabel=torch.argmax(labels, dim=1).numpy()
    sum=0
    ConfuseMatrix=np.zeros((CLASSIFIERS,CLASSIFIERS))
    for i in range(len(Prediction)):
        if Prediction[i]==TrueLabel[i]:
            sum+=1
            ConfuseMatrix[TrueLabel[i],Prediction[i]]+=1
        else:
            ConfuseMatrix[TrueLabel[i], Prediction[i]] += 1
    Accuracy=sum/(len(Prediction))
    print("Accuracy=",Accuracy)
    print("ConfuMatrix=\n",ConfuseMatrix)
    return 0
if __name__ == '__main__':
    print("If GPU work? :",torch.cuda.is_available())
    train_data,features_key,Train_labels=LoadData('.\data\iris_train.csv')
    #print(train_data,Train_labels)
    Parameters=Train(train_data,Train_labels)
    print("Prediction:")
    Predixtion(Parameters,train_data,Train_labels)
    print("Test:")
    test_data, features_key, test_labels = LoadData('.\data\iris_test.csv')
    Predixtion(Parameters,test_data,test_labels)
    #Labels processing finished,Start the process of learning the parameters


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
