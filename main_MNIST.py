import torch
import pandas as pd
import numpy as np
from main_MNIST_FeaturesEngineering import *
'''
torch.set_printoptions(
    precision=2,    # 精度，保留小数点后几位，默认4
    threshold=1000,
    edgeitems=3,
    linewidth=250,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)
'''
classes=[0,1,2,3,4,5,6,7,8,9]
CLASSIFIERS=1#define the num of classifiers
FEATURES=784
ADD = 5000
#four features
BIAS=0 #wx+b  let etha=(b,w) x=(1,x)
def LabelsProcess(labels,classNum=CLASSIFIERS):
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
def LoadData(path,classNum=CLASSIFIERS):

    df = pd.read_csv(path)
    X = df.iloc[:, 1:].to_numpy() / 255.0  # values are scaled to be between 0 and 1
    labels = df.iloc[:, 0].to_numpy()  # labels of images
    labels=LabelsProcess(labels)
    data=torch.tensor(X)
    data=torch.hstack([torch.ones([data.size()[0], 1]), data])
    Labels=torch.tensor(labels)
    return data, Labels

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
def PrintConfuseMatrix(Matrix,dataNUm):
    x=Matrix.shape[0]
    y=Matrix.shape[1]
    RowTotal=np.sum(Matrix,1)#sum of each row
    ColumnTotal=np.sum(Matrix,0)
    tplt = "{0:10}\t{1:10}\t{2:10}\t{3:10}\t{4:10}\t{5:10}\t{6:10}\t{7:10}\t{8:10}\t{9:10}\t{10:10}\t{11:10}"
    print(tplt.format("Real\prediction", "0", "1", "2","3","4","5","6","7","8","9","Total"))
    for i in range(x):
        print("{:10d}".format(i),end='')
        for j in range(y):
            print("&{:10d}".format(int(Matrix[i][j])),end='')
        print("&{:10d}".format(int(RowTotal[i])))
    print("{:10}".format("ALL"),end='')
    for i in range(len(ColumnTotal)):
        print("&{:10d}".format(int(ColumnTotal[i])),end='')
    print("&{:10}".format(dataNUm))
def TwoPrediction(LearnedPs,data,labels):
    Pre_Y = data.mm(LearnedPs)
    Pre_Y[Pre_Y>0]=1
    Pre_Y[Pre_Y<=0]=-1
    sum=0
    print(Pre_Y)
    for i in range(Pre_Y.size()[0]):
        if Pre_Y[i,0]==labels[i,0]:
            sum+=1
    print("sum=",sum,"Pre_Y.size()[0] =",Pre_Y.size()[0])
    accuracy=sum/Pre_Y.size()[0]
    print("accuracy=,",accuracy)
def Predixtion(LearnedPs,data,labels):
    Pre_Y=data.mm(LearnedPs)
    #print(Pre_Y)
    #figure the number of the classification
    Prediction = torch.argmax(Pre_Y, dim=1).numpy()
    #print(Prediction)
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
    PrintConfuseMatrix(ConfuseMatrix,int(data.size()[0]))
    #print("ConfuMatrix=\n",ConfuseMatrix)
    return 0

def MainFeature(train_data):
    print("FEA=",FEATURES)
    T_train_data=[]
    data_index=[]
    for i in range(FEATURES):
        NoneZeroNum=torch.linalg.norm(train_data[:,i],0)#check the nonzeronum for each column
        if NoneZeroNum>0.1*(train_data.size()[0]):
            T_train_data.append(train_data[:,i].unsqueeze(1))
            data_index.append(i)
    Valid=len(T_train_data)#valid features
    new_Train_data=T_train_data[0]
    #print(T_train_data,len(new_Train_data))
    #print(data_index)
    for i in range(1,Valid):
        new_Train_data = torch.hstack([new_Train_data, T_train_data[i]])
    return new_Train_data,data_index
def TakeMainFeature(orin_data,index):
    testdata=orin_data[:,index]
    return testdata
if __name__ == '__main__':
    print("If GPU work? :", torch.cuda.is_available())
    TestFeatureEngineering=1
    IfTakeMainFeature=1
    IfTwo=1# use for test boolean classifier
    if IfTwo:
        CLASSIFIERS=1
        if TestFeatureEngineering!=1:
            train_data, Train_labels = LoadData('.\data\mnist_train.csv')
            #print(Train_labels)
            FEATURES=train_data.size()[1]
            train_data,data_index=MainFeature(train_data)
            LearnedPs=Train(train_data,Train_labels)
            print("----------  Training  ----------")
            TwoPrediction(LearnedPs, train_data, Train_labels)
            print("----------  Test  ----------")
            test_data, test_labels = LoadData('.\data\mnist_test.csv')
            test_data=TakeMainFeature(test_data,data_index)
            print(test_data.size())
            TwoPrediction(LearnedPs, test_data, test_labels)
        else:    #Add random features
            train_data, Train_labels = LoadData('.\data\mnist_train.csv')
            print("Train_labels",Train_labels)
            FEATURES=train_data.size()[1]
            if IfTakeMainFeature==0:
                RMatrix = RandMatrix(ADD, train_data.size()[1])  # t
                NewFeatures = getRandFeatures(RMatrix, train_data)
                train_data = torch.hstack([train_data, NewFeatures])
                #print(train_data)
                LearnedPs = Train(train_data, Train_labels)
                print("----------  Training  ----------")
                TwoPrediction(LearnedPs, train_data, Train_labels)
                print("----------  Test  ----------")
                test_data, test_labels = LoadData('.\data\mnist_test.csv')
                NewFeatures = getRandFeatures(RMatrix, test_data)
                test_data = torch.hstack([test_data, NewFeatures])
                print(test_data.size())
                TwoPrediction(LearnedPs, test_data, test_labels)
            else:
                train_data,data_index=MainFeature(train_data)
                RMatrix=RandMatrix(ADD,train_data.size()[1])# t
                NewFeatures=getRandFeatures(RMatrix,train_data)
                train_data = torch.hstack([train_data, NewFeatures])
                print(train_data)
                LearnedPs=Train(train_data,Train_labels)
                print("----------  Training  ----------")
                TwoPrediction(LearnedPs, train_data, Train_labels)
                print("----------  Test  ----------")
                test_data, test_labels = LoadData('.\data\mnist_test.csv')
                test_data=TakeMainFeature(test_data,data_index)
                #RMatrix = RandMatrix(ADD, test_data.size()[1])  # t
                NewFeatures = getRandFeatures(RMatrix, test_data)
                test_data = torch.hstack([test_data, NewFeatures])
                print(test_data.size())
                TwoPrediction(LearnedPs, test_data, test_labels)
    else:
        if TestFeatureEngineering!=1:
            train_data, Train_labels = LoadData('.\data\mnist_train.csv')
            #print(Train_labels)
            FEATURES=train_data.size()[1]
            train_data,data_index=MainFeature(train_data)
            LearnedPs=Train(train_data,Train_labels)
            print("----------  Training  ----------")
            Predixtion(LearnedPs, train_data, Train_labels)
            print("----------  Test  ----------")
            test_data, test_labels = LoadData('.\data\mnist_test.csv')
            test_data=TakeMainFeature(test_data,data_index)
            print(test_data.size())
            Predixtion(LearnedPs, test_data, test_labels)
        else:    #Add random features
            train_data, Train_labels = LoadData('.\data\mnist_train.csv')
            #print(Train_labels)
            FEATURES=train_data.size()[1]
            if IfTakeMainFeature==0:
                RMatrix = RandMatrix(ADD, train_data.size()[1])  # t
                NewFeatures = getRandFeatures(RMatrix, train_data)
                train_data = torch.hstack([train_data, NewFeatures])
                print(train_data)
                LearnedPs = Train(train_data, Train_labels)
                print("----------  Training  ----------")
                Predixtion(LearnedPs, train_data, Train_labels)
                print("----------  Test  ----------")
                test_data, test_labels = LoadData('.\data\mnist_test.csv')
                NewFeatures = getRandFeatures(RMatrix, test_data)
                test_data = torch.hstack([test_data, NewFeatures])
                print(test_data.size())
                Predixtion(LearnedPs, test_data, test_labels)
            else:
                train_data,data_index=MainFeature(train_data)
                RMatrix=RandMatrix(ADD,train_data.size()[1])# t
                NewFeatures=getRandFeatures(RMatrix,train_data)
                train_data = torch.hstack([train_data, NewFeatures])
                print(train_data)
                LearnedPs=Train(train_data,Train_labels)
                print("----------  Training  ----------")
                Predixtion(LearnedPs, train_data, Train_labels)
                print("----------  Test  ----------")
                test_data, test_labels = LoadData('.\data\mnist_test.csv')
                test_data=TakeMainFeature(test_data,data_index)
                #RMatrix = RandMatrix(ADD, test_data.size()[1])  # t
                NewFeatures = getRandFeatures(RMatrix, test_data)
                test_data = torch.hstack([test_data, NewFeatures])
                print(test_data.size())
                Predixtion(LearnedPs, test_data, test_labels)




