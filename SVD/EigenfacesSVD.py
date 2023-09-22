import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def loadData(path):
    df = pd.read_csv(path)
    A = df.iloc[:,:4096].to_numpy().transpose()
    return A
def getMeanVector(data:"data Matrix"):
    mean=np.mean(data,axis=1)#get the mean of each row
    return mean
def deMeanData(oData:"original data",mVec:"mean vector"):
    for i in range(oData.shape[1]):#for each col
        oData[:,i]=oData[:,i]-mVec
    return oData
def pltSemilog(Y):
    x = np.arange(1,Y.shape[0]+1,1)
    y = [Y[i-1]for i in x]
    plt.figure()
    plt.title("Singular Values")
    plt.semilogy(x, y)
def showImages(numx,numy,Images,title):
    plt.figure()
    plt.suptitle(title)
    num=numx*numy
    for i in range(num):
        plt.subplot(numx,numy,i+1)
        imag = Images[:,i].reshape((64, 64))
        plt.imshow(imag, cmap='gray')
def showCompareImages(numx,numy,Images1,Images2,Images3,title,shape1=(64,64),shape2=(8,8)):#numx=3 3 classes, numy=N  number of each class
    plt.figure()
    plt.suptitle(title)
    num=numx*numy
    for i in range(num):
        plt.subplot(numx,numy,i+1)
        if (i+1)%numy==1:
            plt.title("original")
            imag = Images1[:, i//numy].reshape(shape1)
            plt.imshow(imag, cmap='gray')
        elif (i+1)%numy==2:
            plt.title("projection")
            imag = Images2[:, i//numy].reshape(shape2)
            plt.imshow(imag, cmap='gray')
        elif (i+1)%numy==0:
            plt.title("reconstrction")
            imag = Images3[:, i//numy].reshape(shape1)
            plt.imshow(imag, cmap='gray')

def projectData(pMatrix:"project Matrix",data:"original data"):
    pMatrixT=pMatrix.transpose()
    projection=np.dot(pMatrixT,data)
    return projection
def reconstruct(cMatrix:"construct Matrix",data:"projected data"):
    reconstruction=np.dot(cMatrix,data)
    return reconstruction
def showImage(A:"Iamge data",title,shape=(64,64)):
    # display the 10th image in the data set
    plt.figure()
    plt.title(title)
    imag = A.reshape(shape)
    plt.imshow(imag, cmap='gray')
