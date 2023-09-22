from EigenfacesSVD import *
from EmbeddingViaSVD import *
from MNIST_CF import *
def EigenfacesSVDFunction():
    K=64
    Testnum=5
    dataM=loadData('./face_data.csv')#get data matrix
    showImage(dataM[:, 9],"original image")
    meanA=getMeanVector(dataM)
    deMeData=deMeanData(dataM,meanA)# get the de-mean data
    showImage(deMeData[:,9],"de-mean image")
    U, Sigma, V = np.linalg.svd(deMeData, full_matrices=False)
    print("Sigma.shape",Sigma.shape)
    print("U.shape", U.shape)
    print("V.shape", V.shape)
    pltSemilog(Sigma)
    showImages(3,2,U[:,:6],"Singular Vector-6 Largest")
    showImages(3, 2, U[:, U.shape[1]-6:], "Singular Vector-6 Smallest")
    pData=projectData(U[:,:K],deMeData[:,9:9+Testnum])#get the projected data
    reconstruction=reconstruct(U[:,:K],pData)
    for i in range(reconstruction.shape[1]):#for each col
        reconstruction[:,i]=reconstruction[:,i]+meanA
    showCompareImages(Testnum,3, deMeData[:,9:9+Testnum], pData, reconstruction, "Comparision")
    plt.show()
# Press the green button in the gutter to run the script.
def MNIST_SVD():
    K=50
    Testnum=5
    TestCF=1
    MNISTx,MNISTy=LoadData(".\mnist_train.csv")#N*D, need to transpose
    print("MNISTx",MNISTx.shape)
    MNISTxT=np.transpose(MNISTx)
    U, Sigma, V = np.linalg.svd(MNISTxT, full_matrices=False)
    pltSemilog(Sigma)

    if TestCF:
        pData = projectData(U[:, :K], MNISTxT)  # get the projected data
        MNISTSVDPrediction(pData.transpose(),MNISTy,U[:, :K])
    else:
        pData = projectData(U[:, :K], MNISTxT[:, 9:9 + Testnum])  # get the projected data
        reconstruction = reconstruct(U[:, :K], pData)
        showCompareImages(Testnum, 3, MNISTxT[:, 9:9 + Testnum], pData, reconstruction, "Comparision", (28, 28),
                          (5, 10))
        plt.show()
    return pData,
if __name__ == '__main__':
    print("If GPU work? :",torch.cuda.is_available())
    #EigenfacesSVDFunction()
    MNIST_SVD()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
