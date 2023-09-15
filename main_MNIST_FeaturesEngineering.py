import torch
import pandas as pd
import numpy as np
from numpy import random
def RandMatrix(x,y):
    matrix=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            num=random.randint(2)
            if num>0:
                matrix[i,j]=num# set as 1
            else:
                matrix[i, j] =-1# otherwise, set as -1
    return torch.tensor(matrix)
def getRandFeatures(RandMatrix,OriMatrix):
    AddFeatures=torch.mm(OriMatrix, RandMatrix.t())#size: N*5000
    AddFeatures[AddFeatures > 0] = 1
    AddFeatures[AddFeatures <= 0] = 0
    return AddFeatures
