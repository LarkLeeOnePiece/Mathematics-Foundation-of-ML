import pandas as pd
import numpy as np
def LoadData(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
    Y = df.iloc[:, 0].to_numpy()                # labels of images
    return X,Y