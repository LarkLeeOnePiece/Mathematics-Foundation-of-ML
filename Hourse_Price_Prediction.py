import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from LeastSquaresPredictor import *
from NetworkPredictor import *

# Read training data
df = pd.read_csv('data/houses_train.txt', delimiter = "\t", skiprows=[0, 1, 2, 3, 4], header=None) 
X = df.iloc[:, [0,1,2,3,4, 6,7,8]].to_numpy().astype(np.float32)
y = df.iloc[:, 5].to_numpy().astype(np.float32)
N, D = X.shape
print(f"N={N},D={D}")#N=200,D=8N=200,D=8
print(f"y.shape={y.shape}")#y.shape=(200,) 
# Read testing data
df2 = pd.read_csv('data/houses_test.txt', delimiter = "\t", header=None) 
X_test = df2.iloc[:, [0,1,2,3,4, 6,7,8]].to_numpy().astype(np.float32)
y_test = df2.iloc[:, 5].to_numpy().astype(np.float32)
print(f"X_test.shape={X_test.shape}")#X_test.shape=(40, 8)
print(f"y_test.shape={y_test.shape}")#y_test.shape=(40,) 


#First for Leat Square predict
#step build the model
LQP_model=LeastSquareModel(X,y)
LQP_pred=LeastSquarePredictor(LQP_model,X_test)
# Step 3: Evaluate the model's prediction accuracy.
mse = mean_squared_error(y_test, LQP_pred)
mae = mean_absolute_error(y_test, LQP_pred)
r2 = r2_score(y_test, LQP_pred)
print(f"Regression Mean Squared Error: {mse}")
print(f"Regression Mean Absolute Error: {mae}")
print(f"Regression R-squared (R2) Score: {r2}")
NetworkHousePricePredict(X,y,X_test,y_test,450)