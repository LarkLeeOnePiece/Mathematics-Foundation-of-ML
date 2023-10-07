import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from figurePLT import *
df = pd.read_csv('./data/iris.csv')

# extract only two classes 'Iris-setosa' and 'Iris-versicolor'. Drop 'Iris-virginica'
df = df[df['species'] != 'Iris-virginica']

# make the labels 1 and 0
df['species'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True)

# generate X and y tensors, adding the ``1'' feature for the bias
N, D = df.shape# 100,5
X = torch.tensor(df.iloc[:, 0:D-1].values, dtype=torch.float32)
X = torch.cat((torch.ones((N,1)), X), dim=1)# Add 1 as column
y = torch.tensor(df.iloc[:, D-1].values, dtype=torch.float32)

#print("N=",N,", D=",D)

#Step 1: Get data
train_ratio=0.8
test_ratio=0.2
train_size=int(X.size()[0]*train_ratio)
test_size=int(X.size()[0]*test_ratio)
Randon_indices = torch.randperm(len(X))
#shuffle the data
X=X[Randon_indices]
y=y[Randon_indices]

x_train_data, x_test_data = torch.split(X, [train_size, test_size])
y_train_label, y_test_label = torch.split(y, [train_size,test_size])
print("train_data.size()=",x_train_data.size(),"test_data.size()=",x_test_data.size())

# Step 2: Define Model and Loss Function
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
#initialize my oject
model = LogisticRegression(input_dim=x_train_data.size()[1])#the number of features train_data.size()[1]
criterion = nn.BCELoss() #Binary cross entropy loss function


# Step 3: Set Hyperparameters
lr = 0.01
B = 10
num_epochs = 100

# Step 4: Initialize Optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)
#print("model.parameters()=",model.parameters())

# Step 5: SGD Optimization Loop
train_losses=[]
test_losses=[]
#print("len(x_train_data)=",len(x_train_data)) len(x_train_data)=80
for epoch in range(num_epochs):
        # Shuffle the training data at the beginning of each epoch (optional)
    indices = torch.randperm(len(x_train_data))
    X_train_data_shuffled = x_train_data[indices]
    y_train_label_shuffled = y_train_label[indices]
        #shuffle the testing data
    indices_test = torch.randperm(len(x_test_data))
    X_test_data_shuffled = x_test_data[indices_test]
    y_test_label_shuffled = y_test_label[indices_test]
    for i in range(0, len(X_train_data_shuffled), B):
        minibatch_X = X_train_data_shuffled[i:i + B]
        minibatch_y = y_train_label_shuffled[i:i + B]

        optimizer.zero_grad()  # Zero gradients
        outputs = model(minibatch_X)  # Forward pass
        loss = criterion(outputs.view(-1), minibatch_y.float())  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
    # Optionally, print or log the loss for monitoring training progress
    #print(f'Training Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')
    train_losses.append(loss.item())
    test_preY = model(X_test_data_shuffled)  # Forward pass
    test_loss = criterion(test_preY.view(-1), y_test_label_shuffled.float())  # Compute loss
    #print(f'Testing Epoch [{epoch+1}/{num_epochs}], Testing Loss: {test_loss.item():.4f}')
    test_losses.append(test_loss.item())
# Training is complete, and model parameters are optimized.
#plot the figure
pltsemi(train_losses,test_losses)
confux(model,x_test_data,y_test_label)
print("y_test_label=",y_test_label)
plt.show()