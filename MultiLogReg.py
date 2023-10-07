import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from figurePLT import *
import pandas as pd
# Load the Iris dataset
df = pd.read_csv('./data/iris.csv')
# make the labels 0,1,2
df['species'].replace(["Iris-setosa","Iris-versicolor",'Iris-virginica'], [0,1,2], inplace=True)
N, D = df.shape# 
X = torch.tensor(df.iloc[:, 0:D-1].values, dtype=torch.float32)
X = torch.cat((torch.ones((N,1)), X), dim=1)# Add 1 as column
y = torch.tensor(df.iloc[:, D-1].values, dtype=torch.int64)
# Standardize features (optional but can help with convergence)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
x_train_data, x_test_data, y_train_label, y_test_label = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model class
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# Initialize the model and set hyperparameters
input_dim = x_train_data.shape[1]
output_dim = len(set(y_train_label))
model = LogisticRegressionModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
B = 10
# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    
    

    indices = torch.randperm(len(x_train_data))
    X_train_data_shuffled = x_train_data[indices]
    inputs = torch.FloatTensor(X_train_data_shuffled)
    y_train_label_shuffled = y_train_label[indices]
    labels = torch.LongTensor(y_train_label_shuffled)
    

    for i in range(0, len(X_train_data_shuffled), B):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Model evaluation
with torch.no_grad():
        #shuffle the testing data
    indices_test = torch.randperm(len(x_test_data))
    X_test_data_shuffled = x_test_data[indices_test]
    y_test_label_shuffled = y_test_label[indices_test]
    inputs = torch.FloatTensor(X_test_data_shuffled)
    labels = torch.LongTensor(y_test_label_shuffled)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1) #
    print("labels.numpy()=",labels.numpy())
    accuracy = accuracy_score(labels.numpy(), predicted.numpy())
    print(f'Accuracy: {accuracy*100:.2f}%')
confux(model,torch.FloatTensor(x_test_data),torch.LongTensor(y_test_label),CLASSIFIERS=3)
print("y_test_label=",y_test_label)
plt.show()