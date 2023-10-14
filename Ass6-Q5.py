import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
# read MNIST training data
df = pd.read_csv('./data/mnist_train.csv')
X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
y = df.iloc[:, 0].to_numpy()                # labels of images

# divide the data into batches (we loop through the batches in training)
batch_size = 32
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


# Define the neural network model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # First hidden layer with 128 neurons and ReLU activation
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU()
        )
        # Second hidden layer with 128 neurons and ReLU activation
        self.hidden2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Output layer with 10 neurons (for 10 classes)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x

# Create an instance of the model
model = MNISTClassifier()
momentum=0.9
lr=0.01
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)  # Stochastic Gradient Descent

# Train the model on the MNIST dataset using your data loading and training code
# Training loop
num_epochs = 10
import matplotlib.pyplot as plt
def ShowFigure(X,RepsLabels,ith):
    plt.figure()
    num=X.shape[0]
    #print("num=",num)
    row=4
    column=num//row
    plt.title("epoch : "+str(ith))
    for i in range(num):
        ax=plt.subplot(row, column, i + 1, xticks=[], yticks=[])
        image = X[i, :].reshape((28, 28))
        ax.set_title("Pred:%d"%RepsLabels[i],loc='center')
        plt.imshow(image, cmap='gray')
    plt.tight_layout()
    #plt.show()
index=[1,1000,2000,4000,5000,7000,8000,9000]
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data.view(-1, 784))  # Reshape the input data
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % 1500 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
    if epoch%4==0:
        pltonce=0#only plt once
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted=predicted.tolist()
            predicted=np.array(predicted)
            print("predicted.shape",predicted.shape)
            if pltonce==0:
                X=inputs[index]
                y=predicted[index]
                ShowFigure(X,y,epoch)
                pltonce=1
plt.show()
print('Training finished.')
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        #print("outputs.shape",outputs.size())#outputs.shape torch.Size([12000, 10])
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        


accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy}%')
summary(model)
