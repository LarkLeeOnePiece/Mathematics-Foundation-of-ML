import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load your data and preprocess it as needed.
# Assuming you have X_train, y_train, X_test, and y_test as PyTorch tensors or NumPy arrays.

# Define the neural network architecture.
class HousePricePredictor(nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()
        self.fc1 = nn.Linear(8, 10)  # Input size is 8, output size is 10.
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)  # Output layer with one unit for regression.

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def NetworkHousePricePredict(X_train,y_train,X_test,y_test,EPOCH,BATCH=32):
    model = HousePricePredictor()
    # Define the loss and optimizer.
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters())

    # Define training hyperparameters.
    batch_size = BATCH
    epochs = EPOCH  # You can adjust this number. #100 ---》 0.6900373960751197   300---》      4000-----》 0.9116883594431882

    # Convert data to PyTorch tensors if not already.
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # Lists to store training and validation losses for plotting.
    train_losses = []
    val_losses = []

    # Training loop.
    for epoch in range(epochs):
        model.train()  # Set the model in training mode.
        for i in range(0, len(X_train), batch_size):
            optimizer.zero_grad()  # Zero the gradients.
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            outputs = model(batch_X)
            #print(f"outputs={outputs.size()},batch_y={batch_y.size()}")#outputs=torch.Size([32, 1])--->2D    batch_y=torch.Size([32]) --->1D
            outputs = outputs.view(-1)  # Remove the extra dimension to match the shape of y_test.
            #print(f"CHANGED:  outputs={outputs.size()},batch_y={batch_y.size()}")#outputs=torch.Size([32, 1])--->2D    batch_y=torch.Size([32]) --->1D
            loss = criterion(outputs, batch_y)
            loss.backward()  # Compute gradients.
            optimizer.step()  # Update weights.

        # Validation loss.
        model.eval()  # Set the model in evaluation mode.
        with torch.no_grad():
            val_output = model(X_test)
            val_output = val_output.view(-1)
            val_loss = criterion(val_output, y_test)
            val_losses.append(val_loss.item())
        train_losses.append(loss.item())

    # Plot the loss versus the number of training epochs.
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    

    # Make predictions on the test data.
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        print(f"y_pred={y_pred.size()}")
        y_pred = y_pred.view(-1)
        print(f"CHANGED---> y_pred={y_pred.size()}")

    # Evaluate the model's prediction accuracy.

    mse = ((y_test - y_pred)**2).mean().item()
    mae = (torch.abs(y_test - y_pred)).mean().item()
    r2 = 1 - mse / (y_test.var().item())

    print(f"Network Mean Squared Error (MSE): {mse}")
    print(f"Network Mean Absolute Error (MAE): {mae}")
    print(f"Network R-squared (R2) Score: {r2}")
    plt.show()