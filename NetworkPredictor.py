import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
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

## follow is for sentimennt analysis with pretrained word vector, I need to build the neural network
class SentimentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentimentNeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.relu(x)                    # nonlinear activation, try tanh perhaps? 
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
def Sentimenttrainingloop(train_data):
    # Load your training and testing data, and apply text_transform to convert text to indices
    # Initialize the model and other hyperparameters
    input_size = 50  # Assuming your input data is of size 256
    hidden_size = 32
    learning_rate = 0.01
    num_epochs = 200
    print("Start training:")
    model = SentimentNeuralNetwork(input_size, hidden_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    train_losses = []
    for epoch in range(num_epochs):
        for data, labels in train_data:  # You'll need to adapt this to your data loading setup
            optimizer.zero_grad()
            outputs = model(data)
            #print("outputs.dtye=",outputs.dtype)
            outputs = outputs.view(-1)
            #print("labels.dtye=",labels.dtype)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        if(epoch%20==0):
            print(f"{epoch}/{num_epochs},losses:{loss.item()}")
        train_losses.append(loss.item())

    return model,train_losses

    # Print or log the loss for monitoring
def SentimentValidFn(model,test_data):
    # Testing loop
    correct = 0
    total = 0
    predicted=[]
    with torch.no_grad():
        for data, labels in test_data:  # Adjust this to your test data
            print(f"labels={labels.size()[0]}")
            outputs = model(data)
            outputs = outputs.view(-1)
            predicted = (outputs > 0.5).float()  # Convert to 0 or 1 based on a threshold
            print(f"predicted.size={predicted.size()}")
            total += labels.size()[0]
            #print("predicted == labels=",predicted == labels)
            correct += (predicted == labels).sum().item()
    print(f"correct={correct},total={total}")
    accuracy = 100 * correct / total
    print('Accuracy on test data: {}%'.format(accuracy))
    return predicted
# Confusion draw function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    import itertools
    """
    绘制混淆矩阵
    :param cm: 混淆矩阵
    :param classes: 类别标签
    :param normalize: 是否归一化
    :param title: 图表标题
    :param cmap: 颜色图谱
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

