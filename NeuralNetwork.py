import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
# define the neural network
class MLP(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim):
    super(MLP, self).__init__()
    # Embedding layer with trainable embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(embedding_dim, hidden_dim)      # define layer 1 parameters
    self.linear2 = nn.Linear(hidden_dim, output_dim)               # define layer 2 parameters

  def forward(self, x):
    rdim = 0 if x.ndim == 1 else 1   # if x is a batch, reduction is along dim 1
    x = x.to(torch.long)             # change type (to get around bug in summary!)
    x = self.embed(x)                # returns tensor of shape (*, T, D)
    #x = torch.mean(x, dim=rdim)      # average the embeddings
    x,_ = torch.max(x, dim=rdim)      # max-pooling the embeddings this size should be (*,1,D)
    #print("x.size=",x.size(),"x=",x)
    x = self.linear1(x)              # first linear layer
    x = F.relu(x)                    # nonlinear activation, try tanh perhaps? 
    x = self.linear2(x)              # output logit
    return x

def ModelTrainLoop(model,train_data,num_epochs=15):
    # Example usage:
    learning_rate = 0.001
    print("Start training:")
    criterion = nn.CrossEntropyLoss()  #  Cross-Entropy Loss with Logits
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    train_losses = []
    for epoch in range(num_epochs):
        for data, labels in train_data:  # You'll need to adapt this to your data loading setup
            optimizer.zero_grad()
            outputs = model(data)
            #outputs.float()
            #labels.float()
            #print("outputs.dtye=",outputs.dtype)
            #outputs = outputs.view(-1)
            #print("labels.dtye=",labels.dtype)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        if(epoch%2==0):
            print(f"{epoch}/{num_epochs},losses:{loss.item()}")
        train_losses.append(loss.item())
    print("Train finished!")
    return train_losses

def MultiClassification(model,test_dataloader):
   # Evaluate the model on the test data
  model.eval()  # Set the model to evaluation mode

  true_labels = []
  predicted_labels = []
  total = 0
  correct = 0
  # Iterate through the test data
  with torch.no_grad():
      for batch in test_dataloader:  # Assuming you have a DataLoader for the test data
          text, labels = batch
          outputs = model(text)
          _, predicted = torch.max(outputs, 1)
          true_labels.extend(labels.numpy())
          predicted_labels.extend(predicted.numpy())
          total += labels.size()[0]
          correct += (predicted == labels).sum().item()
  print(f"correct={correct},total={total}")
  accuracy = 100 * correct / total
  print('Accuracy on test data: {}%'.format(accuracy))
  return true_labels,predicted_labels



# Follow are for CNN
import torch
import torch.nn as nn
import torch.optim as optim
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # CNN-1
        self.conv1 = nn.Conv2d(1, 6, padding=2,kernel_size=5)#28+2*2-5 +1=28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#(28-2)/2+1=14
        
        # CNN-2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)#14-5+1=10
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)#(10-2)/2+1=5
        
        # FC
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #print("before conv1,x.size=",x.size())
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

