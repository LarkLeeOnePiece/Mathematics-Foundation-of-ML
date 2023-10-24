import torch
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchtext
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader
from collections import Counter
from torchinfo import summary
from NetworkPredictor import *
from sklearn.metrics import confusion_matrix
# Define the sentiment analysis model
class TrainableSentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TrainableSentimentAnalysisModel, self).__init__()
        
        # Embedding layer with trainable embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)      # define layer 1 parameters
        # Fully connected layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rdim = 0 if x.ndim == 1 else 1   # if x is a batch, reduction is along dim 1
        # Apply the embedding layer to get word embeddings
        x = self.embedding(x)            # returns tensor of shape (*, T, D)             
        x = torch.mean(x, dim=rdim)      # average the embeddings
        #print(f"x_average size={x.size()}")
        x = self.linear1(x)              # first linear layer
        x = F.relu(x)                    # nonlinear activation, try tanh perhaps? 
        x = self.linear2(x)              
        x = self.sigmoid(x)             # output logit
        return x
def TrainableSentimentLoop(vocab_size,train_data):
    # Example usage:
    # Define the model with D=50 (embedding dimension)
    
    learning_rate = 0.001
    num_epochs = 200
    print("Start training:")
    model = TrainableSentimentAnalysisModel(vocab_size, embedding_dim=50, hidden_dim=32, output_dim=1)
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
    print("Train finished!")
    return model,train_losses

    # Print or log the loss for monitoring
def TrainableSentimentValidFn(model,test_data):
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

# download or read from cache IMDb data
train_data = list(IMDB(split='train'))
test_data = list(IMDB(split='test'))

# download or read from cache word embeddings
vectors = torchtext.vocab.GloVe(name='6B', dim=50, max_vectors=100_000)

# build the vocabulary---all words in training data
tokenizer = get_tokenizer('basic_english')
counter = Counter()
for (label, line) in train_data:
    counter.update(tokenizer(line))
counter = dict(counter.most_common())    
vocab = torchtext.vocab.vocab(counter, min_freq=10, specials=('<PAD>', '<unk>'))
vocab.set_default_index(vocab['<unk>'])# if encounter a word that it didn't know, set as '<unk>'

# takes in a review (string), returns a vector of indices (maxlength T)
def text_transform(x, T=256):
    indices = [vocab[token] for token in tokenizer(x)]
    return indices[:256] if len(indices) > T else indices + [0] * (T-len(indices)) # if the length of review over 256, take the first 256 words, else padding with '<PAD>' unless it reach to 256

# takes in the sentiment, returns 1/0 labels (1 for positive review)
def label_transform(x):
    return 1 if x == 2 else 0

# returns X, y tensors from training data
def create_tensors(train_data):
    label_list, text_list = [], []
    for idx, (label, text) in enumerate(train_data):
        label_list.append(label_transform(label))
        processed_text = torch.tensor(text_transform(text))
        text_list.append(processed_text)
    return torch.stack(text_list), torch.tensor(label_list)

X, y = create_tensors(train_data)
Test_X, Test_y = create_tensors(test_data)
dataset = TensorDataset(X, y)
train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
Testdataset = TensorDataset(Test_X, Test_y)
Test_dataloader = DataLoader(Testdataset, batch_size=len(Testdataset), shuffle=False)
print("Vocabulary size: ", len(vocab))
Mymodel,train_losses=TrainableSentimentLoop(len(vocab),train_dataloader)
predictlabel=SentimentValidFn(Mymodel,Test_dataloader)
confusion = confusion_matrix(Test_y, predictlabel)
print("Confusion Matrix:")
print(confusion)
# classes label
class_labels = ["Class 1", "Class 2"]
# 绘制混淆矩阵
plot_confusion_matrix(confusion, classes=class_labels, title='Confusion Matrix', normalize=False)
# Plot the loss versus the number of training epochs.
plt.figure()
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()