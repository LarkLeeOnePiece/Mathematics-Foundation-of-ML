import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader
from collections import Counter
from torchinfo import summary
from NeuralNetwork import *
from utilis import *
from sklearn.metrics import confusion_matrix

train_data = list(AG_NEWS(split='train'))
test_data  = list(AG_NEWS(split='test'))

# build a vocabulary---all words in training data
tokenizer = get_tokenizer('basic_english')
counter = Counter()
for (label, line) in train_data:
    counter.update(tokenizer(line))
counter = dict(counter.most_common())    
vocab = torchtext.vocab.vocab(counter, min_freq=10, specials=('<PAD>', '<unk>'))
vocab.set_default_index(vocab['<unk>'])

# takes in a review (string), returns a vector of indices of length T
def text_transform(x, T=50):
    indices = [vocab[token] for token in tokenizer(x)]
    return indices[:T] if len(indices) > T else indices + [0] * (T-len(indices)) 

# raw labels are 1-4, ['World', 'Sports', 'Business', 'Sci/Tech']. Make them 0-3.
def label_transform(x):
    return x - 1

# returns X, y tensors from training data
def create_tensors(train_data):
    label_list, text_list = [], []
    for idx, (label, text) in enumerate(train_data):
        label_list.append(label_transform(label))
        processed_text = torch.tensor(text_transform(text))
        text_list.append(processed_text)
    return torch.stack(text_list), torch.tensor(label_list)

X, y = create_tensors(train_data)
X_test,y_test = create_tensors(test_data)
# create dataloader. See https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
dataset = TensorDataset(X, y)
train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
dataset_test = TensorDataset(X_test,y_test)
test_dataloader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

model=MLP(len(vocab),embedding_dim=20,hidden_dim=64,output_dim=4)
loss_history=ModelTrainLoop(model,train_dataloader,num_epochs=50)
true_labels,predicted_labels=MultiClassification(model,test_dataloader)
plotLossHistory(loss_history)
confusion = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion)
plot_confusion_matrix2(true_labels,predicted_labels,class_names=['class1','class2','class3','class4'])
#plot_confusion_matrix(confusion, classes=['class1','class2','class3','class4'], title='Confusion Matrix', normalize=False)
plt.show()

