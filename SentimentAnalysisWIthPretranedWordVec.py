import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader
from collections import Counter
from torchinfo import summary
from NetworkPredictor import *
from sklearn.metrics import confusion_matrix

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


#print("Vocabulary size: ", len(vocab))
#print(train_data[0][1])   # print the text of first review


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
#print(f"X.size()={X.size()},y.size()={y.size()}")#X.size()=torch.Size([25000, 256]),y.size()=torch.Size([25000])
print(f"Test_X.size()={Test_X.size()},Test_y.size()={Test_y.size()}")#X.size()=torch.Size([25000, 256]),y.size()=torch.Size([25000])
#print("X[0]:",X[0])

# create dataloader. See https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
dataset = TensorDataset(X, y)
train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


# build the embedding matrix for our vocabulary, from pretrained word vectors, in fact, I don't need to build the embedding, I can just use the existing embedding
def build_embedding(vocab, vectors):
    # loop over all vocab indices, for each get token, get embedding (from token), add it to list
    itos = vocab.get_itos()    # List mapping indices to tokens.
    embed_list = [vectors[tok] for tok in itos]
    return torch.stack(embed_list)

W = build_embedding(vocab, vectors)
#print(f"W.size()={W.size()}")#W.size()=torch.Size([20437, 50])  with 20437 vocabularies,each word represented as 50 dims vector
#print("W[0]:",W[0])#W[0]: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0.])

# Averaging word embeddings for each review
def average_word_embeddings(review,embedding):#the para review is a tensor with 256 indices for it
    wordVecs = embedding[review]
    return torch.mean(wordVecs, dim=0)#dim=0 means calculate element at each row, after that I get a row vector
def average_review(data,embedding):
    averageVec=[]
    for review in data:
        averageVec.append(average_word_embeddings(review,embedding))
    return torch.stack(averageVec,dim=0)
    

## testx_vec=average_word_embeddings(X[0],W)#print(f"testx_vec.size={testx_vec.size()}")#prediction is [1,50] for each review, for total train data, it should be 25000*50
avergaeX=average_review(X,W)#print(f"avergaeX.size={avergaeX.size()}")#prediction is [1,50] for each review, for total train data, it should be 25000*50
avergeTestX=average_review(Test_X,W)

avergaeX.to(torch.float32)
avergeTestX.to(torch.float32)
y.to(torch.float32)

averge_train_dataSet = TensorDataset(avergaeX, y)
averge_train_dataloader = DataLoader(averge_train_dataSet, batch_size=512, shuffle=True)
averge_test_dataSet = TensorDataset(avergeTestX, Test_y)
averge_test_dataloader = DataLoader(averge_test_dataSet, batch_size=len(averge_test_dataSet), shuffle=False)

Mymodel,train_losses=Sentimenttrainingloop(averge_train_dataloader)
predictlabel=SentimentValidFn(Mymodel,averge_test_dataloader)
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
# here I test my own neural network



# define the neural network
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.embed = nn.Embedding.from_pretrained(W)  # define params of embedding layer
    self.embed.weight.requires_grad = False       # freeze params, default
    self.linear1 = nn.Linear(W.shape[1], 32)      # define layer 1 parameters
    self.linear2 = nn.Linear(32, 1)               # define layer 2 parameters

  def forward(self, x):
    rdim = 0 if x.ndim == 1 else 1   # if x is a batch, reduction is along dim 1
    x = x.to(torch.long)             # change type (to get around bug in summary!)
    x = self.embed(x)                # returns tensor of shape (*, T, D)
    x = torch.mean(x, dim=rdim)      # average the embeddings
    x = self.linear1(x)              # first linear layer
    x = F.relu(x)                    # nonlinear activation, try tanh perhaps? 
    x = self.linear2(x)              # output logit
    return x
  
model = MLP()

summary(model, input_size=([10, 256,]))

plt.show()