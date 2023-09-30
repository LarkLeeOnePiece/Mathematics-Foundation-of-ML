import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# read MNIST training data
def LoadData(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
    y = df.iloc[:, 0].to_numpy()                # labels of images
    y[y !=5] = 0
    y[y == 5] = 1
    return torch.tensor(X,dtype=torch.float64),torch.tensor(y,dtype=torch.float64)
def model(X,w):
    return torch.sigmoid(torch.matmul(X,w))
def loss_fn(y,yhat):
    return -torch.mean(y*torch.log(yhat)+(1-y)*torch.log(1-yhat))
def GraEx1(xpoint,pred,y):
    print(xpoint.size(),pred.size(),y.size())
    # Compute the gradient using the provided formula
    gradient=xpoint*(pred-y)
    print("gradient.size",gradient.size())
    return gradient

def Backward(xn,yn,W):
    pred=model(xn,W)
    f=loss_fn(yn,pred)
    f.backward()
    print("W.grad",W.grad.size())
    print(W.grad.resize(W.grad.size()[0]))

path = 'mnist_train.csv'
testpath = 'mnist_test.csv'
X,y=LoadData(path)
N,D=X.size()[0],X.size()[1]
W_FIX= torch.randn((D, 1), requires_grad=True, dtype=torch.float64)  # Modify the shape of W

W1=W_FIX
W2=W_FIX

max_iter = 100
lr = 1e-3
K=100
BS=32
# Define a function to compute a stochastic gradient
def stochastic_gradient(w=W1,batch_size=BS,iter=max_iter):
    for k in range(max_iter):
        # Zero out gradients for W
        indices = torch.randint(0, X.size()[0], (batch_size,))
        batch_images = X[indices]
        batch_labels = (y[indices])
        if w.grad is not None:
            w.grad.zero_()
        pred=model(batch_images,w)
        J = loss_fn(batch_labels.view(-1, 1), pred)  # Ensure y has the same shape as pred
        J.backward()
        w.data -= lr * w.grad  # Update W using data to avoid automatic differentiation
    return w.grad,pred
def FUllGradient(w=W2,iter=max_iter):
    for k in range(max_iter):
        if w.grad is not None:
            w.grad.zero_()
        pred=model(X,w)
        J = loss_fn(y.view(-1, 1), pred)  # Ensure y has the same shape as pred
        J.backward()
        w.data -= lr * w.grad  # Update W using data to avoid automatic differentiation
    return w.grad,pred
sum_stochastic_gradient =torch.tensor([torch.zeros_like(param) for param in W1]).resize(W1.size()[0],1)

for i in range(K):
    SGD,SDGpred=stochastic_gradient()
    #print("SGD.size()", SGD.size())
    #print("sum_stochastic_gradient.size()", sum_stochastic_gradient.size())
    sum_stochastic_gradient+=SGD
sum_stochastic_gradient=sum_stochastic_gradient/K
print("SDG",sum_stochastic_gradient.resize(sum_stochastic_gradient.size()[0],))
FullGra,pred=FUllGradient()
print("FullGra",FullGra.resize(FullGra.size()[0],))
# Compare full gradient and stochastic gradients
gradient_difference = torch.norm(sum_stochastic_gradient - FullGra,p=2)
print("Gradient Difference:", gradient_difference)
FullGra_norm = torch.norm(FullGra,p=2)
print("FullGra_norm:", FullGra_norm)
relative_gradient_difference = gradient_difference/torch.norm(FullGra,p=2)
print("relative_gradient_difference:", relative_gradient_difference)






