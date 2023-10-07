X.size()= torch.Size([100, 5]) #100->examples 5-> features(including "1" features)
Y.size()= torch.Size([100])

if use matrix operation, do I need to use unsqueeze or squeeze?

Torch NOTES:

1. torch.cat(tensors, dim=0, out=None)
concatenate multiple tensors, dim=0->row, dim=1->column

2. train_data, test_data = torch.split(X, [80, 20])
[80,20]->[train_data size, test_data size]
    ! For labels and features, I can also use from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3. y_pred.view(-1) #change the shape of the tensor. -1 means automatically calculated by torch, maybe keep it as 1d tensor

4. Use torch.nn to define a model
# Define the Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):# initial function
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer with input_dim input features and 1 output feature
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):# when we call the model and pass the data to this model, this function will be called automatically
        return self.sigmoid(self.linear(x))
5. optimizer = optim.SGD(model.parameters(), lr=lr) 
    Setting up a SGD optimizer
    model.parameters() includes the weight and bias

6. keep shape the same, use flatten() for those array with 1 dimension,like(20,1)

7. _, predicted = torch.max(outputs.data, 1) # 1 means at the column dimension
    _ the max value of each column
    predicted, the index of  the max value of each column




