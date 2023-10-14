import numpy as np
import matplotlib.pyplot as plt


seed = 215*2
C = 3    # 3 classes
Nc = 30  # 30 data points per class
N = Nc * C   

mean = [[0, 0], [0, 4], [4, 4]]
cov1 = np.array([[4, 1], [1, 2]])
cov2 = np.array([[2, 0], [0, 1]])
cov3 = np.eye(2)
cov = [cov1, cov2, cov3]

# generate data by sampling from 3 gaussians
rng = np.random.default_rng(seed)
X0 = rng.multivariate_normal(mean[0], cov[0], Nc)
X1 = rng.multivariate_normal(mean[1], cov[1], Nc)
X2 = rng.multivariate_normal(mean[2], cov[2], Nc)

X = np.concatenate((X0, X1, X2))      # store sample points
y = np.concatenate((0*np.ones(Nc), 1*np.ones(Nc), 2*np.ones(Nc)))  # class labels
print("X shape:",X.shape) #X shape: (90, 2)  N*D
print("y shape:",y.shape) #y shape: (90,)    N*1

# plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y)

#from here logistic classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Assuming you have loaded the dataset and labels as X and y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_testFIX=y_test
# Normalize the data (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a multiclass logistic classifier
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
Logy_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, Logy_pred)
conf_matrix = confusion_matrix(y_test, Logy_pred)
class_report = classification_report(y_test, Logy_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

plt.figure(0)
# Visualize the decision boundaries (optional)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')



#from here produce a Gaussian classifier
from sklearn.mixture import GaussianMixture

# Assuming X and y contain the data and labels
num_components = 1  # Number of Gaussian components in each class's GMM, here I only use 1 componenet, it means my model with only 1 peak for each class
gmm_models = []

for c in range(3):  # Loop through all classes
    X_c = X_train[y_train == c]  # Extract data points for class c
    #print("X_c",X_c)
    #print("y_train == c",y_train == c)
    gmm = GaussianMixture(n_components=num_components)
    gmm.fit(X_c)
    gmm_models.append(gmm)

def classify_data_point(x):
    class_posteriors = []
    for c, gmm in enumerate(gmm_models):  # Loop through all classes and their respective GMMs
        likelihood = gmm.score_samples(x.reshape(1, -1))
        prior = 1 / 3  # Assuming a uniform prior for all classes
        marginal_likelihood = sum(np.exp(gmm_k.score_samples(x.reshape(1, -1))) for gmm_k in gmm_models)
        posterior = likelihood + np.log(prior) - np.log(marginal_likelihood)
        class_posteriors.append(posterior)
    
    predicted_class = np.argmax(class_posteriors)
    return predicted_class
def gmm_predict(X):
    # Initialize an empty array to store predicted labels
    predicted_labels = []

    # Loop through the data points
    for x in X:
        # Assign the class label with the highest likelihood as the prediction
        predicted_label =classify_data_point(x)
        predicted_labels.append(predicted_label)

    return np.array(predicted_labels)
# Evaluate the classifier
GMM_y_pred=gmm_predict(X_test)
# Evaluate the classifier
print("y_pred.shape:",GMM_y_pred.shape,", y_test.shape:",y_test.shape)
accuracy = accuracy_score(y_test, GMM_y_pred)
conf_matrix = confusion_matrix(y_test, GMM_y_pred)
class_report = classification_report(y_test, GMM_y_pred)

# Print the results
print("-----------------------------------")
print("##Gaussian Classifier##")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


#From here for MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = 4
num_classes = 3  # 3 classes
model = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam (model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()

def MLPclassifier():
    y_pred=[]
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            #y_pred.append(predicted.item())
        y_pred=predicted.tolist()
        y_pred=np.array(y_pred)
        return y_pred

print("MLP Evaluation")
# Evaluate the classifier
MLP_y_pred=MLPclassifier()
print("MLP_y_pred=",MLP_y_pred)
accuracy = accuracy_score(y_testFIX, MLP_y_pred)
conf_matrix = confusion_matrix(y_testFIX, MLP_y_pred)
class_report = classification_report(y_testFIX, MLP_y_pred)

# Print the results
print("-----------------------------------")
print("##MLP Classifier##")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools



# Confusion draw function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
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

# classes label
class_labels = ["Class 0", "Class 1", "Class 2"]

# for logistic
Log_cm = confusion_matrix(y_testFIX, Logy_pred)
# for GMM
GMM_cm = confusion_matrix(y_testFIX, GMM_y_pred)
# for MLP
MLP_cm = confusion_matrix(y_testFIX, MLP_y_pred)
# 绘制混淆矩阵
plot_confusion_matrix(Log_cm, classes=class_labels, title='Logistic Confusion Matrix', normalize=False)
plot_confusion_matrix(GMM_cm, classes=class_labels, title='Gaussian  Matrix', normalize=False)
plot_confusion_matrix(MLP_cm, classes=class_labels, title='MLP Matrix', normalize=False)
plt.show()