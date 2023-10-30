from utilis import *
from NeuralNetwork import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
X,y=LoadData(path=".\data\mnist_train.csv")
X_test,y_test=LoadData(path=".\data\mnist_test.csv")
print("X.dtype=",X.dtype)
dataset = TensorDataset(X, y)
train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
dataset_test = TensorDataset(X_test,y_test)
test_dataloader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

CNN_model=LeNet5()
loss_history=ModelTrainLoop(CNN_model,train_dataloader,num_epochs=100)
true_labels,predicted_labels=MultiClassification(CNN_model,test_dataloader)

plotLossHistory(loss_history)
confusion = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion)
plot_confusion_matrix2(true_labels,predicted_labels,class_names=['class1','class2','class3','class4','class5','class6','class7','class8','class9','class10'])
plt.show()
