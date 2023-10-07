import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
def pltsemi(train_losses,test_losses):
    # Create a list of iteration counts (e.g., 1, 2, 3, ..., num_iterations)
    iterations = list(range(1, len(train_losses) + 1))

    # Create a semilog plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.semilogx(iterations, train_losses, label='Training Loss', marker='o')
    plt.semilogx(iterations, test_losses, label='Testing Loss', marker='x')

    # Add labels and a legend
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.title('Loss vs. Iteration Count')
    
def confux(model,X_test,y_test,CLASSIFIERS=2):
    # Assuming you have a trained logistic regression model named 'model'
    # and testing data 'X_test' and 'y_test'

    # Make predictions on the testing data
    with torch.no_grad():
        if CLASSIFIERS==2:    
            model.eval()  # Set the model to evaluation mode
            y_pred = model(X_test)
        else:
            outputs = model(X_test)
            _, y_pred = torch.max(outputs.data, 1) #
    
    if CLASSIFIERS==2:
        # Convert probability predictions to binary labels (0 or 1)
        y_pred_binary = (y_pred >= 0.5).float()

        # Convert tensors to NumPy arrays
        y_true = y_test.numpy()
        y_pred = y_pred_binary.numpy()
    else:
        y_true = y_test.numpy()
        y_pred = y_pred.numpy()
    # Generate a confusion matrix
    y_true.flatten()
    y_pred=y_pred.flatten()
    #print("y_pred.size()=",y_pred.shape,y_pred)
    #print("y_true.size()=",y_true.shape,y_true)
    ConfuseMatrix=np.zeros((CLASSIFIERS,CLASSIFIERS))
    sum=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            sum+=1
            ConfuseMatrix[int(y_true[i]),int(y_pred[i])]+=1
        else:
            ConfuseMatrix[int(y_true[i]),int( y_pred[i])] += 1
    Accuracy=sum/(len(y_pred))
    print("Accuracy=",Accuracy)
    #PrintConfuseMatrix(ConfuseMatrix,int(data.size()[0]))
    print("confusion",ConfuseMatrix.shape)
    # Plot the confusion matrix using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(ConfuseMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    if CLASSIFIERS==2:
        classes = ["Class 0", "Class 1"]
    else:
        classes = ["Class 0", "Class 1","Class 2"]
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(ConfuseMatrix[i][j]), horizontalalignment="center", color="white" if ConfuseMatrix[i][j] > ConfuseMatrix.max() / 2 else "black")

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    # Print a classification report for more metrics
    #report = classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"])
    #print("Classification Report:\n", report)

