import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def LeastSquareModel(X_train, y_train):
        # Step 1: Initialize and train the linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def LeastSquarePredictor(model,X_test):
    # Step 2: Make predictions on the test data.
    y_pred = model.predict(X_test)
    return y_pred

