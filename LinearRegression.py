import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = []

    def fit(self, train_X=None, train_y=None):
        XtX = np.dot(train_X.T, train_X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(train_X.T, train_y)
        self.weights = np.dot(XtX_inv, Xty)

    def predict(self, val_X):
        prediction = np.dot(val_X, self.weights)
        return prediction
