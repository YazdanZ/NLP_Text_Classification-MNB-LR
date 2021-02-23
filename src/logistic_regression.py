from classifier import Classifier
import numpy as np
from cross_validation import cross_validation_split, kfoldCV


class LogisticRegression(Classifier):
    def __init__(self, learning_rate=0.1, num_iters=100):
        self.learning_rate = learning_rate
        self.num_iters = num_iters

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.bias = 0
        self.weights = np.zeros(num_features)

        # grad descent
        for iteration in range(self.num_iters):
            y_pred = self._linear_model(X)

            # backpropagation
            dw = 1 / num_samples * np.dot(X.T, y_pred - y)
            db = 1 / num_samples * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X_test):
        y_pred = self._linear_model(X_test)
        return [0 if y < 0.5 else 1 for y in y_pred]

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _linear_model(self, X):
        # hypothetesis prediction
        lin_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(lin_model)

    @staticmethod
    def tune_hyperparams(X_train, y_train, folds):
        pass
