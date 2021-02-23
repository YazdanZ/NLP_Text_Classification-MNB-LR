from classifier import Classifier
import numpy as np
from cross_validation import cross_validation_split, kfoldCV


class LogisticRegression(Classifier):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X_test):
        pass

    @staticmethod
    def tune_hyperparams(X_train, y_train, folds):
        pass
