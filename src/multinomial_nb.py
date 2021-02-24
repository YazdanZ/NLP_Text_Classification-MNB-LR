from classifier import Classifier
import numpy as np
from cross_validation import cross_validation_split, kfoldCV


class MultinomialNB(Classifier):
    """Multinomial naive bayes classifier"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # get all available classes
        self._classes = np.unique(y)
        self._priors = np.zeros(len(self._classes))
        self._likelihoods = np.zeros((len(self._classes), n_features))

        for index, cls in enumerate(self._classes):
            cls = int(cls)
            # calculate priors
            X_class = X[cls == y]
            self._priors[cls] = X_class.shape[0] / float(n_samples)
            # calculate likelihoods
            self._likelihoods[index, :] = (X_class.sum(axis=0) + self.alpha) / np.sum(
                X_class.sum(axis=0) + self.alpha
            )

    def predict(self, X_test):
        self.prediction = [self._predict(x) for x in X_test]
        return self.prediction

    def _predict(self, x_test):
        posts = []
        for index, cls in enumerate(self._classes):
            prior = np.log(self._priors[index])
            likelihood = np.log(self._likelihoods[index, :]) * x_test.toarray()
            posteriors = np.sum(likelihood) + prior
            posts.append(posteriors)

        return self._classes[np.argmax(posts)]

    @staticmethod
    def tune_hyperparams(X_train, y_train, folds, alphas):
        sets = cross_validation_split(X_train, y_train, folds)
        accuracies = []
        for alpha in alphas:
            accuracy = kfoldCV(MultinomialNB, sets, alpha=alpha)
            accuracies.append(accuracy)

        return accuracies
