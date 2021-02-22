from classifier import Classifier
import numpy as np


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
            # calculate priors
            X_class = X[cls == y]
            self._priors[cls] = X_class.shape[0] / float(n_samples)
            # calculate likelihoods
            self._likelihoods[index, :] = (X_class.sum(axis=0) + self.alpha) / np.sum(
                X_class.sum(axis=0) + self.alpha
            )

    def predict(self, X_test):
        prediction = [self._predict(x) for x in X_test]
        return prediction

    def _predict(self, x_test):
        posts = []
        for index, cls in enumerate(self._classes):
            prior = np.log(self._priors[index])
            likelihood = self.get_likelihood(self._likelihoods[index, :], x_test)
            posteriors = np.sum(likelihood) + prior
            posts.append(posteriors)

        return self._classes[np.argmax(posts)]

    def get_likelihood(self, class_likelihood, x):
        return np.log(class_likelihood) * x.toarray()

    def tune_hyperparams(self):
        pass
