import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Classifier(ABC):
    @staticmethod
    def evaluate_acc(true_labels, target_labels):
        """Evaluate the model accuracy"""
        accuracy = np.sum(true_labels == target_labels) / len(true_labels)
        return accuracy

    @staticmethod
    def cross_validation_split(df, k):
        """Takes training data as input and splits it into k folds.
        Each fold is then used once as validation while the k − 1
        remaining folds form the training set"""
        # return random sample of items
        df.sample(frac=1)
        num_rows = df.shape[0]
        num_folds_row = num_rows // k

        sets = []
        for f in range(k):
            test = df[f * num_folds_row : f * num_folds_row + num_folds_row]
            train = df[~df.index.isin(test.index)]

            train_x, train_y = (
                train.drop("Class", axis=1).to_numpy(),
                train["Class"].to_numpy(),
            )
            validation_x, validation_y = (
                test.drop("Class", axis=1).to_numpy(),
                test["Class"].to_numpy(),
            )

            sets.append((train_x, train_y, validation_x, validation_y))

        return sets

    @staticmethod
    def kfoldCV(model, sets, **kwargs):
        """Takes train/validation sets generated above and a
        given model as input. The function iterates through each
        train/validation set and returns the average result of each fold."""
        classifier = model(**kwargs)
        accuracy = []
        for k_set in sets:
            classifier.fit(k_set[0], k_set[1])
            predictions = classifier.predict(k_set[2])
            accuracy.append(classifier.evaluate_acc(predictions, k_set[3]))
        return np.array(accuracy).mean()

    @abstractmethod
    def tune_hyperparams():
        pass
