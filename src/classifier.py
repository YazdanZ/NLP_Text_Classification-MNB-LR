import numpy as np
from abc import ABC, abstractmethod


class Classifier(ABC):
    @staticmethod
    def evaluate_acc(true_labels, target_labels):
        """Evaluate the model accuracy"""
        accuracy = np.sum(true_labels == target_labels) / len(true_labels)
        return accuracy

    @abstractmethod
    def tune_hyperparams(self):
        pass
