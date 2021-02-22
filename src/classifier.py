from abc import ABC, abstractmethod


class Classifier(ABC):
    def evaluate_acc(self, target_labels, true_labels=None):
        """Evaluate the model accuracy"""
        if true_labels == None:
            true_labels = self.prediction
        sum = 0
        for y in range(len(true_labels)):
            if true_labels[y] == target_labels[y]:
                sum += 1
        accuracy = sum / float(len(true_labels))
        return accuracy

    @staticmethod
    @abstractmethod
    def tune_hyperparams():
        pass
