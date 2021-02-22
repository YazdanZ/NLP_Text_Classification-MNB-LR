from abc import ABC, abstractmethod


class Classifier(ABC):
    @staticmethod
    def evaluate_acc(true_labels, target_labels):
        """Evaluate the model accuracy"""
        sum = 0
        for y in range(len(true_labels)):
            if true_labels[y] == target_labels[y]:
                sum += 1
        accuracy = sum / float(len(true_labels))
        return accuracy

    @abstractmethod
    def tune_hyperparams(self):
        pass
