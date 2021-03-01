from abc import ABC, abstractmethod


class Classifier(ABC):
    def evaluate_acc(self, target_labels, true_labels):
        """Evaluate the model accuracy"""
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
