from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from abc import ABC, abstractmethod


class Evluation_metrics(ABC):
    @abstractmethod
    def evaluate(y_true, y_pred):
        pass

class accuracy(Evluation_metrics):
    def evaluate(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class classification(Evluation_metrics):
    def evaluate(y_true, y_pred):
        return classification_report(y_true, y_pred)

class confusion(Evluation_metrics):
    def evaluate(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

class model_evaluating:
    def __init__(self, evaluator = Evluation_metrics):
        self.evaluator = evaluator

    def evaluate_model(self, y_true, y_pred):
        return self.evaluator.evaluate(y_true, y_pred)