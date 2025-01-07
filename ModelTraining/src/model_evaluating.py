from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from abc import ABC, abstractmethod


class model_evaluating(ABC):
    @abstractmethod
    def evaluate():
        pass

class model_evaluation(model_evaluating):

    def evaluate(model, X_test, y_test):
        y_pred = model.predict_classes(X_test)
        classification_report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        confusion_matrix = confusion_matrix(y_test, y_pred)

        return classification_report, accuracy, confusion_matrix

class model_evaluator:

    def __init__(self, evaluator = model_evaluating):
        self.evaluator = evaluator

    def evaluate_model(self, model, X_test, y_test):
        self.evaluator.evaluate(model, X_test, y_test)