from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from abc import ABC, abstractmethod


class model_evaluating(ABC):
    @abstractmethod
    def evaluate():
        pass

class model_evaluation(model_evaluating):
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        # Generate the classification report as a dictionary
        classification_report_dict = classification_report(y_test, y_pred_classes, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred_classes)
        confusion_matrix_result = confusion_matrix(y_test, y_pred_classes)

        return classification_report_dict, accuracy, confusion_matrix_result


class model_evaluator:

    def __init__(self, evaluator = model_evaluating):
        self.evaluator = evaluator

    def evaluate_model(self, model, X_test, y_test):
        self.evaluator.evaluate(model, X_test, y_test)