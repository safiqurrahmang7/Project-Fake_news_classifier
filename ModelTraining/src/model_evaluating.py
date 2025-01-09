from abc import ABC, abstractmethod


class model_evaluating(ABC):
    @abstractmethod
    def evaluate():
        pass

class model_evaluation(model_evaluating):
    def evaluate(model, X_test):
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
        y_pred = model.predict(X_test)
        # Convert predictions to binary classes
        y_pred_classes = (y_pred > 0.5).astype(int)
        

        return y_pred_classes


class model_evaluator:

    def __init__(self, evaluator = model_evaluating):
        self.evaluator = evaluator

    def evaluate_model(self, model, X_test):
        return self.evaluator.evaluate(model, X_test)