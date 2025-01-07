from abc import ABC, abstractmethod


class model_evaluating(ABC):
    @abstractmethod
    def evaluate():
        pass

class model_evaluation(model_evaluating):
    def evaluate(model, X_test):
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")

        
        y_pred = model.predict(X_test)
        # Check model summary and input shape
        print(model.summary())
        print(f"Expected input shape: {model.input_shape}")
        print(f"Raw predictions (first 5): {y_pred[:5] if y_pred is not None else 'None'}")
        # Validate y_pred
        if y_pred is None:
            print("The model's predict method returned None.")
            raise ValueError("The model's predict method returned None.")

        # Convert predictions to binary classes
        y_pred_classes = (y_pred > 0.5).astype(int)
        print(f"Raw predictions (first 5): {y_pred_classes[:5] if y_pred is not None else 'None'}")

        return y_pred_classes


class model_evaluator:

    def __init__(self, evaluator = model_evaluating):
        self.evaluator = evaluator

    def evaluate_model(self, model, X_test):
        return self.evaluator.evaluate(model, X_test)