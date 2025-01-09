import unittest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ModelTraining.src.metrics import accuracy, classification, confusion, model_evaluating

class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        # Sample true and predicted values for testing
        self.y_true = [1, 0, 1, 1, 0, 1, 0]
        self.y_pred = [1, 0, 1, 0, 0, 1, 1]

        # Initialize model_evaluating with default evaluator
        self.evaluator = model_evaluating(evaluator=accuracy)

    def test_accuracy(self):
        # Test accuracy evaluation
        self.evaluator.evaluator = accuracy
        result = self.evaluator.evaluate_model(self.y_true, self.y_pred)
        expected = accuracy_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, expected, msg="Accuracy metric is incorrect.")

    def test_classification_report(self):
        # Test classification report evaluation
        self.evaluator.evaluator = classification
        result = self.evaluator.evaluate_model(self.y_true, self.y_pred)
        expected = classification_report(self.y_true, self.y_pred, output_dict=True)
        self.assertEqual(result, expected, msg="Classification report is incorrect.")

    def test_confusion_matrix(self):
        # Test confusion matrix evaluation
        self.evaluator.evaluator = confusion
        result = self.evaluator.evaluate_model(self.y_true, self.y_pred)
        expected = confusion_matrix(self.y_true, self.y_pred)
        self.assertTrue((result == expected).all(), msg="Confusion matrix is incorrect.")

if __name__ == '__main__':
    unittest.main()
