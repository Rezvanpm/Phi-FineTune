from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsLogger:
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="binary"),
            "Recall": recall_score(y_true, y_pred, average="binary"),
            "F1 Score": f1_score(y_true, y_pred, average="binary"),
        }
