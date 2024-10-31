from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class StandardMetrics:
    def __init__(self, suggestions_df, decision_col, true_outcome_col, actions_set, outcomes_set):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col

        self.decision_mapping = {decision: i for i, decision in enumerate(actions_set)}
        self.outcome_mapping = {outcome: i for i, outcome in enumerate(outcomes_set)}
        
        self._y_pred = None
        self._y_true = None


    @property
    def y_pred(self):
        if self._y_pred is None:
            self._y_pred = self.suggestions_df[self.decision_col].map(self.decision_mapping).fillna(0).astype(int)
        return self._y_pred

    @property
    def y_true(self):
        if self._y_true is None:
            self._y_true = self.suggestions_df[self.true_outcome_col].map(self.outcome_mapping).fillna(0).astype(int)
        return self._y_true

    def compute_precision(self):
        return precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)

    def compute_recall(self):
        return recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)

    def compute_f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)

    def compute_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def compute_all_metrics(self):
        return {
            'Precision': self.compute_precision(),
            'Recall': self.compute_recall(),
            'F1 Score': self.compute_f1_score(),
            'Accuracy': self.compute_accuracy()
        }

    def get_metrics(self, standard_metrics_list):
        """Return a dictionary of selected metrics based on a provided list of metric names."""
        available_metrics = {
            'Precision': self.compute_precision,
            'Recall': self.compute_recall,
            'F1 Score': self.compute_f1_score,
            'Accuracy': self.compute_accuracy
        }

        selected_metrics = {}
        for metric in standard_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")

        return selected_metrics
