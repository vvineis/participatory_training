from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
            self._y_pred = self.suggestions_df[self.decision_col].map(self.decision_mapping).astype(int)
        return self._y_pred

    @property
    def y_true(self):
        if self._y_true is None:
            self._y_true = self.suggestions_df[self.true_outcome_col].map(self.outcome_mapping).astype(int)
        return self._y_true

    def get_metrics(self, standard_metrics_list):
        """Compute and return selected metrics based on the provided list."""
        metric_functions = {
            'Precision': lambda: precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'Recall': lambda: recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'F1 Score': lambda: f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'Accuracy': lambda: accuracy_score(self.y_true, self.y_pred)
        }

        selected_metrics = {}
        for metric in standard_metrics_list:
            if metric in metric_functions:
                selected_metrics[metric] = metric_functions[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(metric_functions.keys())}.")

        return selected_metrics
