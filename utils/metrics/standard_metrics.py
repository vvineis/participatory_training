from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class StandardMetrics:
    def __init__(self, suggestions_df, decision_col, true_outcome_col=None, actions_set=None, outcomes_set=None, causal_reg_outcome_cols:list=None, model_type:str=None):
        """
        Initialize the class for both classification and causal regression evaluation.
        :param suggestions_df: DataFrame containing suggestions and outcomes.
        :param decision_col: Column indicating the model's decision.
        :param true_outcome_col: Column for true labels (classification use case).
        :param actions_set: Set of actions (e.g., 'A', 'C') for mapping decisions.
        :param outcomes_set: Set of outcomes (e.g., 0, 1) for mapping classification outcomes.
        :param a_outcome_col: Column with outcomes for action A (causal regression).
        :param c_outcome_col: Column with outcomes for action C (causal regression).
        """
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col if true_outcome_col else None
        self.a_outcome_col = causal_reg_outcome_cols[0] if causal_reg_outcome_cols else None
        self.c_outcome_col = causal_reg_outcome_cols[1] if causal_reg_outcome_cols else None
        self.model_type = model_type

        self.decision_mapping = {decision: i for i, decision in enumerate(actions_set)} if actions_set else None
        self.outcome_mapping = {outcome: i for i, outcome in enumerate(outcomes_set)} if outcomes_set else None
        
        self._y_pred = None
        self._y_true = None
        self._is_correct = None

    @property
    def y_pred(self):
        """
        For classification: Map decisions to numeric labels.
        """
        if self._y_pred is None and self.decision_mapping:
            self._y_pred = self.suggestions_df[self.decision_col].map(self.decision_mapping).astype(int)
        return self._y_pred

    @property
    def y_true(self):
        """
        For classification: Map true outcomes to numeric labels.
        """
        if self._y_true is None and self.outcome_mapping:
            self._y_true = self.suggestions_df[self.true_outcome_col].map(self.outcome_mapping).astype(int)
        return self._y_true

    @property
    def is_correct(self):
        """
        For causal regression: Determine if the decision minimizes the outcome.
        """
        if self._is_correct is None and self.a_outcome_col and self.c_outcome_col:
            self._is_correct = self.suggestions_df.apply(
                lambda row: (
                    (row[self.decision_col] == 'A' and row[self.a_outcome_col] <= row[self.c_outcome_col]) or
                    (row[self.decision_col] == 'C' and row[self.c_outcome_col] <= row[self.a_outcome_col])
                ),
                axis=1
            )
        return self._is_correct

    def get_metrics(self, standard_metrics_list):
        """
        Compute and return selected metrics.
        """
        metrics = {}

        # Classification metrics
        if self.model_type == 'classification':
            classification_metric_functions = {
                'Precision': lambda: precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'Recall': lambda: recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'F1 Score': lambda: f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'Accuracy': lambda: accuracy_score(self.y_true, self.y_pred)
            }
            for metric in standard_metrics_list:
                if metric in classification_metric_functions:
                    metrics[metric] = classification_metric_functions[metric]()
        elif self.model_type == 'causal_regression':
            metrics['Accuracy'] = self.is_correct.mean()
            metrics['Proportion Correct A Decisions'] = self.is_correct[self.suggestions_df[self.decision_col] == 'A'].mean()
            metrics['Proportion Correct C Decisions'] = self.is_correct[self.suggestions_df[self.decision_col] == 'C'].mean()

            # Regret
            regret = self.suggestions_df.apply(
                lambda row: min(row[self.a_outcome_col], row[self.c_outcome_col]) -
                            (row[self.a_outcome_col] if row[self.decision_col] == 'A' else row[self.c_outcome_col]),
                axis=1
            )
            metrics['Mean Regret'] = regret.mean()

        return metrics


'''class StandardMetrics:
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

        return selected_metrics'''
