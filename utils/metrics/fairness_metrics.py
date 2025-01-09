import numpy as np

class FairnessMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, outcome_col='True Outcome'):
        """
        :param cfg: Configuration containing model type, etc.
        :param suggestions_df: DataFrame with columns for decisions and outcomes
        :param decision_col: Column name for the model's predicted decisions
        :param outcome_col: Column name for the true outcome labels or value 
        """
        # Basic checks
        if decision_col not in suggestions_df.columns:
            raise ValueError(f"Column {decision_col} not found in the DataFrame")
        if outcome_col not in suggestions_df.columns:
            raise ValueError(f"Column {outcome_col} not found in the DataFrame")

        self.cfg = cfg
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.outcome_col = outcome_col

        self.group_col = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.positive_group_value = cfg.case_specific_metrics.positive_group_value

        self.actions_set = cfg.actions_outcomes.actions_set  
        self.positive_actions_set = cfg.actions_outcomes.positive_actions_set  
        self.all_decisions = suggestions_df[decision_col].unique()

        self.outcomes_set = cfg.actions_outcomes.outcomes_set 
        self.positive_outcomes_set= cfg.actions_outcomes.positive_outcomes_set  

    def get_metrics(self, fairness_metrics_list):
        available_metrics = {
            'Demographic Parity': self._compute_demographic_parity,
            'Equal Opportunity': self._compute_equal_opportunity,
            'Conditional Outcome Parity': self._compute_cond_outcome_parity
        }

        selected_metrics = {}
        for metric in fairness_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")

        return selected_metrics
    
    def _compute_demographic_parity(self):
        df_pos = self.suggestions_df[self.suggestions_df[self.group_col] == self.positive_group_value]
        df_neg = self.suggestions_df[self.suggestions_df[self.group_col] != self.positive_group_value]

        pos_decision = self.positive_actions_set[0]  # e.g. "Grant"

        p_pos_group = (df_pos[self.decision_col] == pos_decision).mean() if len(df_pos) else np.nan
        p_neg_group = (df_neg[self.decision_col] == pos_decision).mean() if len(df_neg) else np.nan

        dp_diff = p_pos_group - p_neg_group

        return dp_diff 

    def _compute_equal_opportunity(self):
        df_pos_outcome = self.suggestions_df[
            (self.suggestions_df[self.group_col] == self.positive_group_value) & 
            (self.suggestions_df[self.outcome_col] == self.positive_outcomes_set[0])
        ]
        df_neg_outcome = self.suggestions_df[
            (self.suggestions_df[self.group_col] != self.positive_group_value) & 
            (self.suggestions_df[self.outcome_col] == self.positive_outcomes_set[0])
        ]

        pos_decision = self.positive_actions_set[0]

        tpr_pos_group = (df_pos_outcome[self.decision_col] == pos_decision).mean() if len(df_pos_outcome) else np.nan
        tpr_neg_group = (df_neg_outcome[self.decision_col] == pos_decision).mean() if len(df_neg_outcome) else np.nan

        eo_diff = tpr_pos_group - tpr_neg_group

        return eo_diff


    def _compute_cond_outcome_parity(self):
        pos_decision = self.positive_actions_set[0]
        df_pred_pos = self.suggestions_df[self.suggestions_df[self.decision_col] == pos_decision]

        df_pos_pred_pos = df_pred_pos[df_pred_pos[self.group_col] == self.positive_group_value]
        df_neg_pred_pos = df_pred_pos[df_pred_pos[self.group_col] != self.positive_group_value]

        cal_pos_group = (df_pos_pred_pos[self.outcome_col] == self.positive_outcomes_set[0]).mean() if len(df_pos_pred_pos) else np.nan
        cal_neg_group = (df_neg_pred_pos[self.outcome_col] == self.positive_outcomes_set[0]).mean() if len(df_neg_pred_pos) else np.nan

        cal_diff = cal_pos_group - cal_neg_group

        return cal_diff 
