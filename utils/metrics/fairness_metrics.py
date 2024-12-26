import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class FairnessMetrics:
    """
    Corrected version for classification with standard fairness definitions:
    - Demographic Parity
    - Equal Opportunity (TPR Parity)
    - Equalized Odds (TPR & FPR Parity)
    - Calibration (basic discrete approach)
    """

    def __init__(self, cfg, suggestions_df, decision_col, outcome_col='True Outcome'):
        """
        :param cfg: Configuration containing model type, etc.
        :param suggestions_df: DataFrame with columns for decisions and outcomes
        :param decision_col: Column name for the model's predicted decisions
        :param outcome_col: Column name for the true outcome labels (classification)
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

        # Identify the protected attribute and "positive" group
        self.group_col = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.positive_group_value = cfg.case_specific_metrics.positive_group_value

        # Typically, define which decision is considered "positive"
        # or interpret multiple "positive" decisions from the config
        self.actions_set = cfg.actions_outcomes.actions_set  
        self.positive_actions_set = cfg.actions_outcomes.positive_actions_set  
        self.all_decisions = suggestions_df[decision_col].unique()


        # Identify outcome labels. E.g., for binary classification: [0,1]
        self.outcomes_set = cfg.actions_outcomes.outcomes_set  # e.g. [0,1]
        self.positive_outcomes_set= cfg.actions_outcomes.positive_outcomes_set  # e.g. [1]

    def get_metrics(self, fairness_metrics_list):
        """
        Main entry point: compute the requested metrics (Demographic Parity, Equal Opportunity, etc.).
        """
        available_metrics = {
            'Demographic Parity': self._compute_demographic_parity,
            'Equal Opportunity': self._compute_equal_opportunity,
            'Equalized Odds': self._compute_equalized_odds,
            'Calibration': self._compute_calibration
        }

        selected_metrics = {}
        for metric in fairness_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")

        return selected_metrics

    ############################################################################
    # 1. Demographic Parity
    ############################################################################
    def _compute_demographic_parity(self):
        """
        Demographic Parity: 
        DP difference = P(Decision=Positive) in PositiveGroup - P(Decision=Positive) in NegativeGroup
        """
        # Filter groups
        df_pos = self.suggestions_df[self.suggestions_df[self.group_col] == self.positive_group_value]
        df_neg = self.suggestions_df[self.suggestions_df[self.group_col] != self.positive_group_value]

        # For demonstration, assume there's a single positive decision
        # If you have multiple positive decisions, you can average or compare them individually
        pos_decision = self.positive_actions_set[0]  # e.g. "Grant"

        # Probability of receiving the positive decision in each group
        p_pos_group = (df_pos[self.decision_col] == pos_decision).mean() if len(df_pos) else np.nan
        p_neg_group = (df_neg[self.decision_col] == pos_decision).mean() if len(df_neg) else np.nan

        dp_diff = p_pos_group - p_neg_group

        return dp_diff #{
            #f'Demographic Parity ({pos_decision})': dp_diff
       # }

    ############################################################################
    # 2. Equal Opportunity (TPR Parity)
    ############################################################################
    def _compute_equal_opportunity(self):
        # Filter for outcome=1 (the truly positive individuals)
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

        return eo_diff #{
            #f'Equal Opportunity (TPR diff for {pos_decision})': eo_diff
        #}

    ############################################################################
    # 3. Equalized Odds (TPR & FPR Parity)
    ############################################################################
    def _compute_equalized_odds(self):
        pos_decision = self.positive_actions_set[0]

        # TPR (outcome=1)
        df_pos_outcome_1 = self.suggestions_df[
            (self.suggestions_df[self.group_col] == self.positive_group_value) & 
            (self.suggestions_df[self.outcome_col] == self.positive_outcomes_set[0])
        ]
        df_neg_outcome_1 = self.suggestions_df[
            (self.suggestions_df[self.group_col] != self.positive_group_value) & 
            (self.suggestions_df[self.outcome_col] == self.positive_outcomes_set[0])
        ]

        df_pos = self.suggestions_df[self.suggestions_df[self.group_col] == self.positive_group_value]
        df_neg = self.suggestions_df[self.suggestions_df[self.group_col] != self.positive_group_value]

        print("Positive group, outcome=1:", len(df_pos[df_pos[self.outcome_col] == 1]))
        print("Positive group, outcome=0:", len(df_pos[df_pos[self.outcome_col] == 0]))
        print("Negative group, outcome=1:", len(df_neg[df_neg[self.outcome_col] == 1]))
        print("Negative group, outcome=0:", len(df_neg[df_neg[self.outcome_col] == 0]))

        tpr_pos_group = (df_pos_outcome_1[self.decision_col] == pos_decision).mean() if len(df_pos_outcome_1) else np.nan
        tpr_neg_group = (df_neg_outcome_1[self.decision_col] == pos_decision).mean() if len(df_neg_outcome_1) else np.nan
        tpr_diff = tpr_pos_group - tpr_neg_group

        # FPR (outcome=0)
        df_pos_outcome_0 = self.suggestions_df[
            (self.suggestions_df[self.group_col] == self.positive_group_value) & 
            (self.suggestions_df[self.outcome_col] == 0)
        ]
        df_neg_outcome_0 = self.suggestions_df[
            (self.suggestions_df[self.group_col] != self.positive_group_value) & 
            (self.suggestions_df[self.outcome_col] == 0)
        ]

        fpr_pos_group = (df_pos_outcome_0[self.decision_col] == pos_decision).mean() if len(df_pos_outcome_0) else np.nan
        fpr_neg_group = (df_neg_outcome_0[self.decision_col] == pos_decision).mean() if len(df_neg_outcome_0) else np.nan
        fpr_diff = fpr_pos_group - fpr_neg_group

        return tpr_diff #{
            #f'EO TPR diff ({pos_decision})': tpr_diff,
            #f'EO FPR diff ({pos_decision})': fpr_diff
        #}

    ############################################################################
    # 4. Calibration (Simple Discrete Version)
    ############################################################################
    def _compute_calibration(self):
        pos_decision = self.positive_actions_set[0]
        df_pred_pos = self.suggestions_df[self.suggestions_df[self.decision_col] == pos_decision]

        # Positive group among predicted-positive
        df_pos_pred_pos = df_pred_pos[df_pred_pos[self.group_col] == self.positive_group_value]
        # Negative group among predicted-positive
        df_neg_pred_pos = df_pred_pos[df_pred_pos[self.group_col] != self.positive_group_value]

        # Rate of truly positive among predicted-positive
        # i.e. P(Outcome=1 | Predicted=Positive, Group=pos)
        # compare pos-group vs. neg-group
        cal_pos_group = (df_pos_pred_pos[self.outcome_col] == self.positive_outcomes_set[0]).mean() if len(df_pos_pred_pos) else np.nan
        cal_neg_group = (df_neg_pred_pos[self.outcome_col] == self.positive_outcomes_set[0]).mean() if len(df_neg_pred_pos) else np.nan

        cal_diff = cal_pos_group - cal_neg_group

        return cal_diff #{
           # f'Calibration diff ({pos_decision})': cal_diff
        #}


'''class FairnessMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, outcome_col='True Outcome'):
        # Ensure the column exists in the DataFrame
        if decision_col not in suggestions_df.columns or outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {outcome_col} not found in the DataFrame")

        # Assign the config attributes directly
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.outcome_col = outcome_col
        print(f"outcome_col {self.suggestions_df.outcome_col.head()}")

        # Config-specified values
        self.group_col = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.positive_group_value = cfg.case_specific_metrics.positive_group_value
        self.actions_set = cfg.actions_outcomes.positive_actions_set
        self.outcomes_set = cfg.actions_outcomes.outcomes_set

        # Cache for computed rates
        self._rates = None 

    @property
    def rates(self):
        if self._rates is None:
            self._rates = self._compute_rates()
        return self._rates

    def _compute_rates(self):
        # Define groups based on the positive attribute
        positive_group = self.suggestions_df[self.suggestions_df[self.group_col] == self.positive_group_value]
        negative_group = self.suggestions_df[self.suggestions_df[self.group_col] != self.positive_group_value]

        rates = {'positive': {}, 'negative': {}}
        for decision_value in self.actions_set:
            for outcome_value in self.outcomes_set:
                # Calculate mean occurrence rates
                rates['positive'][(decision_value, outcome_value)] = (
                    (positive_group[self.outcome_col] == outcome_value) & 
                    (positive_group[self.decision_col] == decision_value)
                ).mean() if len(positive_group) > 0 else np.nan
                rates['negative'][(decision_value, outcome_value)] = (
                    (negative_group[self.outcome_col] == outcome_value) & 
                    (negative_group[self.decision_col] == decision_value)
                ).mean() if len(negative_group) > 0 else np.nan
        
        df_pos = self.suggestions_df[self.suggestions_df[self.group_col] == self.positive_group_value]
        df_neg = self.suggestions_df[self.suggestions_df[self.group_col] != self.positive_group_value]

        print("Positive group, outcome=1:", len(df_pos[df_pos[self.outcome_col] == 1]))
        print("Positive group, outcome=0:", len(df_pos[df_pos[self.outcome_col] == 0]))
        print("Negative group, outcome=1:", len(df_neg[df_neg[self.outcome_col] == 1]))
        print("Negative group, outcome=0:", len(df_neg[df_neg[self.outcome_col] == 0]))

        return rates

    def compute_demographic_parity(self):
        # Calculate demographic parity metrics
        if len(self.actions_set)>2: # adapt to more actions
            first_pos_parity = self._calculate_parity(self.actions_set[0], self.outcomes_set[0])
            second_pos_parity = self._calculate_parity(self.actions_set[1], self.outcomes_set[1])
            positive_action_parity = first_pos_parity + second_pos_parity

            return {
                f'{self.actions_set[0]} Parity': first_pos_parity,
                f'{self.actions_set[1]} Parity': second_pos_parity,
                'Positive Action Parity': positive_action_parity
            }
        else: 
            positive_action_parity = self._calculate_parity(self.actions_set[0], self.outcomes_set[0])

            return {
                'Positive Action Parity': positive_action_parity
            }

    def compute_equal_opportunity(self):
        # Calculate equal opportunity metrics
        if len(self.actions_set)>2:
            tpr_fully_repaid_parity = self._calculate_parity(self.actions_set[0], self.outcomes_set[0])
            tpr_partially_repaid_parity = self._calculate_parity(self.actions_set[1], self.outcomes_set[1])
            
            tpr_positive_outcome_parity = (
                np.nanmean([tpr_fully_repaid_parity, tpr_partially_repaid_parity])
            ) if not any(np.isnan([tpr_fully_repaid_parity, tpr_partially_repaid_parity])) else np.nan

            return {
                f'TPR {self.outcomes_set[0]} Parity': tpr_fully_repaid_parity,
                f'TPR {self.outcomes_set[1]} Parity': tpr_partially_repaid_parity,
                'TPR Positive Outcome Parity': tpr_positive_outcome_parity
            }
        else:
            tpr_positive_outcome_parity = self._calculate_parity(self.actions_set[0], self.outcomes_set[0])

            return {
                'TPR Positive Outcome Parity': tpr_positive_outcome_parity}


    def compute_equalized_odds(self):
        if len(self.actions_set)>2:
            # Calculate equalized odds metrics
            fully_repaid_equalized_odds = self._calculate_odds_difference(self.actions_set[0], self.outcomes_set[0], self.outcomes_set[2])
            partially_repaid_equalized_odds = self._calculate_odds_difference(self.actions_set[1], self.outcomes_set[1], self.outcomes_set[2])

            average_equalized_odds = (
                np.nanmean([fully_repaid_equalized_odds, partially_repaid_equalized_odds])
            ) if not any(np.isnan([fully_repaid_equalized_odds, partially_repaid_equalized_odds])) else np.nan

            return {
                f'Equalized Odds {self.outcomes_set[0]}': fully_repaid_equalized_odds,
                f'Equalized Odds {self.outcomes_set[1]}': partially_repaid_equalized_odds,
                'Average Equalized Odds': average_equalized_odds
            }
        else:
            # Calculate equalized odds metrics
            pos_outcome_equalized_odds = self._calculate_odds_difference(self.actions_set[0], self.outcomes_set[0], self.outcomes_set[1])

            return {
                f'Equalized Odds {self.outcomes_set[0]}': pos_outcome_equalized_odds
            }
            

    def compute_calibration(self):
         if len(self.actions_set)>2:
        # Calculate calibration metrics
            grant_calibration = self._calculate_parity(self.actions_set[0], self.outcomes_set[0])
            grant_lower_calibration = self._calculate_parity(self.actions_set[1], self.outcomes_set[1])

            average_calibration = np.nanmean([grant_calibration, grant_lower_calibration])

            return {
                f'{self.actions_set[0]} Calibration ({self.outcomes_set[0]})': grant_calibration,
                f'{self.actions_set[1]} Calibration ({self.outcomes_set[1]})': grant_lower_calibration,
                'Average Calibration': average_calibration
            }
         else:
            # Calculate calibration metrics
            pos_outcome_calibration = self._calculate_parity(self.actions_set[0], self.outcomes_set[0])

            return {
                f'{self.actions_set[0]} Calibration ({self.outcomes_set[0]})': pos_outcome_calibration
            }

    def _calculate_parity(self, decision, outcome):
        # Calculate parity for a given decision-outcome pair
        return self.rates['positive'][(decision, outcome)] - self.rates['negative'][(decision, outcome)]

    def _calculate_odds_difference(self, decision, positive_outcome, negative_outcome):
        # Calculate odds difference for a given decision and outcomes
        pos_diff = self.rates['positive'][(decision, positive_outcome)] - self.rates['negative'][(decision, positive_outcome)]
        neg_diff = self.rates['positive'][(decision, negative_outcome)] - self.rates['negative'][(decision, negative_outcome)]
        return pos_diff - neg_diff

    def get_metrics(self, fairness_metrics_list):
        # Select and compute only the requested metrics
        available_metrics = {
            'Demographic Parity': self.compute_demographic_parity,
            'Equal Opportunity': self.compute_equal_opportunity,
            'Equalized Odds': self.compute_equalized_odds,
            'Calibration': self.compute_calibration
        }

        selected_metrics = {}
        for metric in fairness_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")
        
        print(f"selected_metrics {selected_metrics}")
        print(f"fairness_metrics_list {fairness_metrics_list}")

        return selected_metrics'''

