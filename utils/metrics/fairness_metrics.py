import numpy as np
import pandas as pd

class FairnessMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, outcome_col='True Outcome'):
        if decision_col not in suggestions_df.columns:
            raise ValueError(f"Column {decision_col} not found in the DataFrame")

        self.cfg = cfg
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col

        if self.cfg.models.outcome.model_type == 'regression':
            self.action_outcomes = {
                action: f"{action}_predicted_outcome_binary"
                for action in cfg.actions_outcomes.actions_set
            }
            self.outcome_col = None  # Not directly used in regression
            self.outcomes_set = [0, 1]  # Assume binary outcomes for fairness metrics
        else:
            self.action_outcomes = None
            self.outcome_col = outcome_col
            self.outcomes_set = cfg.actions_outcomes.outcomes_set

        self.group_col = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.positive_group_value = cfg.case_specific_metrics.positive_group_value
        self.actions_set = cfg.actions_outcomes.actions_set
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
        for action in self.actions_set:
            for outcome_value in self.outcomes_set:
                if self.cfg.models.outcome.model_type == 'regression':
                    outcome_col = self.action_outcomes[action]
                else:
                    outcome_col = self.outcome_col

                # Positive group rates
                rates['positive'][(action, outcome_value)] = (
                    (positive_group[self.decision_col] == action) &
                    (positive_group[outcome_col] == outcome_value)
                ).mean() if len(positive_group) > 0 else np.nan

                # Negative group rates
                rates['negative'][(action, outcome_value)] = (
                    (negative_group[self.decision_col] == action) &
                    (negative_group[outcome_col] == outcome_value)
                ).mean() if len(negative_group) > 0 else np.nan

        return rates

    def compute_demographic_parity(self):
        parity_metrics = {}
        for action in self.actions_set:
            for outcome_value in self.outcomes_set:
                parity_metrics[f'{action} Parity ({outcome_value})'] = self._calculate_parity(action, outcome_value)

        if len(self.actions_set) > 1:
            parity_metrics['Positive Action Parity'] = np.nanmean(list(parity_metrics.values()))

        return parity_metrics

    def compute_equal_opportunity(self):
        opportunity_metrics = {}
        for action in self.actions_set:
            for outcome_value in self.outcomes_set:
                opportunity_metrics[f'TPR {action} Parity ({outcome_value})'] = self._calculate_parity(action, outcome_value)

        if len(self.actions_set) > 1:
            opportunity_metrics['TPR Positive Outcome Parity'] = np.nanmean(list(opportunity_metrics.values()))

        return opportunity_metrics

    def compute_equalized_odds(self):
        odds_metrics = {}
        for action in self.actions_set:
            for outcome_value in self.outcomes_set:
                odds_metrics[f'Equalized Odds {action} ({outcome_value})'] = self._calculate_parity(action, outcome_value)

        if len(self.actions_set) > 1:
            odds_metrics['Average Equalized Odds'] = np.nanmean(list(odds_metrics.values()))

        return odds_metrics

    def compute_calibration(self):
        calibration_metrics = {}
        for action in self.actions_set:
            for outcome_value in self.outcomes_set:
                calibration_metrics[f'{action} Calibration ({outcome_value})'] = self._calculate_parity(action, outcome_value)

        if len(self.actions_set) > 1:
            calibration_metrics['Average Calibration'] = np.nanmean(list(calibration_metrics.values()))

        return calibration_metrics

    def _calculate_parity(self, action, outcome_value):
        return self.rates['positive'][(action, outcome_value)] - self.rates['negative'][(action, outcome_value)]

    def _calculate_odds_difference(self, action, positive_outcome, negative_outcome):
        pos_diff = self.rates['positive'][(action, positive_outcome)] - self.rates['negative'][(action, positive_outcome)]
        neg_diff = self.rates['positive'][(action, negative_outcome)] - self.rates['negative'][(action, negative_outcome)]
        return pos_diff - neg_diff

    def get_metrics(self, fairness_metrics_list):
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
        
        return selected_metrics



'''class FairnessMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, outcome_col='True Outcome'):
        # Ensure the column exists in the DataFrame
        if decision_col not in suggestions_df.columns or outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {outcome_col} not found in the DataFrame")

        # Assign the config attributes directly
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.outcome_col = outcome_col

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

