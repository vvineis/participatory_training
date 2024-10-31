import numpy as np
import pandas as pd

class FairnessMetrics:
    def __init__(self, suggestions_df, decision_col, group_col, positive_group_value, outcome_col='True Outcome'):
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.group_col = group_col
        self.positive_group_value = positive_group_value
        self.outcome_col = outcome_col
        self._rates = None 

    @property
    def rates(self):
        if self._rates is None:
            self._rates = self._compute_rates()
        return self._rates

    def _compute_rates(self):
        positive_group = self.suggestions_df[self.suggestions_df[self.group_col] == self.positive_group_value]
        negative_group = self.suggestions_df[self.suggestions_df[self.group_col] != self.positive_group_value]

        rates = {'positive': {}, 'negative': {}}
        for decision_value in ['Grant', 'Grant lower']:
            for outcome_value in ['Fully Repaid', 'Partially Repaid', 'Not Repaid']:
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
        grant_parity = self._calculate_parity('Grant', 'Fully Repaid')
        grant_lower_parity = self._calculate_parity('Grant lower', 'Partially Repaid')
        positive_action_parity = grant_parity + grant_lower_parity

        return {
            'Grant Parity': grant_parity,
            'Grant lower Parity': grant_lower_parity,
            'Positive Action Parity': positive_action_parity
        }

    def compute_equal_opportunity(self):
        tpr_fully_repaid_parity = self._calculate_parity('Grant', 'Fully Repaid')
        tpr_partially_repaid_parity = self._calculate_parity('Grant lower', 'Partially Repaid')
        
        tpr_positive_outcome_parity = (
            np.nanmean([tpr_fully_repaid_parity, tpr_partially_repaid_parity])
        ) if not any(np.isnan([tpr_fully_repaid_parity, tpr_partially_repaid_parity])) else np.nan

        return {
            'TPR Fully Repaid Parity': tpr_fully_repaid_parity,
            'TPR Partially Repaid Parity': tpr_partially_repaid_parity,
            'TPR Positive Outcome Parity': tpr_positive_outcome_parity
        }

    def compute_equalized_odds(self):
        fully_repaid_equalized_odds = self._calculate_odds_difference('Grant', 'Fully Repaid', 'Not Repaid')
        partially_repaid_equalized_odds = self._calculate_odds_difference('Grant lower', 'Partially Repaid', 'Not Repaid')

        average_equalized_odds = (
            np.nanmean([fully_repaid_equalized_odds, partially_repaid_equalized_odds])
        ) if not any(np.isnan([fully_repaid_equalized_odds, partially_repaid_equalized_odds])) else np.nan

        return {
            'Equalized Odds Fully Repaid': fully_repaid_equalized_odds,
            'Equalized Odds Partially Repaid': partially_repaid_equalized_odds,
            'Average Equalized Odds': average_equalized_odds
        }

    def compute_calibration(self):
        grant_calibration = self._calculate_parity('Grant', 'Fully Repaid')
        grant_lower_calibration = self._calculate_parity('Grant lower', 'Partially Repaid')

        average_calibration = np.nanmean([grant_calibration, grant_lower_calibration])

        return {
            'Grant Calibration (Fully Repaid)': grant_calibration,
            'Grant lower Calibration (Partially Repaid)': grant_lower_calibration,
            'Average Calibration': average_calibration
        }

    def _calculate_parity(self, decision, outcome):
        return self.rates['positive'][(decision, outcome)] - self.rates['negative'][(decision, outcome)]

    def _calculate_odds_difference(self, decision, positive_outcome, negative_outcome):
        pos_diff = self.rates['positive'][(decision, positive_outcome)] - self.rates['negative'][(decision, positive_outcome)]
        neg_diff = self.rates['positive'][(decision, negative_outcome)] - self.rates['negative'][(decision, negative_outcome)]
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
