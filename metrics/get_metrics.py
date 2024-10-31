from metrics.standard_metrics import StandardMetrics
from metrics.fairness_metrics import FairnessMetrics
from metrics.case_specific_metrics import CaseMetrics

class MetricsCalculator:
    def __init__(self, fairness_metrics_list, standard_metrics_list, case_metrics_list, actions_set, outcomes_set, positive_actions_set):
        self.fairness_metrics_list = fairness_metrics_list
        self.standard_metrics_list = standard_metrics_list
        self.case_metrics_list = case_metrics_list
        self.actions_set = actions_set
        self.outcomes_set = outcomes_set
        self.positive_actions_set=positive_actions_set

    @staticmethod
    def _compute_max_min_fairness(value1, value2, value3=None):
        """ Compute max-min fairness as the maximum absolute difference from 0. """
        if value3 is not None:
            return max(abs(value1), abs(value2), abs(value3))
        return max(abs(value1), abs(value2))

    def compute_all_metrics(self, suggestions_df, actor_list, decision_criteria_list, positive_attribute_for_fairness, true_outcome_col='True Outcome'):
        metrics = {actor: {} for actor in actor_list + decision_criteria_list}
        positive_attribute_for_fairness = positive_attribute_for_fairness
        positive_group_value = 1      # Group considered vulnerable

        # Pre-compute and cache common fairness metrics and action counts
        fairness_cache = {}
        action_counts_cache = {}

        for actor in actor_list + decision_criteria_list:
            decision_col = f'{actor} Suggested Action' if actor in actor_list else actor

            if decision_col not in suggestions_df.columns:
                continue

            # Fairness Metrics: Cache results for each decision column
            if decision_col not in fairness_cache:
                fairness_calculator = FairnessMetrics(
                    suggestions_df, decision_col, positive_attribute_for_fairness, positive_group_value, 
                    true_outcome_col, positive_actions_set=self.positive_actions_set, outcomes_set=self.outcomes_set
                )
                fairness_cache[decision_col] = fairness_calculator.get_metrics(self.fairness_metrics_list)

            # Action Counts: Cache action percentages for efficiency
            if decision_col not in action_counts_cache:
                action_counts_cache[decision_col] = suggestions_df[decision_col].value_counts(normalize=True).to_dict()

            # Actor-Specific Metrics
            actor_metrics_calculator = CaseMetrics(suggestions_df, decision_col, true_outcome_col)
            for metric in self.case_metrics_list:
                if metric == 'Total Profit':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_profit()
                elif metric == 'Unexploited Profit':
                    metrics[actor][metric] = actor_metrics_calculator.compute_unexploited_profit()
                elif metric == 'Total Loss':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_loss()

            # Standard Metrics
            standard_metrics_calculator = StandardMetrics(suggestions_df, decision_col, true_outcome_col)
            standard_metrics = standard_metrics_calculator.compute_all_metrics()
            for metric in self.standard_metrics_list:
                metrics[actor][metric] = standard_metrics.get(metric, None)

            # Fairness Metrics: Retrieve from cache and dynamically update based on actions and outcomes
            fairness_metrics = fairness_cache[decision_col]
            for metric_name, metric_values in fairness_metrics.items():
                for key, value in metric_values.items():
                    metrics[actor][f'{metric_name}_{key}'] = value

            # Max-Min Fairness Calculations for Fairness Metrics
            metrics[actor]['Demographic Parity_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Demographic Parity'].get(f'{self.actions_set[0]} Parity'),
                fairness_metrics['Demographic Parity'].get(f'{self.actions_set[1]} Parity'),
                fairness_metrics['Demographic Parity'].get('Positive Action Parity')
            )
            metrics[actor]['Equal Opportunity_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Equal Opportunity'].get(f'TPR {self.outcomes_set[0]} Parity'),
                fairness_metrics['Equal Opportunity'].get(f'TPR {self.outcomes_set[1]} Parity')
            )
            metrics[actor]['Equalized Odds_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Equalized Odds'].get(f'Equalized Odds {self.outcomes_set[0]}'),
                fairness_metrics['Equalized Odds'].get(f'Equalized Odds {self.outcomes_set[1]}')
            )
            metrics[actor]['Calibration_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Calibration'].get(f'{self.actions_set[0]} Calibration ({self.outcomes_set[0]})'),
                fairness_metrics['Calibration'].get(f'{self.actions_set[1]} Calibration ({self.outcomes_set[1]})')
            )

            # Action Percentages: Retrieve from cache and dynamically update per actor
            action_counts = action_counts_cache[decision_col]
            metrics[actor].update({
                f'Percent_{action}': action_counts.get(action, 0) for action in self.actions_set
            })

        return metrics
