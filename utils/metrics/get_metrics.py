from utils.metrics.standard_metrics import StandardMetrics
from utils.metrics.fairness_metrics import FairnessMetrics
from utils.metrics.case_specific_metrics import CaseMetrics
from utils.metrics.real_payoffs import RealPayoffMetrics
from utils.rewards.get_rewards import RewardCalculator

class MetricsCalculator:
    def __init__(self, cfg):
        self.cfg = cfg  # Store the config for direct access

    @staticmethod
    def _compute_max_min_fairness(value1, value2, value3=None):
        """ Compute max-min fairness as the maximum absolute difference from 0. """
        if value3 is not None:
            return max(abs(value1), abs(value2), abs(value3))
        return max(abs(value1), abs(value2))

    def compute_all_metrics(self, suggestions_df, true_outcome_col='True Outcome'):
        metrics = {actor: {} for actor in self.cfg.setting.actor_list + self.cfg.criteria.decision_criteria}

        # Pre-compute and cache common fairness metrics and action counts
        fairness_cache = {}
        action_counts_cache = {}

        for actor in self.cfg.setting.actor_list + self.cfg.criteria.decision_criteria:
            decision_col = f'{actor} Suggested Action' if actor in self.cfg.setting.actor_list else actor

            if decision_col not in suggestions_df.columns:
                continue

            # Fairness Metrics: Cache results for each decision column
            if decision_col not in fairness_cache:
                fairness_calculator = FairnessMetrics(cfg=self.cfg, suggestions_df=suggestions_df, decision_col=decision_col)
                fairness_cache[decision_col] = fairness_calculator.get_metrics(self.cfg.metrics.fairness_metrics)

            # Action Counts: Cache action percentages for efficiency
            if decision_col not in action_counts_cache:
                action_counts_cache[decision_col] = suggestions_df[decision_col].value_counts(normalize=True).to_dict()

            # Actor-Specific Metrics
            actor_metrics_calculator = CaseMetrics(suggestions_df, decision_col, true_outcome_col)
            for metric in self.cfg.metrics.case_specific_metrics:
                if metric == 'Total Profit':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_profit()
                elif metric == 'Unexploited Profit':
                    metrics[actor][metric] = actor_metrics_calculator.compute_unexploited_profit()
                elif metric == 'Total Loss':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_loss()

            # Standard Metrics
            standard_metrics_calculator = StandardMetrics(
                suggestions_df, decision_col, true_outcome_col, 
                self.cfg.setting.actions_set, self.cfg.setting.outcomes_set
            )
            standard_metrics = standard_metrics_calculator.get_metrics(self.cfg.metrics.standard_metrics)

            for metric in self.cfg.metrics.standard_metrics:
                metrics[actor][metric] = standard_metrics.get(metric, None)

            # Fairness Metrics: Retrieve from cache and dynamically update based on actions and outcomes
            fairness_metrics = fairness_cache[decision_col]
            for metric_name, metric_values in fairness_metrics.items():
                for key, value in metric_values.items():
                    metrics[actor][f'{metric_name}_{key}'] = value

            # Max-Min Fairness Calculations for Fairness Metrics
            grant_parity = fairness_metrics['Demographic Parity'].get(f'{self.cfg.setting.positive_actions_set[0]} Parity')
            grant_lower_parity = fairness_metrics['Demographic Parity'].get(f'{self.cfg.setting.positive_actions_set[1]} Parity')
            positive_action_parity = fairness_metrics['Demographic Parity'].get('Positive Action Parity')

            metrics[actor]['Demographic Parity_Worst'] = self._compute_max_min_fairness(
                grant_parity, grant_lower_parity, positive_action_parity
            )
            
            metrics[actor]['Equal Opportunity_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Equal Opportunity'].get(f'TPR {self.cfg.setting.outcomes_set[0]} Parity'),
                fairness_metrics['Equal Opportunity'].get(f'TPR {self.cfg.setting.outcomes_set[1]} Parity')
            )
            metrics[actor]['Equalized Odds_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Equalized Odds'].get(f'Equalized Odds {self.cfg.setting.outcomes_set[0]}'),
                fairness_metrics['Equalized Odds'].get(f'Equalized Odds {self.cfg.setting.outcomes_set[1]}')
            )
            metrics[actor]['Calibration_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Calibration'].get(f'{self.cfg.setting.positive_actions_set[0]} Calibration ({self.cfg.setting.outcomes_set[0]})'),
                fairness_metrics['Calibration'].get(f'{self.cfg.setting.positive_actions_set[1]} Calibration ({self.cfg.setting.outcomes_set[1]})')
            )

            # Action Percentages: Retrieve from cache and dynamically update per actor
            action_counts = action_counts_cache[decision_col]
            metrics[actor].update({
                f'Percent_{action}': action_counts.get(action, 0) for action in self.cfg.setting.actions_set
            })

            for reward_actor in self.cfg.setting.reward_types:
                reward_structures = RewardCalculator.REWARD_STRUCTURES

                payoff_metrics_calculator = RealPayoffMetrics(
                    cfg=self.cfg, suggestions_df=suggestions_df, decision_col=decision_col,
                    true_outcome_col=true_outcome_col,
                    reward_actor=reward_actor, reward_structures=reward_structures
                    )
                total_real_payoff = payoff_metrics_calculator.compute_total_real_payoff()
                
                # Store the payoff under reward_actor's metrics but using the context of `actor`
                metrics[actor][f'Total Real Payoff ({reward_actor})'] = total_real_payoff

        return metrics
