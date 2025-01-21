from utils.metrics.standard_metrics import StandardMetrics
from utils.metrics.fairness_metrics import FairnessMetrics
from utils.metrics.case_specific_metrics import HealthCaseMetrics, LendingCaseMetrics
from utils.metrics.real_payoffs import RealPayoffMetrics
from utils.rewards.get_rewards import RewardCalculator


class MetricsCalculator:
    """
    This class computes all metrics for the decision suggestions output by the decision strategies"""
    def __init__(self, cfg):
        self.cfg = cfg  

    def merge_predicted(self, suggestions_df, decision_col):
        def select_pred(row):
            if row[decision_col] == 'A':
                return row['A_outcome_binary']
            else:
                return row['C_outcome_binary']
        suggestions_df['Predicted_Binary'] = suggestions_df.apply(select_pred, axis=1)
        return suggestions_df

    @staticmethod
    def _compute_max_min_fairness(*values):
        return max(abs(value) for value in values if value is not None)

    def compute_all_metrics(self, suggestions_df, true_outcome_col='True Outcome'):
        metrics = {actor: {} for actor in self.cfg.actors.actor_list + self.cfg.decision_criteria}

        fairness_cache = {}
        action_counts_cache = {}

        for actor in self.cfg.actors.actor_list + self.cfg.decision_criteria:
            decision_col = f'{actor} Suggested Action' if actor in self.cfg.actors.actor_list else actor

            if decision_col not in suggestions_df.columns:
                continue
            
            if decision_col not in fairness_cache:
                if self.cfg.models.outcome.model_type == 'classification':
                    fairness_calculator = FairnessMetrics(cfg=self.cfg, suggestions_df=suggestions_df, decision_col=decision_col)
                    actor_metrics_calculator = LendingCaseMetrics(
                    suggestions_df, decision_col, true_outcome_col)

                else:
                    for action in self.cfg.actions_outcomes.actions_set:
                        suggestions_df[f"{action}_outcome_binary"] = (
                        suggestions_df[f"{action}_outcome"] <= self.cfg.case_specific_metrics.threshold_outcome
                            ).astype(int)
                    suggestions_df = self.merge_predicted(suggestions_df, decision_col)
                    fairness_calculator = FairnessMetrics(cfg=self.cfg, suggestions_df=suggestions_df, decision_col=decision_col, outcome_col='Predicted_Binary')
                    actor_metrics_calculator = HealthCaseMetrics(suggestions_df, decision_col, true_outcome_col, self.cfg)
                
            fairness_cache[decision_col] = fairness_calculator.get_metrics(self.cfg.fairness_metrics)

            # Action Counts
            if decision_col not in action_counts_cache:
                action_counts_cache[decision_col] = suggestions_df[decision_col].value_counts(normalize=True).to_dict()

            # Case Specific Metrics
            for metric in self.cfg.case_specific_metrics.metrics:
                try:
                    metrics[actor][metric] = actor_metrics_calculator.get_metrics([metric])[metric]
                except ValueError as e:
                    print(f"Error computing metric '{metric}': {e}")

            # Standard Metrics
            if self.cfg.models.outcome.model_type == 'classification':
                standard_metrics_calculator = StandardMetrics(
                    suggestions_df, decision_col, true_outcome_col, 
                    self.cfg.actions_outcomes.actions_set, self.cfg.actions_outcomes.outcomes_set, model_type='classification'
                )
            elif self.cfg.models.outcome.model_type == 'causal_regression':
                standard_metrics_calculator = StandardMetrics(
                    suggestions_df, decision_col, true_outcome_col,
                    causal_reg_outcome_cols=[f'{action}_outcome' for action in self.cfg.actions_outcomes.actions_set],  model_type='causal_regression'
                )

            standard_metrics = standard_metrics_calculator.get_metrics(self.cfg.standard_metrics)

            for metric in self.cfg.standard_metrics:
                metrics[actor][metric] = standard_metrics.get(metric, None)

            fairness_metrics = fairness_cache[decision_col]
            for k, v in fairness_metrics.items():
                metrics[actor][k] = v

            action_counts = action_counts_cache[decision_col]
            metrics[actor].update({
                f'Percent_{action}': action_counts.get(action, 0) for action in self.cfg.actions_outcomes.actions_set
            })

            for reward_actor in self.cfg.actors.reward_types:
                if self.cfg.models.outcome.model_type == 'classification':
                    reward_structures = RewardCalculator.REWARD_STRUCTURES

                    payoff_metrics_calculator = RealPayoffMetrics(
                        cfg=self.cfg, suggestions_df=suggestions_df, decision_col=decision_col,
                        true_outcome_col=true_outcome_col,
                        reward_actor=reward_actor, reward_structures=reward_structures
                        )
                    total_real_payoff = payoff_metrics_calculator.compute_total_real_payoff()
       
                # Store the payoff under reward_actor's metrics but using the context of `actor`
                    metrics[actor][f'Total Real Payoff ({reward_actor})'] = total_real_payoff
                else:
                    continue
            
        return metrics
