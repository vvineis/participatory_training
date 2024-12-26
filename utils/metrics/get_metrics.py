from utils.metrics.standard_metrics import StandardMetrics
from utils.metrics.fairness_metrics import FairnessMetrics
from utils.metrics.real_payoffs import RealPayoffMetrics
from utils.rewards.get_rewards import RewardCalculator
from utils.metrics.case_specific_metrics import HealthCaseMetrics, LendingCaseMetrics


class MetricsCalculator:
    def __init__(self, cfg):
        self.cfg = cfg  # Store the config for direct access

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

        # Pre-compute and cache common fairness metrics and action counts
        fairness_cache = {}
        action_counts_cache = {}

        for actor in self.cfg.actors.actor_list + self.cfg.decision_criteria:
            decision_col = f'{actor} Suggested Action' if actor in self.cfg.actors.actor_list else actor

            if decision_col not in suggestions_df.columns:
                continue
            
                
            # Fairness Metrics: Cache results for each decision column
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
                    print(suggestions_df[[decision_col,'A_outcome_binary','C_outcome_binary', 'Predicted_Binary']])
                    fairness_calculator = FairnessMetrics(cfg=self.cfg, suggestions_df=suggestions_df, decision_col=decision_col, outcome_col='Predicted_Binary')
                    actor_metrics_calculator = HealthCaseMetrics(suggestions_df, decision_col, true_outcome_col, self.cfg)
                
                fairness_cache[decision_col] = fairness_calculator.get_metrics(self.cfg.fairness.fairness_metrics)
                print(f'case_metrics:{actor_metrics_calculator}')
                print(f'fairness_metrics:{fairness_cache[decision_col]}')


            # Action Counts: Cache action percentages for efficiency
            if decision_col not in action_counts_cache:
                action_counts_cache[decision_col] = suggestions_df[decision_col].value_counts(normalize=True).to_dict()

            # Instantiate the class
            print(self.cfg.case_specific_metrics.metrics)
            for metric in self.cfg.case_specific_metrics.metrics:
                if metric == 'Total Profit':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_profit()
                elif metric == 'Unexploited Profit':
                    metrics[actor][metric] = actor_metrics_calculator.compute_unexploited_profit()
                elif metric == 'Total Loss':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_loss()
                elif metric == 'Total Cost':
                    metrics[actor][metric] = actor_metrics_calculator.compute_total_cost()
                elif metric=='Avg_no_recovery_weeks':
                    metrics[actor][metric] = actor_metrics_calculator.compute_avg_no_recovery_weeks()
            print(f'case_metrics:{metrics[actor]}')

            # Standard Metrics
            standard_metrics_calculator = StandardMetrics(
                suggestions_df, decision_col, true_outcome_col, 
                self.cfg.actions_outcomes.actions_set, self.cfg.actions_outcomes.outcomes_set
            )
            standard_metrics = standard_metrics_calculator.get_metrics(self.cfg.standard_metrics)

            for metric in self.cfg.standard_metrics:
                metrics[actor][metric] = standard_metrics.get(metric, None)

            # Fairness Metrics: Retrieve from cache and dynamically update based on actions and outcomes
            fairness_metrics = fairness_cache[decision_col]
            # After computing fairness metrics
            print(f"Fairness metrics for actor: {actor}")
            print(f"Demographic Parity: {fairness_metrics.get('Demographic Parity')}")
            print(f"Equal Opportunity: {fairness_metrics.get('Equal Opportunity')}")
            for k, v in fairness_metrics.items():
                metrics[actor][k] = v
            print(metrics[actor])


            '''# Retrieve all positive action parities dynamically
            positive_actions = self.cfg.actions_outcomes.positive_actions_set  # List of positive actions
            positive_action_parities = []

            # Collect parities for all positive actions
            for action in positive_actions:
                parity_value = fairness_metrics['Demographic Parity'].get(f'{action} Parity', None)
                positive_action_parities.append(parity_value)

            # Add any additional metrics like 'Positive Action Parity'
            positive_action_parities.append(fairness_metrics['Demographic Parity'].get('Positive Action Parity', None))

            # Compute the worst-case demographic parity
            metrics[actor]['Demographic Parity_Worst'] = self._compute_max_min_fairness(*positive_action_parities)

            
            metrics[actor]['Equal Opportunity_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Equal Opportunity'].get(f'TPR {self.cfg.actions_outcomes.outcomes_set[0]} Parity'),
                fairness_metrics['Equal Opportunity'].get(f'TPR {self.cfg.actions_outcomes.outcomes_set[1]} Parity')
            )
            metrics[actor]['Equalized Odds_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Equalized Odds'].get(f'Equalized Odds {self.cfg.actions_outcomes.outcomes_set[0]}'),
                fairness_metrics['Equalized Odds'].get(f'Equalized Odds {self.cfg.actions_outcomes.outcomes_set[1]}')
            )
            metrics[actor]['Calibration_Worst'] = self._compute_max_min_fairness(
                fairness_metrics['Calibration'].get(f'{self.cfg.actions_outcomes.positive_actions_set[0]} Calibration ({self.cfg.actions_outcomes.outcomes_set[0]})'),
                fairness_metrics['Calibration'].get(f'{self.cfg.actions_outcomes.positive_actions_set[1]} Calibration ({self.cfg.actions_outcomes.outcomes_set[1]})')
            )'''

            # Action Percentages: Retrieve from cache and dynamically update per actor
            action_counts = action_counts_cache[decision_col]
            metrics[actor].update({
                f'Percent_{action}': action_counts.get(action, 0) for action in self.cfg.actions_outcomes.actions_set
            })

            for reward_actor in self.cfg.actors.reward_types:
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
