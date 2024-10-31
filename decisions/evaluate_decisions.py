import pandas as pd

class SummaryProcessor:
    def __init__(self, metrics_calculator, ranking_criteria, ranking_weights, metrics_for_evaluation,
                 actor_list, decision_criteria_list, actions_set, outcomes_set, strategy):
        """
        Initialize the SummaryProcessor with necessary parameters, external objects, and the solution strategy.
        """
        self.metrics_calculator = metrics_calculator
        self.ranking_criteria = ranking_criteria
        self.ranking_weights = ranking_weights
        self.metrics_for_evaluation = metrics_for_evaluation
        self.actor_list = actor_list
        self.decision_criteria_list = decision_criteria_list
        self.actions_set = actions_set
        self.outcomes_set = outcomes_set
        self.strategy = strategy  # Use the provided strategy for computing actions

    def create_summary_df(self, X_val_reward, y_val_outcome, decision_solutions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list):
        """
        Create a summary DataFrame that includes the unscaled feature context, true outcomes,
        and suggested actions by different decision criteria and each actor.
        """
        # Unscaled feature context with true outcomes
        feature_context_df = unscaled_X_val_reward.copy()
        feature_context_df['True Outcome'] = y_val_outcome.values

        # Pivot decision solutions to show best actions by decision type
        decision_solutions_summary = decision_solutions_df.pivot(index='Row Index', columns='Decision Type', values='Best Action')
        summary_df = pd.concat([feature_context_df.reset_index(drop=True), decision_solutions_summary.reset_index(drop=True)], axis=1)

        # Initialize lists for suggested actions
        suggested_actions = {actor: [] for actor in self.actor_list}

        for idx, expected_rewards in enumerate(expected_rewards_list):
            # Use the strategy to compute individual actions for each actor
            individual_actions = self.strategy.compute(expected_rewards, disagreement_point=None, ideal_point=None, all_actions=self.actions_set)
            suggested_actions['Bank'].append(individual_actions['Bank']['action'])
            suggested_actions['Applicant'].append(individual_actions['Applicant']['action'])
            suggested_actions['Regulatory'].append(individual_actions['Regulatory']['action'])

            # Oracle suggested actions based on true outcomes
            true_outcome = y_val_outcome.iloc[idx]
            oracle_action = self._map_outcome_to_action(true_outcome)
            suggested_actions['Oracle'].append(oracle_action)

            # Classifier suggested actions based on classifier predictions
            clfr_action = self._map_outcome_to_action(clfr_pred_list[idx])
            suggested_actions['Classifier'].append(clfr_action)

        # Add suggested actions to summary DataFrame
        for actor in self.actor_list:
            summary_df[f'{actor} Suggested Action'] = suggested_actions[actor]

        return summary_df

    def _map_outcome_to_action(self, outcome):
        """Map outcomes to corresponding actions for oracle and classifier suggestions."""
        if outcome == 'Fully Repaid':
            return 'Grant'
        elif outcome == 'Not Repaid':
            return 'Not Grant'
        else:
            return 'Grant lower'

    def metrics_to_dataframe(self, metrics):
        """Convert a dictionary of metrics to a DataFrame."""
        rows = []
        for actor_or_criterion, metric_values in metrics.items():
            row = {'Actor/Criterion': actor_or_criterion}
            row.update(metric_values)
            rows.append(row)
        return pd.DataFrame(rows)

    def process_decision_metrics(self, suggestions_df, X_val_reward, y_val_outcome, decision_solutions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list):
        """
        Process and generate all decision metrics and return a dictionary of DataFrames and ranking results.
        """
        # Create summary DataFrame
        summary_df = self.create_summary_df(X_val_reward, y_val_outcome, decision_solutions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list)

        # Calculate decision metrics using MetricsCalculator
        decision_metrics = self.metrics_calculator.compute_all_metrics(suggestions_df, self.actor_list, self.decision_criteria_list, true_outcome_col='True Outcome')
        decision_metrics_df = self.metrics_to_dataframe(decision_metrics)

        # Apply ranking and weighted sum calculations
        ranked_decision_metrics_df, rank_dict, best_criterion = self._add_ranking_and_weighted_sum_of_normalized_scores(decision_metrics_df)

        return {
            'summary_df': summary_df,
            'decision_metrics_df': decision_metrics_df,
            'ranked_decision_metrics_df': ranked_decision_metrics_df,
            'rank_dict': rank_dict,
            'best_criterion': best_criterion
        }

    def _add_ranking_and_weighted_sum_of_normalized_scores(self, decision_metrics_df):
        """
        Rank decision metrics and compute the best criterion based on weighted sum of normalized scores.
        """
        # Normalize and apply weights to metrics for each ranking criterion
        normalized_scores = {}
        for criterion in self.ranking_criteria:
            criterion_scores = decision_metrics_df[criterion]
            normalized_scores[criterion] = criterion_scores / criterion_scores.max()  # Normalize by max score

        # Calculate weighted sum and rank criteria
        weighted_sums = {}
        for criterion in normalized_scores:
            weighted_sums[criterion] = sum(normalized_scores[criterion] * self.ranking_weights[criterion])

        ranked_df = pd.DataFrame.from_dict(weighted_sums, orient='index', columns=['Weighted Sum']).sort_values(by='Weighted Sum', ascending=False)
        best_criterion = ranked_df.index[0]
        rank_dict = ranked_df['Weighted Sum'].to_dict()

        return ranked_df, rank_dict, best_criterion
