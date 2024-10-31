import pandas as pd
from decisions.get_decisions import DecisionProcessor
from metrics.get_metrics import MetricsCalculator

class SummaryProcessor:
    def __init__(self, metrics_calculator, ranking_criteria, ranking_weights, metrics_for_evaluation,
                 reward_types, decision_criteria_list, actions_set, outcomes_set, strategy):
        """
        Initialize the SummaryProcessor with necessary parameters, external objects, and the solution strategy.
        """
        self.metrics_calculator = metrics_calculator
        self.ranking_criteria = ranking_criteria
        self.ranking_weights = ranking_weights
        self.metrics_for_evaluation = metrics_for_evaluation
        self.reward_types = reward_types
        self.decision_criteria_list = decision_criteria_list
        self.actions_set = actions_set
        self.outcomes_set = outcomes_set
        self.strategy = strategy  # Use the provided strategy for computing actions

    def create_summary_df(self, y_val_outcome, decisions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list):
        """
        Create a summary DataFrame that includes the unscaled feature context, true outcomes,
        and suggested actions by different decision criteria and each actor.

        Args:
            y_val_outcome (pd.Series): True outcomes for the validation set.
            decisions_df (pd.DataFrame): DataFrame of best actions for each decision criterion.
            unscaled_X_val_reward (pd.DataFrame): Original feature context before scaling.
            expected_rewards_list (list): List of expected rewards for each row in the validation set.
            clfr_pred_list (list): List of classifier predictions for each row.

        Returns:
            pd.DataFrame: A summary DataFrame with the unscaled features, true outcomes, and suggested actions.
        """
        # Ensure consistent lengths for inputs
        num_rows = len(unscaled_X_val_reward)
        if not (len(y_val_outcome) == num_rows == len(expected_rewards_list) == len(clfr_pred_list)):
            raise ValueError("Input lengths must match: unscaled_X_val_reward, y_val_outcome, expected_rewards_list, and clfr_pred_list.")

        # Unscaled feature context with true outcomes
        feature_context_df = unscaled_X_val_reward.copy()
        feature_context_df['True Outcome'] = y_val_outcome.values

        # Pivot decision solutions to show best actions by decision type
        decision_solutions_summary = decisions_df.pivot(index='Row Index', columns='Decision Type', values='Best Action')
        summary_df = pd.concat([feature_context_df.reset_index(drop=True), decision_solutions_summary.reset_index(drop=True)], axis=1)

        # Initialize lists for suggested actions
        suggested_actions = {actor: [] for actor in self.reward_types + ['Oracle', 'Classifier']}

        # Compute suggested actions for each row
        for idx, expected_rewards in enumerate(expected_rewards_list):
            # Use the strategy to compute individual actions for each actor
            individual_actions = self.strategy.compute(expected_rewards, disagreement_point=None, ideal_point=None, all_actions=self.actions_set)
            for actor in self.reward_types:
                suggested_actions[actor].append(individual_actions[actor]['action'])

            # Oracle and Classifier suggested actions based on true outcomes and predictions
            suggested_actions['Oracle'].append(self._map_outcome_to_action(y_val_outcome.iloc[idx]))
            suggested_actions['Classifier'].append(self._map_outcome_to_action(clfr_pred_list[idx]))

        # Add suggested actions to summary DataFrame
        for actor, actions in suggested_actions.items():
            summary_df[f'{actor} Suggested Action'] = actions

        return summary_df

    def _map_outcome_to_action(self, outcome):
        """Map outcomes to corresponding actions for oracle and classifier suggestions."""
        return {
            'Fully Repaid': 'Grant',
            'Not Repaid': 'Not Grant'
        }.get(outcome, 'Grant lower')

    def metrics_to_dataframe(self, metrics):
        """Convert a dictionary of metrics to a DataFrame."""
        return pd.DataFrame([{'Actor/Criterion': k, **v} for k, v in metrics.items()])

    def _calculate_weighted_sum_scores(self, decision_metrics_df):
        """
        Rank decision metrics and compute the best criterion based on the weighted sum of normalized scores.

        Args:
            decision_metrics_df (pd.DataFrame): DataFrame containing decision metrics for ranking.

        Returns:
            tuple: A tuple containing the ranked DataFrame, a dictionary of rank scores, and the best criterion.
        """
        normalized_scores = {criterion: decision_metrics_df[criterion] / decision_metrics_df[criterion].max() for criterion in self.ranking_criteria}
        weighted_sums = {criterion: (normalized_scores[criterion] * self.ranking_weights[criterion]).sum() for criterion in normalized_scores}
        ranked_df = pd.DataFrame.from_dict(weighted_sums, orient='index', columns=['Weighted Sum']).sort_values(by='Weighted Sum', ascending=False)

        return ranked_df, ranked_df['Weighted Sum'].to_dict(), ranked_df.index[0]

    def process_decision_metrics(self, suggestions_df, y_val_outcome, decisions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list, positive_actions_set):
        """
        Process and generate all decision metrics and return a dictionary of DataFrames and ranking results.

        Args:
            suggestions_df (pd.DataFrame): DataFrame with suggested actions by each actor.
            y_val_outcome (pd.Series): True outcomes for the validation set.
            decisions_df (pd.DataFrame): DataFrame of best actions for each decision criterion.
            unscaled_X_val_reward (pd.DataFrame): Original feature context before scaling.
            expected_rewards_list (list): List of expected rewards for each row in the validation set.
            clfr_pred_list (list): List of classifier predictions for each row.

        Returns:
            dict: A dictionary containing the summary DataFrame, decision metrics DataFrame, 
                  ranked metrics DataFrame, rank dictionary, and best criterion.
        """
        # Create summary DataFrame
        summary_df = self.create_summary_df(y_val_outcome, decisions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list)
        
        # Calculate decision metrics using MetricsCalculator
        decision_metrics = self.metrics_calculator.compute_all_metrics(suggestions_df, self.reward_types, self.decision_criteria_list, positive_actions_set, true_outcome_col='True Outcome')
        decision_metrics_df = self.metrics_to_dataframe(decision_metrics)

        # Apply ranking and weighted sum calculations
        ranked_decision_metrics_df, rank_dict, best_criterion = self._calculate_weighted_sum_scores(decision_metrics_df)

        return {
            'summary_df': summary_df,
            'decision_metrics_df': decision_metrics_df,
            'ranked_decision_metrics_df': ranked_decision_metrics_df,
            'rank_dict': rank_dict,
            'best_criterion': best_criterion
        }
