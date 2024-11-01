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
 
        # Ensure consistent lengths for inputs
        num_rows = len(unscaled_X_val_reward)
        print (len(y_val_outcome), num_rows, len(expected_rewards_list),  len(clfr_pred_list))
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

        summary_df.keys()

        return summary_df

    def _map_outcome_to_action(self, outcome):
        """Map outcomes to corresponding actions for oracle and classifier suggestions."""
        return {
            'Fully Repaid': 'Grant',
            'Not Repaid': 'Not Grant'
        }.get(outcome, 'Grant lower')

    def metrics_to_dataframe(self, metrics):
        return pd.DataFrame([{'Actor/Criterion': k, **v} for k, v in metrics.items()])


    def _add_ranking_and_weighted_sum_of_normalized_scores(self, metrics_df, actor_criterion_col='Actor/Criterion'):
        """
        Add normalized columns, rank actors/criteria based on normalized metrics, and compute the weighted sum of normalized scores.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing metrics for each actor or criterion.
            ranking_criteria (dict): Mapping metric column names to ranking strategies ('max', 'min', 'zero').
            ranking_weights (dict): Mapping metric column names to their respective weights.
            metrics_for_evaluation (list): List of metrics to evaluate and compute the weighted sum.
            actor_criterion_col (str): The name of the column to be used as the key in the output dictionary.

        Returns:
            ranked_df (pd.DataFrame): DataFrame with normalized columns, ranking columns, and weighted sum of normalized scores.
            weighted_sum_dict (dict): Dictionary with Actor/Criterion as the key and the weighted sum as the value.
            best_actor_criterion (str): The actor/criterion with the highest weighted sum of normalized scores.
        """
        normalized_df = metrics_df.copy()

        # Verify ranking weights are correctly formatted
        if not isinstance(self.ranking_weights, dict):
            raise ValueError("ranking_weights must be a dictionary.")

        # Normalize weights
        total_weight = sum(self.ranking_weights.get(metric, 0) for metric in self.metrics_for_evaluation)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero.")

        normalized_weights = {metric: self.ranking_weights[metric] / total_weight for metric in self.metrics_for_evaluation}

        normalized_columns = []
        ranking_columns = []

        # Normalize the columns based on the criteria provided (min, max, zero)
        for column in self.metrics_for_evaluation:
            if column == "Accuracy":
                normalized_df[f'{column} Normalized'] = normalized_df[column]
            elif self.ranking_criteria[column] == 'max':
                min_value, max_value = normalized_df[column].min(), normalized_df[column].max()
                normalized_df[f'{column} Normalized'] = (normalized_df[column] - min_value) / max((max_value - min_value), 1e-9)
            elif self.ranking_criteria[column] == 'min':
                min_value, max_value = normalized_df[column].min(), normalized_df[column].max()
                normalized_df[f'{column} Normalized'] = (max_value - normalized_df[column]) / max((max_value - min_value), 1e-9)
            elif self.ranking_criteria[column] == 'zero':
                max_abs_value = normalized_df[column].abs().max()
                normalized_df[f'{column} Normalized'] = 1 - (normalized_df[column].abs() / max(max_abs_value, 1e-9))

            normalized_columns.append(f'{column} Normalized')

        # Compute rankings based on normalized columns and the specified criteria
        for column in self.metrics_for_evaluation:
            norm_column = f'{column} Normalized'
            rank_column = f'{column} Rank'

            # Rank higher normalized values better for all metrics (as they are normalized)
            normalized_df[rank_column] = normalized_df[norm_column].rank(ascending=False, method='min')
            ranking_columns.append(rank_column)

        # Compute weighted sum of normalized scores for each actor/criterion
        normalized_df['Weighted Normalized-Sum'] = sum(
            normalized_df[f'{metric} Normalized'] * normalized_weights.get(metric, 0) for metric in self.metrics_for_evaluation
        )

        # Create a dictionary with Actor/Criterion as the key and the weighted normalized sum as the value
        weighted_sum_dict = normalized_df.set_index(actor_criterion_col)['Weighted Normalized-Sum'].to_dict()

        # Find the actor/criterion with the highest weighted sum of normalized scores
        best_actor_criterion = max(weighted_sum_dict, key=weighted_sum_dict.get)

        return normalized_df.sort_values(by='Weighted Normalized-Sum', ascending=False).reset_index(drop=True), weighted_sum_dict, best_actor_criterion


    def process_decision_metrics(self,actor_list,  y_val_outcome, decisions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list, positive_attribute_for_fairness):

        # Create summary DataFrame
        summary_df = self.create_summary_df(y_val_outcome, decisions_df, unscaled_X_val_reward, expected_rewards_list, clfr_pred_list)
        print (f"summary_df_columns {summary_df.columns}")
        
        # Calculate decision metrics using MetricsCalculator
        decision_metrics_df = self.metrics_to_dataframe(self.metrics_calculator.compute_all_metrics(summary_df, actor_list, self.decision_criteria_list, positive_attribute_for_fairness, true_outcome_col='True Outcome'))
        print (f"decision_metrics_df_head {decision_metrics_df.head()}")

        # Apply ranking and weighted sum calculations
        ranked_decision_metrics_df, rank_dict, best_criterion = self._add_ranking_and_weighted_sum_of_normalized_scores(decision_metrics_df)

        return {
            'summary_df': summary_df,
            'decision_metrics_df': decision_metrics_df,
            'ranked_decision_metrics_df': ranked_decision_metrics_df,
            'rank_dict': rank_dict,
            'best_criterion': best_criterion
        }
