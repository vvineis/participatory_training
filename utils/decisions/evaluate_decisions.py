import pandas as pd
import random
import numpy as np

class SummaryProcessor:
    def __init__(self, cfg, metrics_calculator, strategy, seed=None):
        """
        Initialize the SummaryProcessor with necessary parameters, external objects, and the solution strategy.
        """
        self.metrics_calculator = metrics_calculator
        self.ranking_criteria = cfg.criteria.ranking_criteria
        self.ranking_weights = dict(cfg.ranking_weights)
        self.metrics_for_evaluation = cfg.criteria.metrics_for_evaluation
        self.reward_types = cfg.actors.reward_types
        self.decision_criteria_list = cfg.decision_criteria
        self.actions_set = cfg.actions_outcomes.actions_set
        self.outcomes_set = cfg.actions_outcomes.outcomes_set
        self.strategy=strategy
        self.model_type = cfg.models.outcome.model_type 
        self.seed = seed
        if self.model_type == 'classification':
            self.mapping=cfg.actions_outcomes.mapping
        if self.seed is not None:
            random.seed(self.seed)


    def create_summary_df(self, y_val_outcome, X_val_outcome, treatment_val, decisions_df, unscaled_X_val_reward, expected_rewards_list, pred_list):
        # Unscaled feature context with true outcomes
        feature_context_df = unscaled_X_val_reward.copy()
        feature_context_df['True Outcome'] = y_val_outcome.values
        feature_context_df['Real Treatment'] = treatment_val.values if treatment_val is not None else None 
        
        # Pivot decision solutions to show best actions by decision type
        decision_solutions_summary = decisions_df.pivot(index='Row Index', columns='Decision Type', values='Best Action')
        summary_df = pd.concat([feature_context_df.reset_index(drop=True), decision_solutions_summary.reset_index(drop=True)], axis=1)

        if self.model_type=='causal_regression':
            actions_set = pred_list[0].keys()  # Assuming all elements in pred_list have the same keys
            for action in  actions_set:
                predicted_column_name = f"{action}_predicted_outcome"
                summary_df[predicted_column_name] = [
                float(clfr_pred[action][0]) if action in clfr_pred else np.nan
                for clfr_pred in pred_list
                ]
            summary_df['A_outcome'] = summary_df.apply(
                lambda row: row['True Outcome'] if row['Real Treatment'] == 'A' else row['A_predicted_outcome'], axis=1
            )

            summary_df['C_outcome'] = summary_df.apply( 
                lambda row: row['True Outcome'] if row['Real Treatment'] == 'C' else row['C_predicted_outcome'], axis=1
            )
        
            #print(summary_df[['Real Treatment', 'True Outcome', 'A_predicted_outcome', 'C_predicted_outcome', 'A_outcome', 'C_outcome']].head())


        # Initialize lists for suggested actions
        if self.model_type == 'classification':
            suggested_actions = {actor: [] for actor in self.reward_types + ['Oracle', 'Outcome_Pred_Model', 'Random']}
        elif self.model_type == 'causal_regression':
            suggested_actions = {actor: [] for actor in self.reward_types + ['Outcome_Maxim', 'Random']}

        # Compute suggested actions for each row
        for idx, (expected_rewards, predicted_outcomes) in enumerate(zip(expected_rewards_list, pred_list)):
            # Use the strategy to compute individual actions for each actor
            individual_actions = self.strategy.compute(expected_rewards, disagreement_point=None, ideal_point=None, all_actions=self.actions_set)
            for actor in self.reward_types:
                suggested_actions[actor].append(individual_actions[actor]['action'])

            # Oracle and Outcome_Pred_Model suggested actions based on true outcomes and predictions
            if self.model_type == 'classification':
                suggested_actions['Oracle'].append(self._map_outcome_to_action(y_val_outcome.iloc[idx]))
                suggested_actions['Outcome_Pred_Model'].append(self._map_outcome_to_action(pred_list[idx]))
                
            elif self.model_type == 'causal_regression':
                # Compute suggested actions for each row
                # Extract the expected outcomes for each action
                exp_outcome = {action: value[0] for action, value in predicted_outcomes.items()}
                # Determine the best action for Outcome_Pred_Model (e.g., minimize recovery time)
                best_action = self._get_best_action_given_outcome(exp_outcome, obj='max')
                #print(f'best_action {best_action}')
                # Append the single best action for this row
                suggested_actions['Outcome_Maxim'].append(best_action)

                # Add a random action for comparison
            suggested_actions['Random'].append(self._get_random_action())

        # Add suggested actions to summary DataFrame
        for actor, actions in suggested_actions.items():
            #print(f"Actor: {actor}, Suggested Actions Length: {len(actions)}, Expected Length: {len(summary_df)}")
            summary_df[f'{actor} Suggested Action'] = actions
        #print the first rows of the last ten columns of summary_df
        print(f' summ_df: {summary_df.columns}')
        
        return summary_df
    
    def _map_outcome_to_action(self, outcome):
        return self.mapping.get(outcome, self.mapping.get('default', 'Grant_lower'))
    
    def _get_random_action(self):
        """Select a random action from the actions set."""

        return random.choice(list(self.actions_set))
    
    def _get_best_action_given_outcome(self, recovery_times, obj='min'):
        if obj == 'min':
            # Find the minimum value
            min_value = min(recovery_times.values())
            # Filter actions with the minimum value
            best_actions = [action for action, value in recovery_times.items() if value == min_value]
            # Favor action 'C' in case of a tie
            return 'C' if 'C' in best_actions else best_actions[0]
        elif obj == 'max':
            # Find the maximum value
            max_value = max(recovery_times.values())
            # Filter actions with the maximum value
            best_actions = [action for action, value in recovery_times.items() if value == max_value]
            # Favor action 'C' in case of a tie
            return 'C' if 'C' in best_actions else best_actions[0]
        else:
            raise ValueError(f"Unsupported objective: {obj}. Use 'min' or 'max'.")

    def metrics_to_dataframe(self, metrics):
        return pd.DataFrame([{'Actor/Criterion': k, **v} for k, v in metrics.items()])


    def _add_ranking_and_weighted_sum_of_normalized_scores(self, metrics_df, actor_criterion_col='Actor/Criterion'):
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


    def process_decision_metrics(self, y_val_outcome, X_val_outcome, treatment_val,  decisions_df, unscaled_X_val_reward, expected_rewards_list, pred_list):

        # Create summary DataFrame
        summary_df = self.create_summary_df(y_val_outcome, X_val_outcome,  treatment_val, decisions_df, unscaled_X_val_reward, expected_rewards_list, pred_list)
        
        # Calculate decision metrics using MetricsCalculator
        decision_metrics_df = self.metrics_to_dataframe(self.metrics_calculator.compute_all_metrics(summary_df, true_outcome_col='True Outcome'))

        # Apply ranking and weighted sum calculations
        ranked_decision_metrics_df, rank_dict, best_criterion = self._add_ranking_and_weighted_sum_of_normalized_scores(decision_metrics_df)

        return {
            'summary_df': summary_df,
            'decision_metrics_df': decision_metrics_df,
            'ranked_decision_metrics_df': ranked_decision_metrics_df,
            'rank_dict': rank_dict,
            'best_criterion': best_criterion
        }
