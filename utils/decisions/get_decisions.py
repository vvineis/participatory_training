
import numpy as np
import pandas as pd
from utils.decisions.compromise_functions import*
class DecisionProcessor:
    def __init__(self, outcome_model, reward_models, onehot_encoder, cfg):
        self.outcome_model = outcome_model
        self.reward_models = reward_models
        self.onehot_encoder = onehot_encoder
        self.cfg = cfg
        self.model_type = cfg.models.outcome.model_type

    def encode_features(self, feature_context):
        feature_columns = self.cfg.context.feature_columns
        categorical_columns = self.cfg.categorical_columns

        # Exclude 'Outcome' if not in the DataFrame
        available_categorical_columns = [col for col in categorical_columns if col in feature_context.columns]

        numerical_features = feature_context[feature_columns]
        categorical_features = feature_context[available_categorical_columns]

        if not categorical_features.empty:
            categorical_encoded = self.onehot_encoder.transform(categorical_features)
            encoded_feature_names = list(numerical_features.columns) + list(
                self.onehot_encoder.get_feature_names_out(available_categorical_columns)
            )
            context_encoded = np.concatenate([numerical_features.values, categorical_encoded], axis=1)
        else:
            # If no categorical features available, use only numerical features
            encoded_feature_names = list(numerical_features.columns)
            context_encoded = numerical_features.values

        return pd.DataFrame(context_encoded, columns=encoded_feature_names)

    
    def compute_expected_reward(self, feature_context):
        """
        Compute expected rewards for all actors and actions based on model type.
        """
        feature_columns = self.cfg.context.feature_columns
        feature_context = feature_context[feature_columns]

        actions_set = self.cfg.actions_outcomes.actions_set
        expected_rewards = {actor: {} for actor in self.reward_models.keys()}
        predictions_list = []

        # Classification Case
        if self.model_type == 'classification':
            outcome_probs = self.outcome_model.classifier.predict_proba(feature_context)
            outcome_classes = self.outcome_model.classifier.classes_
            predictions_list = [outcome_classes[np.argmax(prob)] for prob in outcome_probs]

            for action in actions_set:
                for idx, outcome in enumerate(outcome_classes):
                    context_with_action_outcome = feature_context.copy()
                    context_with_action_outcome['Action'] = action
                    context_with_action_outcome['Outcome'] = outcome

                    context_encoded_df = self.encode_features(context_with_action_outcome)

                    for actor, model in self.reward_models.items():
                        reward = model.predict(context_encoded_df)[0]
                        outcome_prob = outcome_probs[0][idx]

                        # Aggregate expected rewards
                        if action not in expected_rewards[actor]:
                            expected_rewards[actor][action] = 0
                        expected_rewards[actor][action] += outcome_prob * reward

        # Regression Case to complete:
        elif self.model_type == 'causal_regression':
            predicted_outcomes_A, predicted_outcomes_C = self.outcome_model.predict_outcomes(feature_context)
            for idx in range(len(feature_context)):
                # Create a dictionary to store predictions for this row
                row_predictions = {
                    'A': [predicted_outcomes_A[idx]],
                    'C': [predicted_outcomes_C[idx]]
                }
                predictions_list.append(row_predictions)
                
            for action in actions_set:
                # Select the appropriate predicted outcomes based on the action
                predicted_outcomes = (
                    predicted_outcomes_C if action == self.outcome_model.control_name else predicted_outcomes_A
                )

                for actor, model in self.reward_models.items():

                    # Add predicted outcomes as a feature for the reward model
                    feature_context_with_action = feature_context.copy()
                    feature_context_with_action['Action'] = action
                    feature_context_with_action['Outcome'] = predicted_outcomes.flatten().astype(str)

                    # Encode features (including Outcome) for reward model
                    context_encoded_df = self.encode_features(feature_context_with_action)

                    # Predict rewards for this actor and action
                    predicted_rewards = model.predict(context_encoded_df)
                    #print(f"Predicted rewards for actor {actor}, action {action}: {predicted_rewards}")

                    # Store expected rewards for this action
                    expected_rewards[actor][action] = predicted_rewards[0]

        return expected_rewards, predictions_list

    def get_decisions(self, X_val_or_test_reward):
        all_expected_rewards = []
        all_decision_solutions = []
        all_clfr_preds = []

        # Iterate through each row in the validation reward set
        for _, feature_context in X_val_or_test_reward.iterrows():
            feature_context_df = pd.DataFrame([feature_context])

            # Compute expected rewards and classifier predictions
            expected_rewards, clfr_pred = self.compute_expected_reward(feature_context_df)

            # Compute decision-making solutions based on expected rewards and disagreement points
            suggestion = SuggestAction(expected_rewards)
            decision_solutions = suggestion.compute_all_compromise_solutions()

            all_expected_rewards.append(expected_rewards)
            all_decision_solutions.append(decision_solutions)
            if clfr_pred is not None: 
                all_clfr_preds.extend(clfr_pred)
            #print(f'clfr_pred {clfr_pred}')

        return all_expected_rewards, all_decision_solutions, all_clfr_preds, self._convert_decision_solutions_to_df(all_decision_solutions)

    def _convert_decision_solutions_to_df(self, all_decision_solutions):
        rows = []
        for row_idx, decision_dict in enumerate(all_decision_solutions):
            for decision_type, solution in decision_dict.items():
                rows.append({
                    'Row Index': row_idx,
                    'Decision Type': decision_type,
                    'Best Action': solution['action'],
                    'Value': solution['value']
                })
        return pd.DataFrame(rows)

    