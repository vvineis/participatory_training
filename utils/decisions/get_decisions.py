
import numpy as np
import pandas as pd
from utils.decisions.compromise_functions import*
class DecisionProcessor:
    def __init__(self, outcome_model, reward_models, onehot_encoder, cfg):
        self.outcome_model = outcome_model
        self.reward_models = reward_models
        self.onehot_encoder = onehot_encoder
        self.cfg = cfg

    def encode_features(self, feature_context):
        """
        Encode numerical and categorical features.
        """
        feature_columns = self.cfg.context.feature_columns
        categorical_columns = self.cfg.categorical_columns

        numerical_features = feature_context[feature_columns]
        categorical_features = feature_context[categorical_columns]
        categorical_encoded = self.onehot_encoder.transform(categorical_features)
        
        # Combine encoded categorical and numerical features
        encoded_feature_names = list(numerical_features.columns) + list(
            self.onehot_encoder.get_feature_names_out(categorical_columns)
        )
        context_encoded = np.concatenate([numerical_features.values, categorical_encoded], axis=1)
        
        return pd.DataFrame(context_encoded, columns=encoded_feature_names)

    def compute_expected_reward(self, feature_context):
        """
        Compute expected rewards for all actors and actions.
        """
        feature_columns = self.cfg.context.feature_columns
        feature_context = feature_context[feature_columns]

        # Predict probabilities for each possible outcome
        outcome_probs = self.outcome_model.predict_proba(feature_context)
        outcome_classes = self.outcome_model.classes_
        predicted_class_list = [outcome_classes[np.argmax(prob)] for prob in outcome_probs]

        expected_rewards = {actor: {} for actor in self.reward_models.keys()}
        actions_set = self.cfg.actions_outcomes.actions_set

        for action in actions_set:
            for idx, outcome in enumerate(outcome_classes):
                context_with_action_outcome = feature_context.copy()
                context_with_action_outcome['Action'] = action
                context_with_action_outcome['Outcome'] = outcome
                
                # Encode features
                context_encoded_df = self.encode_features(context_with_action_outcome)

                # Compute rewards for each actor
                for actor, model in self.reward_models.items():
                    reward = model.predict(context_encoded_df)[0]
                    outcome_prob = outcome_probs[0][idx]

                    # Aggregate expected rewards
                    if action not in expected_rewards[actor]:
                        expected_rewards[actor][action] = 0
                    expected_rewards[actor][action] += outcome_prob * reward

        return expected_rewards, predicted_class_list

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
            all_clfr_preds.extend(clfr_pred)

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

    