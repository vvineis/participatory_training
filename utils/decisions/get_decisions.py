from config import (feature_columns)
import numpy as np
import pandas as pd
from utils.decisions.compromise_functions import*
from utils.metrics.get_metrics import MetricsCalculator
from config import (fairness_metrics_list, standard_metrics_list, case_metrics_list, positive_actions_set, outcomes_set)

class DecisionProcessor:
    def __init__(self, outcome_model, reward_models, onehot_encoder, actions_set, feature_columns, categorical_columns, 
                 actor_list, decision_criteria_list, ranking_criteria, ranking_weights, metrics_for_evaluation):
        self.outcome_model = outcome_model
        self.reward_models = reward_models
        self.onehot_encoder = onehot_encoder
        self.actions_set = actions_set
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.actor_list = actor_list
        self.decision_criteria_list = decision_criteria_list
        self.ranking_criteria = ranking_criteria
        self.ranking_weights = ranking_weights
        self.metrics_for_evaluation = metrics_for_evaluation

    def compute_expected_reward(self, feature_context):
        feature_context = feature_context[self.feature_columns]

        # Predict probabilities for each possible outcome
        outcome_probs = self.outcome_model.predict_proba(feature_context)
        outcome_classes = self.outcome_model.classes_

        expected_rewards = {'Bank': {}, 'Applicant': {}, 'Regulatory': {}}
        predicted_class_list = [outcome_classes[np.argmax(prob)] for prob in outcome_probs]

        # Calculate expected rewards for each action
        for action in self.actions_set:
            reward_sum_bank, reward_sum_applicant, reward_sum_regulatory = 0, 0, 0

            for idx, outcome in enumerate(outcome_classes):
                context_with_action_outcome = feature_context.copy()
                context_with_action_outcome['Action'] = action
                context_with_action_outcome['Outcome'] = outcome

                # Encode features
                numerical_features = context_with_action_outcome[self.feature_columns]
                categorical_features = context_with_action_outcome[self.categorical_columns]
                categorical_encoded = self.onehot_encoder.transform(categorical_features)
                context_encoded = np.concatenate([numerical_features.values, categorical_encoded], axis=1)

                encoded_feature_names = list(numerical_features.columns) + list(self.onehot_encoder.get_feature_names_out(self.categorical_columns))
            
                # Convert to DataFrame with column names
                context_encoded_df = pd.DataFrame(context_encoded, columns=encoded_feature_names)

                # Predict rewards
                bank_reward = self.reward_models[0].predict(context_encoded_df)[0]
                applicant_reward = self.reward_models[1].predict(context_encoded_df)[0]
                regulatory_reward = self.reward_models[2].predict(context_encoded_df)[0]


                outcome_prob = outcome_probs[0][idx]
                reward_sum_bank += outcome_prob * bank_reward
                reward_sum_applicant += outcome_prob * applicant_reward
                reward_sum_regulatory += outcome_prob * regulatory_reward

            expected_rewards['Bank'][action] = reward_sum_bank
            expected_rewards['Applicant'][action] = reward_sum_applicant
            expected_rewards['Regulatory'][action] = reward_sum_regulatory

        return expected_rewards, predicted_class_list
    

    def get_decisions(self, X_val_or_test_reward):
        self.all_expected_rewards = []
        self.all_decision_solutions = []
        self.all_clfr_preds = []

        # Iterate through each row in the validation reward set
        #index=0
        for _, feature_context in X_val_or_test_reward.iterrows():
            feature_context_df = pd.DataFrame([feature_context])

            # Compute expected rewards and classifier predictions
            expected_rewards, clfr_pred = self.compute_expected_reward(feature_context_df)

            # Compute decision-making solutions based on expected rewards and disagreement points
            suggestion = SuggestAction(expected_rewards)
            decision_solutions = suggestion.compute_all_compromise_solutions()


            self.all_expected_rewards.append(expected_rewards)
            self.all_decision_solutions.append(decision_solutions)
            self.all_clfr_preds.extend(clfr_pred)

            #index+=1
            #if index==2:
             #   break
            #else:
             #   continue
        return  self.all_expected_rewards, self.all_decision_solutions, self.all_clfr_preds, self._convert_decision_solutions_to_df()
        
    def _convert_decision_solutions_to_df(self):
        rows = []
        for row_idx, decision_dict in enumerate(self.all_decision_solutions):
            for decision_type, solution in decision_dict.items():
                row = {
                    'Row Index': row_idx,
                    'Decision Type': decision_type,
                    'Best Action': solution['action'],
                    'Value': solution['value']
                }
                rows.append(row)
        return pd.DataFrame(rows)
    