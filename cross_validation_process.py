import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from models.reward_models import RewardModels
from models.outcome_model import OutcomeModel
from decisions.get_decisions import DecisionProcessor
from decisions.evaluate_decisions import SummaryProcessor
from decisions.compromise_functions import MaxIndividualReward
from metrics.get_metrics import MetricsCalculator


from config import fairness_metrics_list, standard_metrics_list, case_metrics_list, positive_actions_set, actions_set,  outcomes_set, positive_attribute_for_fairness, actor_list

class CrossValidator:
    def __init__(self,classifier, regressor, param_grid_outcome, param_grid_reward, n_splits, 
                 process_train_val_folds, feature_columns, categorical_columns, actions_set, actor_list, reward_types,
                 decision_criteria_list, ranking_criteria, ranking_weights, metrics_for_evaluation, positive_attribute_for_fairness):
        self.classifier = classifier
        self.regressor=regressor
        self.param_grid_outcome = param_grid_outcome
        self.param_grid_reward = param_grid_reward
        self.n_splits = n_splits
        self.process_train_val_folds = process_train_val_folds
        self.feature_columns= feature_columns
        self.categorical_columns= categorical_columns
        self.actions_set = actions_set
        self.actor_list = actor_list
        self.reward_types= reward_types
        self.decision_criteria_list = decision_criteria_list
        self.ranking_criteria = ranking_criteria
        self.ranking_weights = ranking_weights
        self.metrics_for_evaluation = metrics_for_evaluation
        self.positive_attribute_for_fairness=positive_attribute_for_fairness
        self.max_individual_strategy = MaxIndividualReward()

        # Instantiate MetricsCalculator and SummaryProcessor with the strategy
        self.metrics_calculator = MetricsCalculator(fairness_metrics_list, standard_metrics_list, case_metrics_list, actions_set, outcomes_set, positive_actions_set)
        self.summary_processor = SummaryProcessor(
                metrics_calculator=self.metrics_calculator,
                ranking_criteria=self.ranking_criteria,
                ranking_weights=self.ranking_weights,
                metrics_for_evaluation=self.metrics_for_evaluation,
                reward_types=self.reward_types,
                decision_criteria_list=self.decision_criteria_list,
                actions_set=actions_set,
                outcomes_set=outcomes_set,
                strategy=self.max_individual_strategy  
            )

        # To store results for each fold
        self.best_hyperparams_outcome_per_fold = []
        self.best_hyperparams_reward_per_fold = []
        self.best_outcome_models_per_fold = []
        self.best_reward_models_per_fold = []
        self.fold_scores_outcome = []
        self.fold_scores_reward = []

    def print_best_params_per_fold(self):
        """
        Print the best hyperparameters and models per fold for outcome and reward models.
        """
        print("Best Hyperparameters and Models per Fold:")
        for i, (params_outcome, params_reward, outcome_model, reward_models) in enumerate(zip(
            self.best_hyperparams_outcome_per_fold,
            self.best_hyperparams_reward_per_fold,
            self.best_outcome_models_per_fold,
            self.best_reward_models_per_fold
        )):
            print(f"Fold {i+1}: Best Outcome Hyperparameters: {params_outcome}, Best Reward Hyperparameters: {params_reward}")
            print(f"Outcome Model: {outcome_model}, Reward Models: {reward_models}")

        # Suggested hyperparameters for future models
        print(f"Suggested Outcome Hyperparameters: {self.cv_results['suggested_params_outcome']}")
        print(f"Suggested Reward Hyperparameters: {self.cv_results['suggested_params_reward']}")

    def check_summary_lengths(self):
        """
        Check the length of each summary DataFrame in `all_fold_summaries` to ensure consistency.
        """
        for i, summary_df in enumerate(self.cv_results.get('all_fold_summaries', [])):
            print(f"Fold {i+1} summary DataFrame length: {summary_df.shape[0]}")


    def tune_outcome_model(self, X_train, y_train, X_val, y_val):
        best_params, best_model, best_score = None, None, -float('inf')
        for params in ParameterGrid(self.param_grid_outcome):
            self.outcome_model = OutcomeModel(self.classifier)
            print(f"Trying parameters for outcome model: {params}")
            model = self.outcome_model.train(X_train, y_train, **params)
            score = self.outcome_model.evaluate(X_val, y_val)
            if score > best_score:
                best_score, best_params, best_model = score, params, model
        return best_params, best_model, best_score

    def tune_reward_models(self, X_train, y_train_bank, y_train_applicant, y_train_regulatory, 
                           X_val, y_val_bank, y_val_applicant, y_val_regulatory):
        best_params, best_models, best_score = None, None, float('inf')
        for params in ParameterGrid(self.param_grid_reward):
            print(f"Trying parameters for reward model: {params}")
            
            # Train reward models using RewardModel's train method
            self.reward_model = RewardModels(self.regressor, **params)
            # Train reward models using RewardModel's train method
            bank_model, applicant_model, regulatory_model = self.reward_model.train(
                X_train, y_train_bank, y_train_applicant, y_train_regulatory
            )
            
            # Evaluate reward models using RewardModel's evaluate method
            scores = self.reward_model.evaluate(X_val, y_val_bank, y_val_applicant, y_val_regulatory)
            avg_mse = (scores['bank_mse'] + scores['applicant_mse'] + scores['regulatory_mse']) / 3
            
            if avg_mse < best_score:
                best_score, best_params, best_models = avg_mse, params, (bank_model, applicant_model, regulatory_model)

        return best_params, best_models, best_score
    
    def run(self):
        # List to store summary data for aggregation across folds
        all_fold_summaries = []
        all_fold_decision_metrics = []
        all_fold_ranked_decision_metrics = []
        all_fold_rank_dicts = []
        all_fold_best_criteria = []

        # Execute cross-validation with hyperparameter tuning
        for fold, fold_dict in enumerate(self.process_train_val_folds):
            print(f"Processing fold {fold + 1}/{self.n_splits}")
            
            # Unpack train and validation sets for the fold
            X_train_outcome, y_train_outcome = fold_dict['train_outcome']
            X_val_outcome, y_val_outcome = fold_dict['val_or_test_outcome']
            X_train_reward, y_train_bank, y_train_applicant, y_train_regulatory = fold_dict['train_reward']
            X_val_reward, y_val_bank, y_val_applicant, y_val_regulatory = fold_dict['val_or_test_reward']

            # Tune outcome model
            best_params_outcome, best_model_outcome, best_score_outcome = self.tune_outcome_model(
                X_train_outcome, y_train_outcome, X_val_outcome, y_val_outcome
            )
            self.best_hyperparams_outcome_per_fold.append(best_params_outcome)
            self.best_outcome_models_per_fold.append(best_model_outcome)
            self.fold_scores_outcome.append(best_score_outcome)

            # Tune reward models
            best_params_reward, best_models_reward, best_score_reward = self.tune_reward_models(
                X_train_reward, y_train_bank, y_train_applicant, y_train_regulatory,
                X_val_reward, y_val_bank, y_val_applicant, y_val_regulatory
            )
            self.best_hyperparams_reward_per_fold.append(best_params_reward)
            self.best_reward_models_per_fold.append(best_models_reward)
            self.fold_scores_reward.append(best_score_reward)

            # Process validation set with the best models
            print("Processing validation set...")
            decision_processor = DecisionProcessor(
                outcome_model=best_model_outcome,
                reward_models=best_models_reward,
                onehot_encoder=fold_dict['onehot_encoder'],
                actions_set=self.actions_set,
                feature_columns=self.feature_columns,
                categorical_columns=self.categorical_columns,
                actor_list=self.actor_list,
                decision_criteria_list=self.decision_criteria_list,
                ranking_criteria=self.ranking_criteria,
                ranking_weights=self.ranking_weights,
                metrics_for_evaluation=self.metrics_for_evaluation
            )
            
            # Get decisions from the decision processor
            all_expected_rewards, all_decisions, all_clsf_pred, decisions_df = decision_processor.get_decisions(X_val_reward)

            # Summarize and rank decision metrics for the current fold
            result = self.summary_processor.process_decision_metrics(
                actor_list=self.actor_list,
                y_val_outcome=y_val_outcome,
                decisions_df=decisions_df,
                unscaled_X_val_reward=fold_dict['unscaled_val_or_test_set'],
                expected_rewards_list=all_expected_rewards,
                clfr_pred_list=all_clsf_pred,
                positive_attribute_for_fairness=self.positive_attribute_for_fairness
            )

            # Store fold results for later aggregation
            all_fold_summaries.append(result['summary_df'])
            all_fold_decision_metrics.append(result['decision_metrics_df'])
            all_fold_ranked_decision_metrics.append(result['ranked_decision_metrics_df'])
            all_fold_rank_dicts.append(result['rank_dict'])
            all_fold_best_criteria.append(result['best_criterion'])

        # Select best hyperparameters across folds
        suggested_params_outcome = self.select_best_hyperparameters(self.best_hyperparams_outcome_per_fold, self.fold_scores_outcome, maximize=True)
        suggested_params_reward = self.select_best_hyperparameters(self.best_hyperparams_reward_per_fold, self.fold_scores_reward, maximize=False)

        # Store results for later use and aggregation
        self.cv_results = {
            'best_hyperparams_outcome_per_fold': self.best_hyperparams_outcome_per_fold,
            'best_outcome_models_per_fold': self.best_outcome_models_per_fold,
            'best_hyperparams_reward_per_fold': self.best_hyperparams_reward_per_fold,
            'best_reward_models_per_fold': self.best_reward_models_per_fold,
            'suggested_params_outcome': suggested_params_outcome,
            'suggested_params_reward': suggested_params_reward,
            'all_fold_summaries': all_fold_summaries,
            'all_fold_decision_metrics': all_fold_decision_metrics,
            'all_fold_ranked_decision_metrics': all_fold_ranked_decision_metrics,
            'all_fold_rank_dicts': all_fold_rank_dicts,
            'all_fold_best_criteria': all_fold_best_criteria
        }

    @staticmethod
    def select_best_hyperparameters(hyperparams_per_fold, scores_per_fold, maximize=True):
        if maximize:
            best_index = np.argmax(scores_per_fold)  # For maximizing (e.g., accuracy)
        else:
            best_index = np.argmin(scores_per_fold)  # For minimizing (e.g., MSE)
        return hyperparams_per_fold[best_index]
    
    def aggregate_cv_results(self):
        """
        Aggregate the cross-validation results across all folds and process the summary.
        """
        fold_decision_summaries= self.cv_results['all_fold_summaries']
        # Concatenate fold summaries for aggregation
        CV_summary_df = pd.concat(fold_decision_summaries, ignore_index=True)
        print(f"len CV_summary {CV_summary_df.shape[0]}")
        
        # Compute overall decision metrics
        CV_decision_metrics_df = self.summary_processor.metrics_to_dataframe(
            self.metrics_calculator.compute_all_metrics(CV_summary_df, self.actor_list, self.decision_criteria_list, self.positive_attribute_for_fairness, true_outcome_col='True Outcome')
        )
        
        # Rank and compute weighted sum for overall performance
        CV_ranked_decision_metrics_df, CV_rank_dict, CV_best_criterion = self.summary_processor._add_ranking_and_weighted_sum_of_normalized_scores(CV_decision_metrics_df)
        
        # Summarize results in a dictionary
        CV_results_dict = {
            'summary_df': CV_summary_df,
            'decision_metrics_df': CV_decision_metrics_df,
            'ranked_decision_metrics_df': CV_ranked_decision_metrics_df,
            'rank_dict': CV_rank_dict,
            'best_criterion': CV_best_criterion,
            'suggested_params_outcome': self.cv_results['suggested_params_outcome'],
            'suggested_params_reward': self.cv_results['suggested_params_reward']
                    }
        
        print("Cross-validation results aggregation completed.")
        return CV_results_dict
