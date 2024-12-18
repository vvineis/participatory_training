import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from utils.models.reward_models import RewardModels
from utils.models.outcome_model import OutcomeModel
from utils.decisions.get_decisions import DecisionProcessor
from utils.decisions.evaluate_decisions import SummaryProcessor
from utils.decisions.compromise_functions import MaxIndividualReward
from utils.metrics.get_metrics import MetricsCalculator
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.utils import instantiate
from importlib import import_module
import logging

class CrossValidator:
    def __init__(self, cfg, process_train_val_folds):
        self.cfg = cfg
        self.process_train_val_folds = process_train_val_folds

        # Dynamically load the outcome model class
        self.model_class = self.get_model_class(cfg.models.outcome.model_class)
        if "classifier" in cfg.models.outcome:
            self.model_component = instantiate(cfg.models.outcome.classifier)
        elif "learner" in cfg.models.outcome:
            self.model_component = instantiate(cfg.models.outcome.learner)
        else:
            self.model_component = None
        self.param_grid_outcome = dict(cfg.models.outcome.param_grid)

        self.regressor =  instantiate(cfg.models.rewards.regressor)  
        self.param_grid_reward = dict(cfg.models.rewards.param_grid)

        self.n_splits = cfg.cv_splits
        self.feature_columns = cfg.context.feature_columns
        self.categorical_columns = cfg.categorical_columns
        self.actions_set = cfg.actions_outcomes.actions_set
        self.actor_list = cfg.actors.actor_list
        self.reward_types = cfg.actors.reward_types
        self.decision_criteria_list = cfg.decision_criteria
        self.ranking_criteria = cfg.criteria.ranking_criteria
        self.ranking_weights = cfg.ranking_weights
        self.metrics_for_evaluation = cfg.criteria.metrics_for_evaluation
        self.positive_attribute_for_fairness = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.max_individual_strategy = MaxIndividualReward()

        # Instantiate MetricsCalculator and SummaryProcessor with the strategy
        self.metrics_calculator = MetricsCalculator(cfg=cfg)

        self.summary_processor = SummaryProcessor(
            metrics_calculator=self.metrics_calculator,
            cfg=cfg, 
            strategy=self.max_individual_strategy
        )

        # Store results for each fold
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

    def get_model_class(self, class_name):
        """
        Dynamically import and return the specified class.
        :param class_name: Name of the class as a string.
        :return: Class reference.
        """
        module = import_module("utils.models.outcome_model")  
        return getattr(module, class_name)

    def tune_outcome_model(self, X_train, treatment_train, y_train, X_val, treatment_val, y_val):
        """
        Tune the outcome model dynamically (with or without treatment).
        """
        best_params, best_model, best_score = None, None, None

        # Initialize best_score based on model type
        if self.cfg.models.outcome.model_type == 'classification':
            best_score = -float('inf')  # Higher accuracy is better
        elif self.cfg.models.outcome.model_type == 'regression':
            best_score = float('inf')  # Lower MAE is better

        for params in ParameterGrid(self.param_grid_outcome):
            outcome_model_class = self.get_model_class(self.cfg.models.outcome.model_class)
            print(f'Applying tuning for {self.cfg.models.outcome.model_type} with model class {outcome_model_class }')

            # Dynamically initialize model
            if treatment_train is not None:  # For health use case (regression)
                learner = instantiate(self.cfg.models.outcome.learner)  
                print(f'Learner: {learner} should be XGBRegressor')
                outcome_model = outcome_model_class(learner=learner)
                print(f'outcome_model: {outcome_model}')
            else:  # For lending use case (classification)
                classifier = instantiate(self.cfg.models.outcome.classifier)  # Resolve classifier
                print(f'Classifier: {classifier} should be RandomForest')
                outcome_model = outcome_model_class(classifier=classifier)
                print(f'outcome_model: {outcome_model}')

            print(f"Trying parameters: {params}")

            # Train with or without treatment
            logging.getLogger("causalml").setLevel(logging.WARNING)
            if treatment_train is not None:
                outcome_model.train(X_train, treatment_train, y_train, **params)
            else:
                outcome_model.train(X_train, y_train, **params)

            # Evaluate and update best model
            if treatment_val is not None:
                score = outcome_model.evaluate(X_val, treatment_val, y_val)
            else:
                score = outcome_model.evaluate(X_val, y_val)
            
            print(f'Current best score:{best_score}, current score {score}')

            # Update best score and model based on model type
            if (self.cfg.models.outcome.model_type == 'classification' and score > best_score) or \
            (self.cfg.models.outcome.model_type == 'regression' and score < best_score):
                best_score, best_params, best_model = score, params, outcome_model
        print(f'Finale best model: {best_model}')

        return best_params, best_model, best_score

    def tune_reward_models(self, X_train, y_train_rewards, X_val, y_val_rewards):
        best_params, best_models, best_score = None, {}, float('inf')

        # Iterate through all parameter combinations in the grid
        for params in ParameterGrid(self.param_grid_reward):
            print(f"Trying parameters for reward model: {params}")
            
            # Initialize reward models dynamically
            self.reward_model = RewardModels(self.regressor, self.reward_types, **params)
            
            # Train reward models
            trained_models = self.reward_model.train(X_train, y_train_rewards)
            
            # Evaluate reward models
            scores = self.reward_model.evaluate(X_val, y_val_rewards)
            
            # Calculate average MSE across all reward types
            avg_mse = sum(scores[f"{reward_type}_mse"] for reward_type in self.reward_types) / len(self.reward_types)
            
            # Update the best parameters, models, and score if the current average MSE is better
            if avg_mse < best_score:
                best_score = avg_mse
                best_params = params
                best_models = trained_models

        return best_params, best_models, best_score


    '''def tune_reward_models(self, X_train, y_train_bank, y_train_applicant, y_train_regulatory, 
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

        return best_params, best_models, best_score'''
    
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
            X_train_outcome, treatment_train, y_train_outcome = fold_dict['train_outcome']
            #print(f'X_train_outcome {X_train_outcome.columns}')
            #print(f'treatment_train {treatment_train}')
            #print(f'y_train_outcome {y_train_outcome}')
            X_val_outcome, treatment_val, y_val_outcome = fold_dict['val_or_test_outcome']

            X_train_reward, y_train_rewards = fold_dict['train_reward']
            X_val_reward, y_val_rewards = fold_dict['val_or_test_reward']


            # Tune outcome model
            best_params_outcome, best_model_outcome, best_score_outcome = self.tune_outcome_model(
                X_train_outcome, treatment_train, y_train_outcome, 
                X_val_outcome,treatment_val, y_val_outcome
            )

            self.best_hyperparams_outcome_per_fold.append(best_params_outcome)
            self.best_outcome_models_per_fold.append(best_model_outcome)
            self.fold_scores_outcome.append(best_score_outcome)

            # Tune reward models
            best_params_reward, best_models_reward, best_score_reward = self.tune_reward_models(
                X_train_reward, y_train_rewards, X_val_reward, y_val_rewards
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
                cfg=self.cfg
            )
            
            # Get decisions from the decision processor
            all_expected_rewards, all_decisions, all_predictions, decisions_df = decision_processor.get_decisions(X_val_reward)

            # Summarize and rank decision metrics for the current fold
            result = self.summary_processor.process_decision_metrics(
                y_val_outcome=y_val_outcome,
                X_val_outcome=X_val_outcome,
                decisions_df=decisions_df,
                unscaled_X_val_reward=fold_dict['unscaled_val_or_test_set'],
                expected_rewards_list=all_expected_rewards,
                pred_list=all_predictions
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
            self.metrics_calculator.compute_all_metrics(CV_summary_df,true_outcome_col='True Outcome')
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
