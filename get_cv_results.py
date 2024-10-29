import numpy as np
from sklearn.model_selection import ParameterGrid
from reward_models import *
from outcome_model import *

class CrossValidator:
    def __init__(self, param_grid_outcome, param_grid_reward, n_splits, 
                 process_train_val_folds, actions_set, actor_list, 
                 decision_criteria_list, ranking_criteria, ranking_weights, metrics_for_evaluation):
        self.param_grid_outcome = param_grid_outcome
        self.param_grid_reward = param_grid_reward
        self.n_splits = n_splits
        self.process_train_val_folds = process_train_val_folds
        self.actions_set = actions_set
        self.actor_list = actor_list
        self.decision_criteria_list = decision_criteria_list
        self.ranking_criteria = ranking_criteria
        self.ranking_weights = ranking_weights
        self.metrics_for_evaluation = metrics_for_evaluation

        # To store results for each fold
        self.best_hyperparams_outcome_per_fold = []
        self.best_hyperparams_reward_per_fold = []
        self.best_outcome_models_per_fold = []
        self.best_reward_models_per_fold = []
        self.fold_scores_outcome = []
        self.fold_scores_reward = []

    def tune_outcome_model(self, X_train, y_train, X_val, y_val):
        best_params, best_model, best_score = None, None, -float('inf')
        for params in ParameterGrid(self.param_grid_outcome):
            print(f"Trying parameters for outcome model: {params}")
            model = train_outcome_model(X_train, y_train, **params)
            score = evaluate_outcome_model(model, X_val, y_val)
            if score > best_score:
                best_score, best_params, best_model = score, params, model
        return best_params, best_model, best_score

    def tune_reward_models(self, X_train, y_train_bank, y_train_applicant, y_train_regulatory, 
                           X_val, y_val_bank, y_val_applicant, y_val_regulatory):
        best_params, best_models, best_score = None, None, float('inf')
        for params in ParameterGrid(self.param_grid_reward):
            print(f"Trying parameters for rewards model: {params}")
            models = train_reward_models(X_train, y_train_bank, y_train_applicant, y_train_regulatory, **params)
            mse_bank, mse_applicant, mse_regulatory = evaluate_reward_models(
                models[0], models[1], models[2], X_val, y_val_bank, y_val_applicant, y_val_regulatory
            )
            combined_mse = np.mean([mse_bank, mse_applicant, mse_regulatory])
            if combined_mse < best_score:
                best_score, best_params, best_models = combined_mse, params, models
        return best_params, best_models, best_score

    def run(self):
        # Execute cross-validation with hyperparameter tuning
        for fold, fold_dict in enumerate(self.process_train_val_folds):
            print(f"Processing fold {fold + 1}/{self.n_splits}")
            #print(fold_dict.keys())
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
            best_params_reward, best_model_reward, best_score_reward = self.tune_reward_models(
                X_train_reward, y_train_bank, y_train_applicant, y_train_regulatory,
                X_val_reward, y_val_bank, y_val_applicant, y_val_regulatory
            )
            self.best_hyperparams_reward_per_fold.append(best_params_reward)
            self.best_reward_models_per_fold.append(best_model_reward)
            self.fold_scores_reward.append(best_score_reward)

            if fold==1:
                break
            else:
                continue

        # Select the best hyperparameters across folds
        suggested_params_outcome = self.select_best_hyperparameters(self.best_hyperparams_outcome_per_fold, self.fold_scores_outcome, maximize=True)
        suggested_params_reward = self.select_best_hyperparameters(self.best_hyperparams_reward_per_fold, self.fold_scores_reward, maximize=False)

        return {
            'best_hyperparams_outcome_per_fold': self.best_hyperparams_outcome_per_fold,
            'best_outcome_models_per_fold': self.best_outcome_models_per_fold,
            'best_hyperparams_reward_per_fold': self.best_hyperparams_reward_per_fold,
            'best_reward_models_per_fold': self.best_reward_models_per_fold,
            'suggested_params_outcome': suggested_params_outcome,
            'suggested_params_reward': suggested_params_reward
        }

    @staticmethod
    def select_best_hyperparameters(hyperparams_per_fold, scores_per_fold, maximize=True):
        if maximize:
            best_index = np.argmax(scores_per_fold)  # For maximizing (e.g., accuracy)
        else:
            best_index = np.argmin(scores_per_fold)  # For minimizing (e.g., MSE)
        return hyperparams_per_fold[best_index]
