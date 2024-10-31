import numpy as np
from sklearn.model_selection import ParameterGrid
from models.reward_models import RewardModels
from models.outcome_model import OutcomeModel
from decisions.get_decisions import DecisionProcessor
from decisions.evaluate_decisions import SummaryProcessor
from decisions.compromise_functions import MaxIndividualReward
from metrics.get_metrics import MetricsCalculator


from config import fairness_metrics_list, standard_metrics_list, case_metrics_list, positive_actions_set, actions_set,  outcomes_set

class CrossValidator:
    def __init__(self,classifier, regressor, param_grid_outcome, param_grid_reward, n_splits, 
                 process_train_val_folds, feature_columns, categorical_columns, actions_set, actor_list, 
                 decision_criteria_list, ranking_criteria, ranking_weights, metrics_for_evaluation):
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
            best_params_reward, best_models_reward, best_score_reward = self.tune_reward_models(
                X_train_reward, y_train_bank, y_train_applicant, y_train_regulatory,
                X_val_reward, y_val_bank, y_val_applicant, y_val_regulatory
            )
            self.best_hyperparams_reward_per_fold.append(best_params_reward)
            self.best_reward_models_per_fold.append(best_models_reward)
            self.fold_scores_reward.append(best_score_reward)

            # Step 3: Process the validation set using the best outcome and reward models for the fold
            print("Processing validation set...")
            decision_processor = DecisionProcessor(
                    outcome_model=best_model_outcome,
                    reward_models=best_models_reward,
                    onehot_encoder= fold_dict['onehot_encoder'],
                    actions_set=self.actions_set,
                    feature_columns=self.feature_columns,
                    categorical_columns=self.categorical_columns,
                    actor_list=self.actor_list,
                    decision_criteria_list=self.decision_criteria_list,
                    ranking_criteria=self.ranking_criteria,
                    ranking_weights=self.ranking_weights,
                    metrics_for_evaluation=self.metrics_for_evaluation
                )
            
            decisions_dfs = decision_processor.get_decisions_dfs(X_val_reward)
            print(decisions_dfs)

            max_individual_strategy = MaxIndividualReward()

            # Instantiate MetricsCalculator and SummaryProcessor with the strategy
            metrics_calculator = MetricsCalculator(fairness_metrics_list, standard_metrics_list, case_metrics_list, actions_set, outcomes_set)
            summary_processor = SummaryProcessor(
                metrics_calculator=metrics_calculator,
                ranking_criteria=self.ranking_criteria,
                ranking_weights=self.ranking_weights,
                metrics_for_evaluation=self.metrics_for_evaluation,
                actor_list=self.actor_list,
                decision_criteria_list=self.decision_criteria_list,
                actions_set=actions_set,
                outcomes_set=outcomes_set,
                strategy=max_individual_strategy  
            )

            # Process and retrieve all decision metrics and summary data
            result = summary_processor.process_decision_metrics(
                suggestions_df=suggestions_df,
                X_val_reward=X_val_reward,
                y_val_outcome=y_val_outcome,
                decision_solutions_df=decision_solutions_df,
                unscaled_X_val_reward=   unscaled_X_val_reward,
                expected_rewards_list=expected_rewards_list,
                clfr_pred_list=clfr_pred_list
            )

            # Access the result components
            print(result['summary_df'])
            print(result['decision_metrics_df'])
            print(result['ranked_decision_metrics_df'])
            print(result['rank_dict'])
            print(result['best_criterion'])



            if fold==0:
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
