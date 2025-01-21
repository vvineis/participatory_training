"""
This module contains the functions to train final models on the entire training set and evaluate on the test set.
"""

from utils.models.outcome_model import OutcomeModel
from utils.models.reward_models import RewardModels
from utils.decisions.get_decisions import DecisionProcessor
from utils.decisions.evaluate_decisions import SummaryProcessor
from utils.decisions.compromise_functions import MaxIndividualReward
from utils.metrics.get_metrics import MetricsCalculator
from hydra.utils import instantiate
from importlib import import_module

def get_model_class(class_name):
    """
    Dynamically import and return the specified class.
    :param class_name: Name of the class as a string.
    :return: Class reference.
    """
    module = import_module("utils.models.outcome_model")  
    return getattr(module, class_name)

def run_final_evaluation(data_processor, cv_results, all_train_set, test_set, cfg):
    """
    Train final models on the entire training set and evaluate on the test set.
    :param data_processor: DataProcessor object.
    :param cv_results: Cross-validation results.
    :param all_train_set: Entire training set.
    :param test_set: Test set.
    :param cfg: Config object.
    :return: Final evaluation results.
    """
    # Prepare training and test sets
    reward_types= cfg.actors.reward_types
    final_training_data = data_processor.prepare_for_training(all_train_set, test_set)

    X_train_outcome, treatment_train, y_train_outcome = final_training_data['train_outcome']
    X_test_outcome, treatment_test, mu_test, y_test_outcome = final_training_data['val_or_test_outcome']

    X_train_reward, y_train_rewards = final_training_data['train_reward']
    X_test_reward, y_test_rewards = final_training_data['val_or_test_reward']

    # Suggested hyperparameters from cross-validation
    suggested_params_outcome = cv_results['suggested_params_outcome']
    suggested_params_reward = cv_results['suggested_params_reward']


    # Train final outcome model
    # Dynamically initialize and train the outcome model
    model_class = get_model_class(cfg.models.outcome.model_class)
    if "classifier" in cfg.models.outcome:
        model_component = instantiate(cfg.models.outcome.classifier)
        outcome_model = model_class(classifier=model_component)
        outcome_model.train(X_train_outcome, y_train_outcome, **suggested_params_outcome)
        final_outcome_score = outcome_model.evaluate(X_test_outcome, y_test_outcome)
        print(f"Final Outcome Model Accuracy: {final_outcome_score:.4f}")

    elif "learner" in cfg.models.outcome:
        model_component = instantiate(cfg.models.outcome.learner)
        outcome_model = model_class(learner=model_component)
        outcome_model.train(X_train_outcome, treatment_train, y_train_outcome, **suggested_params_outcome)
        final_outcome_score = outcome_model.evaluate(X_test_outcome, treatment_test, y_test_outcome)
        print(f"Final Outcome Model MAE: {final_outcome_score:.4f}")


    final_training_data['unscaled_val_or_test_set'].columns
    # Train final reward models
    reward_model = RewardModels(instantiate(cfg.models.rewards.regressor), reward_types, **suggested_params_reward)

    final_reward_models = reward_model.train(
        X_train_reward, y_train_rewards)

    # Evaluate the models on the test set
    mse_results = reward_model.evaluate(
        X_test_reward, y_test_rewards)
    
    mse_values = {actor: mse_results.get(f"{actor}_mse", "MSE not found") for actor in reward_types}

    # Print evaluation metrics
    print("Final Reward Models MSE:")
    for actor, mse in mse_values.items():
        print(f"{actor}: {mse}")

    
    # Process test set using final models
    decision_processor = DecisionProcessor(
        outcome_model=outcome_model,
        reward_models=final_reward_models,
        onehot_encoder=final_training_data['onehot_encoder'],
        cfg=cfg
    )
    
    # Get decisions for the test set
    all_expected_rewards, all_decisions, all_predictions, decisions_df = decision_processor.get_decisions(X_test_reward)

    # Evaluate final decisions on the test set
    max_individual_strategy = MaxIndividualReward()
    metrics_calculator = MetricsCalculator(cfg=cfg)
    summary_processor = SummaryProcessor(
        metrics_calculator=metrics_calculator,
        cfg=cfg,
        strategy=max_individual_strategy
    )

    # Process metrics on test set
    test_results_dict = summary_processor.process_decision_metrics(
        y_val_outcome=y_test_outcome,
        X_val_outcome=X_test_outcome,
        treatment_val=treatment_test,
        decisions_df=decisions_df,
        unscaled_X_val_reward=final_training_data['unscaled_val_or_test_set'],
        expected_rewards_list=all_expected_rewards,
        pred_list=all_predictions,
    )

    print("Final evaluation on test set completed.")
    return test_results_dict, suggested_params_outcome, suggested_params_reward, final_outcome_score
