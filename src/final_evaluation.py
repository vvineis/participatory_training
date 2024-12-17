from utils.models.outcome_model import OutcomeModel
from utils.models.reward_models import RewardModels
from utils.decisions.get_decisions import DecisionProcessor
from utils.decisions.evaluate_decisions import SummaryProcessor
from utils.decisions.compromise_functions import MaxIndividualReward
from utils.metrics.get_metrics import MetricsCalculator
from hydra.utils import instantiate

def run_final_evaluation(data_processor, cv_results, all_train_set, test_set, cfg):
    # Prepare training and test sets
    reward_types= cfg.actors.reward_types
    final_training_data = data_processor.prepare_for_training(all_train_set, test_set)

    X_train_outcome, y_train_outcome = final_training_data['train_outcome']
    X_test_outcome, y_test_outcome = final_training_data['val_or_test_outcome']

    X_train_reward, y_train_rewards = final_training_data['train_reward']
    X_test_reward, y_test_rewards = final_training_data['val_or_test_reward']

    # Suggested hyperparameters from cross-validation
    suggested_params_outcome = cv_results['suggested_params_outcome']
    suggested_params_reward = cv_results['suggested_params_reward']

    # Train final outcome model
    outcome_model = OutcomeModel(instantiate(cfg.models.outcome.classifier))
    final_outcome_model = outcome_model.train(X_train_outcome, y_train_outcome, **suggested_params_outcome)
    final_outcome_accuracy = outcome_model.evaluate(X_test_outcome, y_test_outcome)
    print(f"Final Outcome Model Accuracy: {final_outcome_accuracy}")

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
        outcome_model=final_outcome_model,
        reward_models=final_reward_models,
        onehot_encoder=final_training_data['onehot_encoder'],
        cfg=cfg
    )
    
    # Get decisions for the test set
    all_expected_rewards, all_decisions, all_clsf_pred, decisions_df = decision_processor.get_decisions(X_test_reward)

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
        decisions_df=decisions_df,
        unscaled_X_val_reward=final_training_data['unscaled_val_or_test_set'],
        expected_rewards_list=all_expected_rewards,
        clfr_pred_list=all_clsf_pred,
    )

    print("Final evaluation on test set completed.")
    return test_results_dict
