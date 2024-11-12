from utils.models.outcome_model import OutcomeModel
from utils.models.reward_models import RewardModels
from utils.decisions.get_decisions import DecisionProcessor
from utils.decisions.evaluate_decisions import SummaryProcessor
from utils.decisions.compromise_functions import MaxIndividualReward
from utils.metrics.get_metrics import MetricsCalculator

def run_final_evaluation(data_processor, cv_results, all_train_set, test_set, classifier, regressor, cfg):
    # Prepare training and test sets
    final_training_data = data_processor.prepare_for_training(all_train_set, test_set)

    X_train_outcome, y_train_outcome = final_training_data['train_outcome']
    X_test_outcome, y_test_outcome = final_training_data['val_or_test_outcome']
    X_train_reward, y_train_bank, y_train_applicant, y_train_regulatory = final_training_data['train_reward']
    X_test_reward, y_test_bank, y_test_applicant, y_test_regulatory = final_training_data['val_or_test_reward']

    # Suggested hyperparameters from cross-validation
    suggested_params_outcome = cv_results['suggested_params_outcome']
    suggested_params_reward = cv_results['suggested_params_reward']

    # Train final outcome model
    outcome_model = OutcomeModel(classifier)
    final_outcome_model = outcome_model.train(X_train_outcome, y_train_outcome, **suggested_params_outcome)
    final_outcome_accuracy = outcome_model.evaluate(X_test_outcome, y_test_outcome)
    print(f"Final Outcome Model Accuracy: {final_outcome_accuracy}")

    final_training_data['unscaled_val_or_test_set'].columns
    # Train final reward models
    reward_model = RewardModels(regressor, **suggested_params_reward)

    # Train the models
    final_reward_models = reward_model.train(
        X_train_reward, y_train_bank, y_train_applicant, y_train_regulatory
    )

    # Evaluate the models on the test set
    mse_results = reward_model.evaluate(
        X_test_reward, y_test_bank, y_test_applicant, y_test_regulatory
    )

    # Extract individual MSEs
    mse_bank = mse_results['bank_mse']
    mse_applicant = mse_results['applicant_mse']
    mse_regulatory = mse_results['regulatory_mse']

    print(f"Final Reward Models MSE: Bank: {mse_bank}, Applicant: {mse_applicant}, Regulatory: {mse_regulatory}")
    
    # Process test set using final models
    decision_processor = DecisionProcessor(
        outcome_model=final_outcome_model,
        reward_models=final_reward_models,
        onehot_encoder=final_training_data['onehot_encoder'],
        actions_set=cfg.setting.actions_set,
        feature_columns=cfg.setting.feature_columns,
        categorical_columns=cfg.categorical_columns,
        actor_list=cfg.setting.actor_list,
        decision_criteria_list=cfg.criteria.decision_criteria,
        ranking_criteria=cfg.criteria.ranking_criteria, 
        ranking_weights=cfg.criteria.ranking_weights,
        metrics_for_evaluation=cfg.criteria.metrics_for_evaluation
    )
    
    # Get decisions for the test set
    all_expected_rewards, all_decisions, all_clsf_pred, decisions_df = decision_processor.get_decisions(X_test_reward)

    # Evaluate final decisions on the test set
    max_individual_strategy = MaxIndividualReward()
    metrics_calculator = MetricsCalculator(
        fairness_metrics_list=cfg.metrics.fairness_metrics,
        standard_metrics_list=cfg.metrics.standard_metrics,
        case_metrics_list=cfg.metrics.case_specific_metrics,
        actions_set=cfg.setting.actions_set,
        outcomes_set=cfg.setting.outcomes_set,
        positive_actions_set=cfg.setting.positive_actions_set
    )
    summary_processor = SummaryProcessor(
        metrics_calculator=metrics_calculator,
        ranking_criteria=cfg.criteria.ranking_criteria,
        ranking_weights=cfg.criteria.ranking_weights,
        metrics_for_evaluation=cfg.criteria.metrics_for_evaluation,
        reward_types=cfg.setting.reward_types,
        decision_criteria_list=cfg.criteria.decision_criteria,
        actions_set=cfg.setting.actions_set,
        outcomes_set=cfg.setting.outcomes_set,
        strategy=max_individual_strategy
    )

    # Process metrics on test set
    test_results_dict = summary_processor.process_decision_metrics(
        actor_list=cfg.setting.actor_list,
        y_val_outcome=y_test_outcome,
        decisions_df=decisions_df,
        unscaled_X_val_reward=final_training_data['unscaled_val_or_test_set'],
        expected_rewards_list=all_expected_rewards,
        clfr_pred_list=all_clsf_pred,
        positive_attribute_for_fairness=cfg.metrics.positive_attribute_for_fairness
    )

    print("Final evaluation on test set completed.")
    return test_results_dict
