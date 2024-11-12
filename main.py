from src.preprocessing import DataProcessor
from utils.rewards.get_rewards import RewardCalculator
from src.cross_validation_process import CrossValidator
from src.final_evaluation import run_final_evaluation
from utils.ranking_criteria.get_ranking_criteria import create_ranking_criteria
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


#from config import (
  #  actor_list, reward_types, positive_actions_set, actions_set, decision_criteria_list, reward_types,
    #feature_columns, columns_to_display, ranking_criteria,
   # metrics_for_evaluation, ranking_weights, classifier, regressor, param_grid_outcome, param_grid_reward, positive_attribute_for_fairness
#)


@hydra.main(version_base="1.1",config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print full configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # Load data
    df = pd.read_csv(cfg.data_path)
    df = df.iloc[0:cfg.sample_size]  # Limit data if needed for testing

    # Initialize RewardCalculator with reward types from configuration
    reward_calculator = RewardCalculator(cfg.actors.reward_types)
    df_ready = reward_calculator.compute_rewards(df)

    # Instantiate ranking criteria if needed (if create_ranking_criteria returns specific values)
    criteria = instantiate(cfg.evaluation_criteria)
    
    # Extract values from criteria
    ranking_criteria = criteria["ranking_criteria"]
    metrics_for_evaluation = criteria["metrics_for_evaluation"]
    ranking_weights = criteria["ranking_weights"]

    # Initialize the DataProcessor
    data_processor = DataProcessor(
        df=df_ready,
        feature_columns=cfg.context.feature_columns,
        columns_to_display=cfg.context.columns_to_display,
        categorical_columns=cfg.categorical_columns,
        reward_types=cfg.actors.reward_types,
        test_size=cfg.test_size,
        n_splits=cfg.cv_splits,
        random_split=True
    )

    process_train_val_folds, all_train_set, test_set = data_processor.process()

    print("Train set shape:", all_train_set.shape)
    print("Test set shape:", test_set.shape)

    # Instantiate classifier and regressor models
    classifier = instantiate(cfg.models.outcome_classifier)
    regressor = instantiate(cfg.models.rewards_regressor)

    # Initialize the CrossValidator
    cross_validator = CrossValidator(
        classifier=classifier,
        regressor=regressor,
        param_grid_outcome=cfg.models.param_grid_outcome,
        param_grid_reward=cfg.models.param_grid_rewards.param_grid_rewards, 
        n_splits=cfg.cv_splits,
        process_train_val_folds=process_train_val_folds,
        feature_columns=cfg.context.feature_columns, 
        categorical_columns=cfg.categorical_columns,
        actions_set=cfg.action_outcomes.actions_set,
        actor_list=cfg.actors.actor_list,
        reward_types=cfg.actors.reward_types,
        decision_criteria_list=cfg.decision_criteria.decision_criteria_list,
        ranking_criteria=ranking_criteria,
        ranking_weights=ranking_weights,
        metrics_for_evaluation=metrics_for_evaluation,
        positive_attribute_for_fairness=cfg.metrics.fairness.positive_attribute
    )

    cross_validator.run()
    cross_validator.print_best_params_per_fold()

    # Aggregate CV results and check the summary lengths
    cv_results = cross_validator.aggregate_cv_results()
    cross_validator.check_summary_lengths()

    print("Aggregated CV Results:")
    print(cv_results)

    print("Training final models on entire training set and evaluating on test set...")
    final_results = run_final_evaluation(data_processor, cv_results, all_train_set, test_set)

    print("Final evaluation results:")
    print(final_results)

    # Save results
    cv_results['ranked_decision_metrics_df'].to_csv('results/cv_ranked_decision_metrics.csv', index=False)
    final_results['ranked_decision_metrics_df'].to_csv('results/final_ranked_decision_metrics.csv', index=False)

    print("Results saved to the 'results' folder")

if __name__ == "__main__":
    main()



#def main(cfg: DictConfig):
   # df = pd.read_csv('data/lending_club_data.csv')
  #  df=df.iloc[0:1000]

   # reward_calculator = RewardCalculator(reward_types)
   # df_ready = reward_calculator.compute_rewards(df)

    # Define categorical columns for processing
  #  categorical_columns = ['Action', 'Outcome']

    # Initialize the DataProcessor
   # data_processor = DataProcessor(
   #     df=df_ready,
    #    feature_columns=feature_columns,
    #    columns_to_display=columns_to_display,
    #    categorical_columns=categorical_columns,
    #    reward_types=reward_types,
    #    test_size=0.2,
    #    n_splits=5,
    #    random_split=True
   # )
        

    #process_train_val_folds, all_train_set, test_set = data_processor.process()

  #  print("Train set shape:", all_train_set.shape)
   # print("Test set shape:", test_set.shape)


    #cross_validator = CrossValidator(
    #    classifier=classifier,
    #    regressor= regressor,
    #    param_grid_outcome=param_grid_outcome,
    #    param_grid_reward=param_grid_reward,
    #    n_splits=5,
    #    process_train_val_folds=process_train_val_folds,
    #    feature_columns=feature_columns, 
    #    categorical_columns=categorical_columns,
    #    actions_set=actions_set,
    #    actor_list=actor_list,
    #    reward_types=reward_types,
    #    decision_criteria_list=decision_criteria_list,
    #    ranking_criteria=ranking_criteria,
    #    ranking_weights=ranking_weights,
   #     metrics_for_evaluation=metrics_for_evaluation,
   #     positive_attribute_for_fairness=positive_attribute_for_fairness
   # )

  #  cross_validator.run()
  #  cross_validator.print_best_params_per_fold()

    # Aggregate CV results and check the summary lengths
  #  cv_results = cross_validator.aggregate_cv_results()
 #   cross_validator.check_summary_lengths()

  #  print("Aggregated CV Results:")
 #   print(cv_results)

  #  print("Training final models on entire training set and evaluating on test set...")
  #  final_results = run_final_evaluation(data_processor, cv_results, all_train_set, test_set)
#
   # print("Final evaluation results:")
    #print(final_results)

    #cv_results['ranked_decision_metrics_df'].to_csv('results/cv_ranked_decision_metrics.csv', index=False)
   # final_results['ranked_decision_metrics_df'].to_csv('results/final_ranked_decision_metrics.csv', index=False)

   # print("Results saved to the 'results' folder")

