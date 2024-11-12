import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.preprocessing import DataProcessor
from utils.rewards.get_rewards import RewardCalculator
from src.cross_validation_process import CrossValidator
from src.final_evaluation import run_final_evaluation
from hydra.utils import instantiate
import os

@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print full configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    print("Actor List:", cfg.setting.actor_list)
    print("Reward Types:", cfg.setting.reward_types)

    print("Type of param_grid:", type(cfg.models.rewards.param_grid))
    param_grid = dict(cfg.models.rewards.param_grid)
    print("Type of param_grid:", type(param_grid))
    print("Contents of param_grid:", param_grid)

    # Load data
    df = pd.read_csv(cfg.data_path)
    df = df.iloc[0:cfg.sample_size]  # Limit data if needed for testing

    # Initialize RewardCalculator with reward types from configuration
    reward_calculator = RewardCalculator(cfg.setting.reward_types)
    df_ready = reward_calculator.compute_rewards(df)

    # Instantiate ranking criteria if needed (if create_ranking_criteria returns specific values)
    # Access the evaluation criteria directly
    ranking_criteria = cfg.criteria.ranking_criteria
    metrics_for_evaluation = cfg.criteria.metrics_for_evaluation
    ranking_weights = cfg.criteria.ranking_weights

    print("Ranking Criteria:", ranking_criteria)
    print("Metrics for Evaluation:", metrics_for_evaluation)
    print("Ranking Weights:", ranking_weights)
    # Initialize the DataProcessor
    data_processor = DataProcessor(
        df=df_ready,
        cfg= cfg, 
        random_split=True
    )

    process_train_val_folds, all_train_set, test_set = data_processor.process()
    print("Train set shape:", all_train_set.shape)
    print("Test set shape:", test_set.shape)


    # Initialize the CrossValidator with instantiated classifier and regressor
    cross_validator = CrossValidator(
        cfg=cfg,
        process_train_val_folds=process_train_val_folds
    )


    cross_validator.run()
    cross_validator.print_best_params_per_fold()

    # Aggregate CV results and check the summary lengths
    cv_results = cross_validator.aggregate_cv_results()
    cross_validator.check_summary_lengths()
    print("Aggregated CV Results:")
    print(cv_results)

    print("Training final models on entire training set and evaluating on test set...")
    final_results = run_final_evaluation(data_processor, cv_results, all_train_set, test_set,  cfg)
    print("Final evaluation results:")
    print(final_results)

    os.makedirs(cfg.result_path, exist_ok=True)

    # Save results to the specified directory
    cv_results['ranked_decision_metrics_df'].to_csv(os.path.join(cfg.result_path, 'cv_ranked_decision_metrics.csv'), index=False)
    final_results['ranked_decision_metrics_df'].to_csv(os.path.join(cfg.result_path, 'final_ranked_decision_metrics.csv'), index=False)

    print(f"Results saved to the '{cfg.result_path}' folder")

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

