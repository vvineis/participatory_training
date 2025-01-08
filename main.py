import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.preprocessing import DataProcessor
from utils.rewards.get_rewards import RewardCalculator
from src.cross_validation_process import CrossValidator
from src.final_evaluation import run_final_evaluation
from hydra.utils import instantiate
import os

import os
import time
from omegaconf import OmegaConf

def save_results(cfg, cv_results, final_results):
    # Create a unique folder for each run
    unique_id = time.strftime("%Y%m%d-%H%M%S")  # Timestamp for uniqueness
    result_subfolder = os.path.join(cfg.result_path, f"run_{unique_id}_Accuracy_{cfg.ranking_weights.Accuracy}")
    os.makedirs(result_subfolder, exist_ok=True)

    # Save results to the specific subfolder
    cv_results['ranked_decision_metrics_df'].to_csv(
        os.path.join(result_subfolder, 'cv_ranked_decision_metrics.csv'), index=False
    )
    final_results['ranked_decision_metrics_df'].to_csv(
        os.path.join(result_subfolder, 'final_ranked_decision_metrics.csv'), index=False
    )

    print(f"Results saved to: {result_subfolder}")


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Criteria Key:", cfg.get("criteria", "Not Found"))

    # Load data
    df = pd.read_csv(cfg.data_path)
    df = df.iloc[0:cfg.sample_size]  # Limit data if needed for testing

    # Initialize RewardCalculator with reward types from configuration
    # Dynamically instantiate the correct reward calculator
    reward_calculator = instantiate(cfg.reward_calculator)

    print(f"Initialized Reward Calculator: {reward_calculator}")
    df_ready = reward_calculator.compute_rewards(df)
    #print(f"df_ready:{df_ready[['Action', 'Outcome', 'x23', 'Parent_reward', 'Healthcare_Provider_reward', 'Policy_Maker_reward']].head()}")


    # Instantiate ranking criteria if needed (if create_ranking_criteria returns specific values)
    ranking_criteria = cfg.criteria.ranking_criteria

    metrics_for_evaluation = cfg.criteria.metrics_for_evaluation
    ranking_weights = cfg.ranking_weights
    #print("Decision Criteria:", cfg.decision_criteria)

    #print("Ranking Criteria:", ranking_criteria)
    #print("Metrics for Evaluation:", metrics_for_evaluation)
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

    cv_results = cross_validator.aggregate_cv_results()
    cross_validator.check_summary_lengths()
    print("Aggregated CV Results:")
    print(cv_results)

    print("Training final models on entire training set and evaluating on test set...")
    final_results = run_final_evaluation(data_processor, cv_results, all_train_set, test_set,  cfg)
    print("Final evaluation results:")
    print(final_results)

    os.makedirs(cfg.result_path, exist_ok=True)

    save_results(cfg, cv_results, final_results)

    # Save results to the specified directory
    #cv_results['ranked_decision_metrics_df'].to_csv(os.path.join(cfg.result_path, 'cv_ranked_decision_metrics.csv'), index=False)
    #final_results['ranked_decision_metrics_df'].to_csv(os.path.join(cfg.result_path, 'final_ranked_decision_metrics.csv'), index=False)

    print(f"Results saved to the '{cfg.result_path}' folder")

if __name__ == "__main__":
    main()

