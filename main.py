import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.preprocessing import DataProcessor
from utils.rewards.get_rewards import RewardCalculator
from src.cross_validation_process import CrossValidator
from src.final_evaluation import run_final_evaluation
from hydra.utils import instantiate
import os
import time


def save_results(cfg, cv_results, final_results, suggested_params_outcome, suggested_params_reward, final_outcome_score):
    # Create a unique folder for each run
    unique_id = time.strftime("%Y%m%d-%H%M%S")  # Timestamp for uniqueness
    result_subfolder = os.path.join(cfg.result_path, f"run_{unique_id}_Acc_{cfg.ranking_weights.Accuracy}_Fair_{cfg.ranking_weights.Demographic_Parity}")
    os.makedirs(result_subfolder, exist_ok=True)

    # Save results to the specific subfolder
    cv_results['ranked_decision_metrics_df'].to_csv(
        os.path.join(result_subfolder, 'cv_ranked_decision_metrics.csv'), index=False
    )
    final_results['ranked_decision_metrics_df'].to_csv(
        os.path.join(result_subfolder, 'final_ranked_decision_metrics.csv'), index=False
    )

    # Save text results
    text_file_path = os.path.join(result_subfolder, 'suggested_params_and_scores.txt')
    with open(text_file_path, 'w') as f:
        f.write(f"Suggested Params Outcome: {suggested_params_outcome}\n")
        f.write(f"Suggested Params Reward: {suggested_params_reward}\n")
        f.write(f"Final Outcome Score: {final_outcome_score}\n")
        f.write(f"Reward type: Strictest\n")
        f.write(f"samples:{cfg.sample_size}\n")
        f.write(f"cvfolds:{cfg.cv_splits}\n")
        f.write(f"seed: 111\n")


    print(f"Results saved to: {result_subfolder}")


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Criteria Key:", cfg.get("criteria", "Not Found"))

    # Load data
    df = pd.read_csv(cfg.data_path)
    df = df.iloc[0:cfg.sample_size] 

    reward_calculator = instantiate(cfg.reward_calculator)

    print(f"Initialized Reward Calculator: {reward_calculator}")
    df_ready = reward_calculator.compute_rewards(df)
    
    # Process data
    data_processor = DataProcessor(
        df=df_ready,
        cfg= cfg, 
        random_split=True
    )

    process_train_val_folds, all_train_set, test_set = data_processor.process()
    print("Train set shape:", all_train_set.shape)
    print("Test set shape:", test_set.shape)

    # Cross-validation for HPO and decision strategies assessment and ranking
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

    # Train final models on entire training set and evaluate on test set
    print("Training final models on entire training set and evaluating on test set...")
    final_results, suggested_params_outcome, suggested_params_reward, final_outcome_score = run_final_evaluation(data_processor, cv_results, all_train_set, test_set,  cfg)
    print("Final evaluation results:")
    print(final_results)
    
    # Save results
    os.makedirs(cfg.result_path, exist_ok=True)

    save_results(cfg, cv_results, final_results, suggested_params_outcome, suggested_params_reward, final_outcome_score)

    print(f"Results saved to the '{cfg.result_path}' folder")

if __name__ == "__main__":
    main()

