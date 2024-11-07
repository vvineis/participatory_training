from preprocessing import DataProcessor
from rewards.get_rewards import RewardCalculator
from cross_validation_process import CrossValidator
from final_evaluation import run_final_evaluation
import pandas as pd

from config import (
    actor_list, reward_types, positive_actions_set, actions_set, decision_criteria_list, reward_types,
    feature_columns, columns_to_display, ranking_criteria,
    metrics_for_evaluation, ranking_weights, classifier, regressor, param_grid_outcome, param_grid_reward, positive_attribute_for_fairness
)


df = pd.read_csv('data/lending_club_data.csv')
df=df.iloc[0:1000]

reward_calculator = RewardCalculator(reward_types)
df_ready = reward_calculator.compute_rewards(df)

# Define categorical columns for processing
categorical_columns = ['Action', 'Outcome']

# Initialize the DataProcessor
data_processor = DataProcessor(
    df=df_ready,
    feature_columns=feature_columns,
    columns_to_display=columns_to_display,
    categorical_columns=categorical_columns,
    reward_types=reward_types,
    test_size=0.2,
    n_splits=5,
    random_split=True
)
    

process_train_val_folds, all_train_set, test_set = data_processor.process()

print("Train set shape:", all_train_set.shape)
print("Test set shape:", test_set.shape)


cross_validator = CrossValidator(
    classifier=classifier,
    regressor= regressor,
    param_grid_outcome=param_grid_outcome,
    param_grid_reward=param_grid_reward,
    n_splits=5,
    process_train_val_folds=process_train_val_folds,
    feature_columns=feature_columns, 
    categorical_columns=categorical_columns,
    actions_set=actions_set,
    actor_list=actor_list,
    reward_types=reward_types,
    decision_criteria_list=decision_criteria_list,
    ranking_criteria=ranking_criteria,
    ranking_weights=ranking_weights,
    metrics_for_evaluation=metrics_for_evaluation,
    positive_attribute_for_fairness=positive_attribute_for_fairness
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

cv_results['ranked_decision_metrics_df'].to_csv('results/cv_ranked_decision_metrics.csv', index=False)
final_results['ranked_decision_metrics_df'].to_csv('results/final_ranked_decision_metrics.csv', index=False)

print("Results saved to the 'results' folder")