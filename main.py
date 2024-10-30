from a1_get_rewards import RewardCalculator
import pandas as pd
from b0_prepare_data import DataProcessor
from config import (
    actor_list, reward_types, actions_set, decision_criteria_list,
    feature_columns, columns_to_display, ranking_criteria,
    metrics_for_evaluation, ranking_weights, classifier, regressor, param_grid_outcome, param_grid_reward
)
from get_cv_results import CrossValidator

df = pd.read_csv('data/lending_club_data.csv')

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
    n_splits=3,
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
    n_splits=3,
    process_train_val_folds=process_train_val_folds,
    feature_columns=feature_columns, 
    categorical_columns=categorical_columns,
    actions_set=actions_set,
    actor_list=actor_list,
    decision_criteria_list=decision_criteria_list,
    ranking_criteria=ranking_criteria,
    ranking_weights=ranking_weights,
    metrics_for_evaluation=metrics_for_evaluation
)

# Run cross-validation with hyperparameter tuning
results = cross_validator.run()

print("Best outcome parameters per fold:", results['best_hyperparams_outcome_per_fold'])
print("Suggested outcome parameters:", results['suggested_params_outcome'])
print("Suggested reward parameters:", results['suggested_params_reward'])