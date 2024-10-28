from get_ranking_criteria import create_ranking_criteria
from get_rewards import RewardCalculator
import pandas as pd
from prepare_data import DataProcessor


actor_list = ['Bank', 'Applicant', 'Regulatory', 'Oracle', 'Classifier']
actions_set = ['Grant', 'Not Grant', 'Grant lower']
decision_criteria_list = ['Maximin', 'Nash Social Welfare', 'Kalai-Smorodinsky', 'Nash Bargaining', 'Compromise Programming', 'Proportional Fairness']
feature_columns = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate']
columns_to_display = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate', 'Applicant Type', 'Recoveries']
ranking_criteria, metrics_for_evaluation, ranking_weights = create_ranking_criteria()


param_grid_outcome = {
    'n_estimators': [100,200, 300],
    'max_depth': [None,10, 20],
}

param_grid_reward = {
    'n_estimators': [50,100, 150],
    'max_depth': [None,10,20],
}

df = pd.read_csv('data/lending_club_data.csv')
df.shape

reward_calculator = RewardCalculator()
df_ready = reward_calculator.compute_rewards(df)

# Define categorical columns for processing
categorical_columns = ['Action', 'Outcome']

# Initialize the DataProcessor
data_processor = DataProcessor(
    df=df_ready,
    feature_columns=feature_columns,
    columns_to_display=columns_to_display,
    categorical_columns=categorical_columns,
    test_size=0.2,
    n_splits=5,
    random_split=True
)
    

process_train_val_folds, all_train_set, test_set = data_processor.process()

print("Train set shape:", all_train_set.shape)
print("Test set shape:", test_set.shape)
