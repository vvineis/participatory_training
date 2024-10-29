from get_ranking_criteria import create_ranking_criteria

# Actor and Action Lists
actor_list = ['Bank', 'Applicant', 'Regulatory', 'Oracle', 'Classifier']
reward_types=['Bank', 'Applicant', 'Regulatory']

actions_set = ['Grant', 'Not Grant', 'Grant lower']

# Decision Criteria List
decision_criteria_list = [
    'Maximin', 'Nash Social Welfare', 'Kalai-Smorodinsky',
    'Nash Bargaining', 'Compromise Programming', 'Proportional Fairness'
]

# Feature Columns and Columns to Display
feature_columns = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate']
columns_to_display = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate', 'Applicant Type', 'Recoveries']

# Ranking Criteria, Metrics for Evaluation, and Ranking Weights
ranking_criteria, metrics_for_evaluation, ranking_weights = create_ranking_criteria()

# Parameter Grids for Hyperparameter Tuning
param_grid_outcome = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}

param_grid_reward = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
}

print(ranking_criteria)