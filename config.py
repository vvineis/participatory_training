from ranking_criteria.a2_get_ranking_criteria import create_ranking_criteria
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Actor and Action Lists
actor_list = ['Bank', 'Applicant', 'Regulatory', 'Oracle', 'Classifier']
reward_types=['Bank', 'Applicant', 'Regulatory']

actions_set = ['Grant', 'Not Grant', 'Grant lower']
positive_actions_set = ['Grant','Grant lower']
outcomes_set = ['Fully Repaid', 'Partially Repaid', 'Not Repaid'] 

# Feature Columns and Columns to Display
feature_columns = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate']
columns_to_display = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate', 'Applicant Type', 'Recoveries']

# Decision Criteria List
decision_criteria_list = [
    'Maximin', 'Nash Social Welfare', 'Kalai-Smorodinsky',
    'Nash Bargaining', 'Compromise Programming', 'Proportional Fairness'
]

#Define metrics
fairness_metrics_list=['Demographic Parity', 'Equal Opportunity', 'Equalized Odds', 'Calibration']
standard_metrics_list=['Precision', 'Recall', 'F1 Score', 'Accuracy']
case_metrics_list=[ 'Total Profit',  'Total Loss', 'Unexploited Profit']

positive_attribute_for_fairness= 'Applicant Type' #note: should be boolean 


# Ranking Criteria, Metrics used for Evaluation and Ranking Weights
ranking_criteria, metrics_for_evaluation, ranking_weights = create_ranking_criteria()

#Models for the outcome (classifier) and reward (regressor) prediction 
classifier= RandomForestClassifier()
regressor= RandomForestRegressor

# Parameter Grids for Hyperparameter Tuning
param_grid_outcome = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}

param_grid_reward = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
}
