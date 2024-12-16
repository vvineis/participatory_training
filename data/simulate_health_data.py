from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import random
from causalml.inference.meta import BaseTRegressor, BaseXRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from causalml.inference.meta import BaseXRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the number of observations
n_obs = 3000

# Generate patient demographics
genders = np.random.choice(['M', 'F'], size=n_obs)
ages = (np.random.beta(a=5, b=2, size=n_obs) * 67 + 18).astype(int)
regions = np.random.choice(['Urban', 'Rural'], size=n_obs)

# Smoking status (higher prevalence in males)
smoking_status = [
    np.random.choice(['Smoker', 'Non-smoker'], p=[0.3, 0.7]) if gender == 'M' else
    np.random.choice(['Smoker', 'Non-smoker'], p=[0.1, 0.9])
    for gender in genders
]

# Comorbidities based on smoking and age
# Comorbidities based on smoking and age
comorbidities = []
for age, smoker in zip(ages, smoking_status):
    base_prob = [0.05, 0.2, 0.45, 0.3] if smoker == 'Smoker' else [0.2, 0.4, 0.3, 0.1]
    if age > 60:
        prob = [p * 0.5 for p in base_prob]  # Scale probabilities for older patients
    else:
        prob = base_prob
    
    # Normalize probabilities to ensure they sum to 1
    prob = [p / sum(prob) for p in prob]
    
    # Sample comorbidities
    comorbidities.append(np.random.choice([0, 1, 2, 3], p=prob))


# Physical activity
physical_activity = [
    np.random.choice(['Sedentary', 'Moderate', 'Active'], p=[0.6, 0.3, 0.1]) if age > 60 else
    np.random.choice(['Sedentary', 'Moderate', 'Active'], p=[0.3, 0.4, 0.3])
    for age in ages
]

# Treatment assignment: only A and C
treatments = [
    np.random.choice(['A', 'C'], p=[0.6, 0.4]) if region == 'Urban' else
    np.random.choice(['A', 'C'], p=[0.8, 0.2])
    for region in regions
]
unique, counts = np.unique(treatments, return_counts=True)
print(dict(zip(unique, counts)))

# Recovery time
base_recovery_time = {'A': 5.0, 'C': 3.5}
recovery_times = []
for i in range(n_obs):
    treatment = treatments[i]
    factors = [
        (ages[i] / 100) * np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.2),
        comorbidities[i] * np.random.uniform(0.6, 1.4) + np.random.normal(0, 0.3),
        (0.8 if regions[i] == 'Urban' else 1.0) + np.random.normal(0, 0.1),
        (1.0 if smoking_status[i] == 'Smoker' else 0.0) + np.random.normal(0, 0.2),
        (-0.5 if physical_activity[i] == 'Active' else (0.0 if physical_activity[i] == 'Moderate' else 0.5)) + np.random.normal(0, 0.1),
        np.random.normal(0, 0.5)
    ]
    recovery_time = base_recovery_time[treatment] + sum(factors)
    recovery_time = min(12.0, max(1.0, recovery_time))
    recovery_time=round(recovery_time,0)
    recovery_times.append(recovery_time)


# Costs based on recovery time
base_cost = {'A': 4000, 'C': 8000}
costs = [
    base_cost[treatments[i]] + recovery_times[i] * 400
    for i in range(n_obs)
]

# Create the DataFrame
patient_data = pd.DataFrame({
    'Gender': genders,
    'Age': ages,
    'Comorbidities': comorbidities,
    'Region': regions,
    'SmokingStatus': smoking_status,
    'PhysicalActivity': physical_activity,
    'Treatment': treatments,
    'RecoveryWeeks': recovery_times,
    'Cost': costs
})

print((patient_data.head()))


# One-hot encode the features
X = pd.get_dummies(patient_data[['Age', 'Comorbidities', 'Gender', 'Region', 'SmokingStatus']], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Define treatment and outcome
treatment = patient_data['Treatment'].astype(str)
y = patient_data['RecoveryWeeks']

# Split data into training and testing sets
X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
    X, y, treatment, test_size=0.2, random_state=42
)

# Initialize the X-Learner
learner_x = BaseXRegressor(
    learner=XGBRegressor(random_state=42),
    control_name='C'  # Control group is 'C'
)

# Fit the model using training data
learner_x.fit(X=X_train, treatment=treatment_train, y=y_train)

# Predict outcomes for the test set
predicted_outcomes_A = []
predicted_outcomes_C = []
actual_outcomes = []
actual_treatment=[]

for i in range(len(X_test)):
    # Get the row as a DataFrame (single test patient)
    patient = X_test.iloc[i:i+1]

    # Predict baseline outcome for 'C'
    predicted_outcome_C = learner_x.models_mu_c['A'].predict(patient)
    predicted_outcomes_C.append(np.round(predicted_outcome_C,0))
    cate = learner_x.predict(patient, treatment='A')

    # Calculate the absolute predicted outcome
    predicted_outcome_A = np.round(predicted_outcome_C  + cate,0)
    predicted_outcomes_A.append(predicted_outcome_A)

    # Store the actual outcome
    actual_outcomes.append(y_test.iloc[i])
    actual_treatment.append(treatment_test)

print(f'treat {treatment_test[0:5]}, actual_outcome {y_test[0:5]}, pred_A {predicted_outcomes_A[0:5]}, pred_C {predicted_outcomes_C[0:5]}')

