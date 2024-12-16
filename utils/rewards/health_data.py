import numpy as np
import pandas as pd
import random


def simulate_patient_data(n_obs=3000, random_seed=42):
    """
    Simulates patient data for treatment recovery analysis.

    Args:
        n_obs (int): Number of observations to generate.
        random_seed (int): Seed for random number generation.

    Returns:
        pd.DataFrame: A DataFrame containing simulated patient data.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

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
    comorbidities = []
    for age, smoker in zip(ages, smoking_status):
        base_prob = [0.05, 0.2, 0.45, 0.3] if smoker == 'Smoker' else [0.2, 0.4, 0.3, 0.1]
        if age > 60:
            prob = [p * 0.5 for p in base_prob]  # Scale probabilities for older patients
        else:
            prob = base_prob

        # Normalize probabilities to ensure they sum to 1
        prob = [p / sum(prob) for p in prob]
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
        recovery_times.append(round(recovery_time, 0))

    # Costs based on recovery time
    base_cost = {'A': 6000, 'C': 2000}
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

    return patient_data


# Example usage
if __name__ == "__main__":
    simulated_data = simulate_patient_data()
    print(simulated_data.head())

