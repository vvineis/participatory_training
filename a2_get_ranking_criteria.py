def create_ranking_criteria():
    # Define ranking criteria
    ranking_criteria = {
        # Use worst-case for Demographic Parity, Equal Opportunity, Equalized Odds, and Calibration
        'Demographic Parity_Worst': 'zero',
        'Equal Opportunity_Worst': 'zero',
        'Equalized Odds_Worst': 'zero',
        'Calibration_Worst': 'zero',

        # Other metrics: Use 'max' for metrics where higher values are better
        'Profit': 'max',
        'Accuracy': 'max',
        'Unexploited Profit': 'min'
    }

    # Define weights corresponding to each criterion (assign higher weight for more important metrics)
    ranking_weights = {
        'Demographic Parity_Worst': 0.1,
        'Equal Opportunity_Worst': 0.05,
        'Equalized Odds_Worst': 0.05,
        'Calibration_Worst': 0.05,
        'Profit': 0.1,
        'Accuracy': 0.6,
        'Unexploited Profit': 0.05
    }

    # Return both the ranking criteria and weights
    return ranking_criteria, ranking_criteria.keys(), ranking_weights