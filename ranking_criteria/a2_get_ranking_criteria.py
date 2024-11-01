def create_ranking_criteria():
    # Define ranking criteria
    ranking_criteria = {
        # Use worst-case for Demographic Parity, Equal Opportunity, Equalized Odds, and Calibration
        'Demographic Parity_Worst': 'zero',
        'Equal Opportunity_Worst': 'zero',
        'Equalized Odds_Worst': 'zero',
        'Calibration_Worst': 'zero',

        # Other metrics: Use 'max' for metrics where higher values are better
        'Total Profit': 'max',
        'Accuracy': 'max',
        'Unexploited Profit': 'min'
    }

    # Define weights corresponding to each criterion (assign higher weight for more important metrics)
    ranking_weights = {
        'Demographic Parity_Worst': 0.2,
        'Equal Opportunity_Worst': 0,
        'Equalized Odds_Worst': 0,
        'Calibration_Worst': 0.2,
        'Total Profit': 0.2,
        'Accuracy': 0.4,
        'Unexploited Profit': 0.
    }

    # Return both the ranking criteria and weights
    return ranking_criteria, ranking_criteria.keys(), ranking_weights