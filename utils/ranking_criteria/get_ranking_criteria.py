def create_ranking_criteria():
    # Define ranking criteria
    ranking_criteria = {
        # Use worst-case for Demographic_Parity, Equal_Opportunity, Equalized Odds, and Calibration
        'Demographic_Parity_Worst': 'zero',
        'Equal_Opportunity_Worst': 'zero',
        'Equalized Odds_Worst': 'zero',
        'Calibration_Worst': 'zero',

        # Other metrics: Use 'max' for metrics where higher values are better
        'Total_Profit': 'max',
        'Accuracy': 'max',
        'Unexploited_Profit': 'min'
    }

    # Define weights corresponding to each criterion (assign higher weight for more important metrics)
    ranking_weights = {
        'Demographic_Parity_Worst': 0.2,
        'Equal_Opportunity_Worst': 0,
        'Equalized Odds_Worst': 0,
        'Calibration_Worst': 0.2,
        'Total_Profit': 0.2,
        'Accuracy': 0.4,
        'Unexploited_Profit': 0.
    }

    return {
        "ranking_criteria": ranking_criteria,
        "metrics_for_evaluation": ranking_criteria.keys(),
        "ranking_weights": ranking_weights,
    } 