
class MetricsCalculator(): 
    

def _compute_max_min_fairness(value1, value2, value3=None):
    """ Compute max-min fairness as the maximum absolute difference from 0. """
    if value3 is not None:
        return max(abs(value1), abs(value2), abs(value3))
    return max(abs(value1), abs(value2))

def compute_all_metrics(suggestions_df, actor_list, actions_set, decision_criteria_list, true_outcome_col='True Outcome'):
    metrics = {actor: {} for actor in actor_list + decision_criteria_list}
    group_col = 'Applicant Type'  # The column dividing the group
    positive_group_value = 1      # Group considered vulnerable

    # Pre-compute and cache common fairness metrics and action counts
    fairness_cache = {}
    action_counts_cache = {}
    for actor in actor_list + decision_criteria_list:
        decision_col = f'{actor} Suggested Action' if actor in actor_list else actor

        if decision_col not in suggestions_df.columns:
            continue

        # Fairness Metrics: Cache the result if not already computed
        if decision_col not in fairness_cache:
            fairness_calculator = FairnessMetrics(suggestions_df, decision_col, group_col, positive_group_value, true_outcome_col)
            fairness_cache[decision_col] = {
                'Demographic Parity': fairness_calculator.compute_demographic_parity(),
                'Equal Opportunity': fairness_calculator.compute_equal_opportunity(),
                'Equalized Odds': fairness_calculator.compute_equalized_odds(),
                'Calibration': fairness_calculator.compute_calibration()
            }

        # Action Counts: Cache action percentages for efficiency
        if decision_col not in action_counts_cache:
            action_counts_cache[decision_col] = suggestions_df[decision_col].value_counts(normalize=True).to_dict()

        # Actor-Specific Metrics: Calculate only once per actor
        actor_metrics_calculator = ActorMetrics(suggestions_df, decision_col, true_outcome_col)
        metrics[actor].update({
            'Total Profit': actor_metrics_calculator.compute_total_profit(),
            'Unexploited Profit': actor_metrics_calculator.compute_unexploited_profit()
        })

        # Standard Metrics
        standard_metrics_calculator = StandardMetrics(suggestions_df, decision_col, true_outcome_col)
        standard_metrics = standard_metrics_calculator.compute_all_metrics()
        metrics[actor].update(standard_metrics)

        # Fairness Metrics: Retrieve from cache and update per actor
        fairness_metrics = fairness_cache[decision_col]
        demographic_parity = fairness_metrics['Demographic Parity']
        equal_opportunity = fairness_metrics['Equal Opportunity']
        equalized_odds = fairness_metrics['Equalized Odds']
        calibration = fairness_metrics['Calibration']

        # Update metrics with fairness-related entries
        metrics[actor].update({
            'Demographic Parity_Grant': demographic_parity['Grant Parity'],
            'Demographic Parity_Grant lower': demographic_parity['Grant lower Parity'],
            'Demographic Parity_Positive Action': demographic_parity['Positive Action Parity'],
            'Demographic Parity_Worst': self._compute_max_min_fairness(
                demographic_parity['Grant Parity'],
                demographic_parity['Grant lower Parity'],
                demographic_parity['Positive Action Parity']
            ),
            'Equal Opportunity_Fully Repaid': equal_opportunity['TPR Fully Repaid Parity'],
            'Equal Opportunity_Partially Repaid': equal_opportunity['TPR Partially Repaid Parity'],
            'Equal Opportunity_Worst': self._compute_max_min_fairness(
                equal_opportunity['TPR Fully Repaid Parity'],
                equal_opportunity['TPR Partially Repaid Parity']
            ),
            'Equalized Odds_Fully Repaid': equalized_odds['Equalized Odds Fully Repaid'],
            'Equalized Odds_Partially Repaid': equalized_odds['Equalized Odds Partially Repaid'],
            'Equalized Odds_Worst': self.compute_max_min_fairness(
                equalized_odds['Equalized Odds Fully Repaid'],
                equalized_odds['Equalized Odds Partially Repaid']
            ),
            'Calibration_Fully Repaid': calibration['Grant Calibration (Fully Repaid)'],
            'Calibration_Partially Repaid': calibration['Grant lower Calibration (Partially Repaid)'],
            'Calibration_Worst': self._compute_max_min_fairness(
                calibration['Grant Calibration (Fully Repaid)'],
                calibration['Grant lower Calibration (Partially Repaid)']
            )
        })

        # Action Percentages: Retrieve from cache and update per actor
        action_counts = action_counts_cache[decision_col]
        metrics[actor].update({
            f'Percent_{action}': action_counts.get(action, 0) for action in actions_set
        })

    return metrics
