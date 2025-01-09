import numpy as np
class RealPayoffMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, true_outcome_col, reward_actor, reward_structures):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
               
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col
        self.reward_structures = reward_structures
        self.positive_attribute_for_fairness = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.actor = reward_actor

    def calculate_real_payoff(self, row):
        
        suggested_action = row[self.decision_col]
        true_outcome = row[self.true_outcome_col]
        applicant_type = int(row[self.positive_attribute_for_fairness])

        rewards_for_type = self.reward_structures[applicant_type]
        if self.actor not in rewards_for_type:
            raise KeyError(f"Actor {self.actor} not found in reward structures for applicant type {applicant_type}")
        
        reward = rewards_for_type[self.actor].get((suggested_action, true_outcome), 0.0)
        
        return reward

    def compute_total_real_payoff(self):
        total_payoff = self.suggestions_df.apply(self.calculate_real_payoff, axis=1).sum()
        return total_payoff
