import numpy as np
class RealPayoffMetrics:
    def __init__(self, cfg, suggestions_df, decision_col, true_outcome_col, reward_actor, reward_structures):
        # Ensure the columns exist in the DataFrame
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        
        # Use cfg for centralized configuration
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col
        self.reward_structures = reward_structures
        self.positive_attribute_for_fairness = cfg.case_specific_metrics.positive_attribute_for_fairness
        self.actor = reward_actor

    def calculate_real_payoff(self, row):
        # Extract values based on configuration settings
        suggested_action = row[self.decision_col]
        true_outcome = row[self.true_outcome_col]
        applicant_type = int(row[self.positive_attribute_for_fairness])

        # Retrieve the reward for the actor based on action and outcome
        rewards_for_type = self.reward_structures[applicant_type]
        if self.actor not in rewards_for_type:
            raise KeyError(f"Actor {self.actor} not found in reward structures for applicant type {applicant_type}")
        
        # Retrieve the reward for the actor based on action and outcome
        reward = rewards_for_type[self.actor].get((suggested_action, true_outcome), 0.0)
        
        return reward

    def compute_total_real_payoff(self):
        # Sum the rewards calculated for each row
        total_payoff = self.suggestions_df.apply(self.calculate_real_payoff, axis=1).sum()
        return total_payoff
