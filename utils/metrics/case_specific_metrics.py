class LendingCaseMetrics:
    def __init__(self, suggestions_df, decision_col, true_outcome_col, cfg=None):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col

    def compute_total_profit(self):
        granted_fully_repaid = (self.suggestions_df[self.decision_col] == 'Grant') & (self.suggestions_df[self.true_outcome_col] == 'Fully Repaid')
        granted_partially_repaid = (self.suggestions_df[self.decision_col] == 'Grant lower') & (self.suggestions_df[self.true_outcome_col] == 'Partially Repaid')
        
        total_profit_fully_repaid = (self.suggestions_df.loc[granted_fully_repaid, 'Loan Amount'] * 
                                     (self.suggestions_df.loc[granted_fully_repaid, 'Interest Rate']) / 100).sum()
        total_profit_partially_repaid = (self.suggestions_df.loc[granted_partially_repaid, 'Loan Amount'] * 
                                         (self.suggestions_df.loc[granted_partially_repaid, 'Interest Rate']) / 100).sum()
        
        return total_profit_fully_repaid + total_profit_partially_repaid

    def compute_total_loss(self):
        granted_partially_repaid = (self.suggestions_df[self.decision_col] == 'Grant lower') & (self.suggestions_df[self.true_outcome_col] == 'Partially Repaid')
        granted_not_repaid = (self.suggestions_df[self.decision_col] == 'Grant') & (self.suggestions_df[self.true_outcome_col] == 'Not Repaid')
        
        total_loss_partially_repaid = (self.suggestions_df.loc[granted_partially_repaid, 'Loan Amount'] - 
                                       self.suggestions_df.loc[granted_partially_repaid, 'Recoveries']).sum()
        total_loss_not_repaid = self.suggestions_df.loc[granted_not_repaid, 'Loan Amount'].sum()
        
        return total_loss_partially_repaid + total_loss_not_repaid

    def compute_unexploited_profit(self):
        not_granted_fully_repaid = (self.suggestions_df[self.decision_col] == 'Not Grant') & (self.suggestions_df[self.true_outcome_col] == 'Fully Repaid')
        
        unexploited_profit = (self.suggestions_df.loc[not_granted_fully_repaid, 'Loan Amount'] * 
                              self.suggestions_df.loc[not_granted_fully_repaid, 'Interest Rate'] / 100).sum()
        
        return unexploited_profit

    def compute_all_metrics(self):
        return {
            'Total Profit': self.compute_total_profit(),
            'Total Loss': self.compute_total_loss(),
            'Unexploited Profit': self.compute_unexploited_profit()
        }

    def get_metrics(self, case_metrics_list):
        available_metrics = {
            'Total Profit': self.compute_total_profit,
            'Total Loss': self.compute_total_loss,
            'Unexploited Profit': self.compute_unexploited_profit
        }

        selected_metrics = {}
        for metric in case_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")

        return selected_metrics

class HealthCaseMetrics:
    def __init__(self, suggestions_df, decision_col, true_outcome_col, cfg):
        if decision_col not in suggestions_df.columns or true_outcome_col not in suggestions_df.columns:
            raise ValueError(f"Columns {decision_col} or {true_outcome_col} not found in the DataFrame")
        
        self.suggestions_df = suggestions_df
        self.decision_col = decision_col
        self.true_outcome_col = true_outcome_col
        self.cfg = cfg
    
    def compute_percentage_treated(self):
        # Ensure the decision column is valid
        if self.decision_col not in self.suggestions_df.columns:
            raise ValueError(f"Decision column {self.decision_col} not found in the DataFrame")

        # Compute total treated and total samples
        total_samples = len(self.suggestions_df)
        total_treated = len(self.suggestions_df[self.suggestions_df[self.decision_col] == 'A'])

        # Compute percentage of treated
        percentage_treated = (total_treated / total_samples) * 100
        return percentage_treated


    def compute_avg_outcome_difference(self):
        # Ensure the decision and outcome columns are valid
        if self.decision_col not in self.suggestions_df.columns:
            raise ValueError(f"Decision column {self.decision_col} not found in the DataFrame")
        if 'A_outcome' not in self.suggestions_df.columns or 'C_outcome' not in self.suggestions_df.columns:
            raise ValueError("Outcome columns (A_outcome, C_outcome) not found in the DataFrame")

        # Compute mean outcomes for treated and control groups
        treated_outcome = self.suggestions_df[self.suggestions_df[self.decision_col] == 'A']['A_outcome'].mean()
        control_outcome = self.suggestions_df[self.suggestions_df[self.decision_col] == 'C']['C_outcome'].mean()

        # Compute average difference
        avg_difference = treated_outcome - control_outcome
        return avg_difference


    def compute_total_cognitive_score(self):
        # Ensure the outcome columns are valid
        if 'A_outcome' not in self.suggestions_df.columns or 'C_outcome' not in self.suggestions_df.columns:
            raise ValueError("Outcome columns (A_outcome, C_outcome) not found in the DataFrame")

        # Compute total scores
        total_score_treated = self.suggestions_df[self.suggestions_df[self.decision_col] == 'A']['A_outcome'].sum()
        total_score_control = self.suggestions_df[self.suggestions_df[self.decision_col] == 'C']['C_outcome'].sum()

        # Return total combined score
        return total_score_treated + total_score_control
    
    def compute_mean_outcome(self):
        # Initialize totals
        total_outcome_treated = 0.0
        total_outcome_control = 0.0

        # Iterate through groups based on the decision column
        for decision, group in self.suggestions_df.groupby(self.decision_col):
            if decision == 'A':  # Treated group
                total_outcome_treated += group['A_outcome'].mean()
            elif decision == 'C':  # Control group
                total_outcome_control += group['C_outcome'].mean()

        return total_outcome_treated, total_outcome_control
    
    def compute_total_cost_effectiveness(self):
   
        # Ensure the necessary columns are valid
        required_columns = ['A_outcome', 'C_outcome', self.decision_col]
        for col in required_columns:
            if col not in self.suggestions_df.columns:
                raise ValueError(f"Required column {col} not found in the DataFrame.")

        # Initialize cost-effectiveness metrics
        treated_cost_effectiveness = 0.0
        control_cost_effectiveness = 0.0

        # Treatment costs (customize based on your scenario)
        treatment_costs = {'A': self.cfg.reward_calculator.base_cost.get('A', 100), 
                        'C': self.cfg.reward_calculator.base_cost.get('C', 10)}

        # Compute cost-effectiveness for treated group
        treated_group = self.suggestions_df[self.suggestions_df[self.decision_col] == 'A']
        if not treated_group.empty:
            treated_outcome_improve = (
                treated_group['A_outcome'].mean() - self.suggestions_df['C_outcome'].mean()
            ) 
        else:
            treated_outcome_improve=0

        cost_effect= treated_outcome_improve  / (treatment_costs['A'] - treatment_costs['C'])


        return cost_effect

    def compute_all_metrics(self):
        """
        Compute and return all relevant metrics.
        :return: A dictionary containing all computed metrics.
        """
        percentage_treated = self.compute_percentage_treated()
        avg_outcome_difference = self.compute_avg_outcome_difference()
        total_cognitive_score = self.compute_total_cognitive_score()
        mean_outcome_treated, mean_outcome_control = self.compute_mean_outcome()
        cost_effectiveness = self.compute_total_cost_effectiveness()

        return {
            'Percentage_treated': percentage_treated,
            'Avg_outcome_difference': avg_outcome_difference,
            'Total_cognitive_score': total_cognitive_score,
            'Mean_outcome_treated': mean_outcome_treated,
            'Mean_outcome_control': mean_outcome_control,
            'Cost_effectiveness': cost_effectiveness
        }
            
    def get_metrics(self, case_metrics_list):
        """
        Retrieve selected metrics from the available metrics.
        :param case_metrics_list: List of metric names to compute and return.
        :return: Dictionary containing the selected metrics and their values.
        """
        # Define available metrics
        available_metrics = {
            'Percentage_treated': self.compute_percentage_treated,
            'Avg_outcome_difference': self.compute_avg_outcome_difference,
            'Total_cognitive_score': self.compute_total_cognitive_score,
            'Mean_outcome_treated': lambda: self.compute_mean_outcome()[0],  # Treated mean outcome
            'Mean_outcome_control': lambda: self.compute_mean_outcome()[1],  # Control mean outcome
            'Cost_effectiveness': self.compute_total_cost_effectiveness
        }

        # Compute selected metrics
        selected_metrics = {}
        for metric in case_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")

        return selected_metrics

    

