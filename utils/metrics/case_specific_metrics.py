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

    def compute_number_treated(self):
        # Ensure the decision column is valid
        print(self.suggestions_df.columns)
        if self.decision_col not in self.suggestions_df.columns:
            raise ValueError(f"Decision column {self.decision_col} not found in the DataFrame")
        
        
        # Compute total cost
        total_treated  = 0
        for decision, group in self.suggestions_df.groupby(self.decision_col):
            if decision == 'A':  # Adjust this condition based on your definition of treated
                total_treated += len(group)
        
        return total_treated
    
    def compute_mean_outcome(self):
        """
        Compute the total outcome separately for treated and control groups.
        :return: A tuple containing total outcome for treated and control groups.
        """

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
    
    def compute_all_metrics(self):
        """
        Compute and return all relevant metrics.
        :return: A dictionary containing all computed metrics.
        """
        total_treated = self.compute_number_treated()
        mean_outcome_treated, mean_outcome_control = self.compute_mean_outcome()

        return {
            'Total_treated': total_treated,
            'Mean_outcome_treated': mean_outcome_treated,
            'Mean_outcome_control': mean_outcome_control,
        }
    
    def get_metrics(self, case_metrics_list):
        """
        Retrieve selected metrics from the available metrics.
        :param case_metrics_list: List of metric names to compute and return.
        :return: Dictionary containing the selected metrics and their values.
        """
        # Define available metrics
        available_metrics = {
            'Total_treated': self.compute_number_treated,
            'Mean_outcome_treated': lambda: self.compute_mean_outcome()[0],  # Treated mean outcome
            'Mean_outcome_control': lambda: self.compute_mean_outcome()[1],  # Control mean outcome
        }

        # Compute selected metrics
        selected_metrics = {}
        for metric in case_metrics_list:
            if metric in available_metrics:
                selected_metrics[metric] = available_metrics[metric]()
            else:
                raise ValueError(f"Metric '{metric}' is not available. Choose from {list(available_metrics.keys())}.")

        return selected_metrics

    

