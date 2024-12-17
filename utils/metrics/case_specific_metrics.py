class LendingCaseMetrics:
    def __init__(self, suggestions_df, decision_col, true_outcome_col):
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
