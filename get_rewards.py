import numpy as np
import pandas as pd

class RewardCalculator:
    # Define base reward structures
    REWARD_STRUCTURES = {
        0: {  # Non-vulnerable applicants
            'Bank': {
                ('Grant', 'Fully Repaid'): 1.0,
                ('Grant', 'Partially Repaid'): 0.5,
                ('Grant', 'Not Repaid'): 0.0,
                ('Grant lower', 'Fully Repaid'): 0.8,
                ('Grant lower', 'Partially Repaid'): 1,
                ('Grant lower', 'Not Repaid'): 0,
                ('Not Grant', 'Fully Repaid'): 0.2,
                ('Not Grant', 'Partially Repaid'): 0.5,
                ('Not Grant', 'Not Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully Repaid'): 1.0,
                ('Grant', 'Partially Repaid'): 0.5,
                ('Grant', 'Not Repaid'): 0.3,
                ('Grant lower', 'Fully Repaid'): 0.7,
                ('Grant lower', 'Partially Repaid'): 0.8,
                ('Grant lower', 'Not Repaid'): 0.4,
                ('Not Grant', 'Fully Repaid'): 0.2,
                ('Not Grant', 'Partially Repaid'): 0.5,
                ('Not Grant', 'Not Repaid'): 0.7
            },
            'Regulatory': {
                ('Grant', 'Fully Repaid'): 1.0,
                ('Grant', 'Partially Repaid'): 0.2,
                ('Grant', 'Not Repaid'): 0.0,
                ('Grant lower', 'Fully Repaid'): 0.8,
                ('Grant lower', 'Partially Repaid'): 1,
                ('Grant lower', 'Not Repaid'): 0.1,
                ('Not Grant', 'Fully Repaid'): 0.5,
                ('Not Grant', 'Partially Repaid'): 0.7,
                ('Not Grant', 'Not Repaid'): 1.0
            }
        },
        1: {  # Vulnerable applicants
            'Bank': {
                ('Grant', 'Fully Repaid'): 1.0,
                ('Grant', 'Partially Repaid'): 0.5,
                ('Grant', 'Not Repaid'): 0.0,
                ('Grant lower', 'Fully Repaid'): 0.8,
                ('Grant lower', 'Partially Repaid'): 1,
                ('Grant lower', 'Not Repaid'): 0,
                ('Not Grant', 'Fully Repaid'): 0.0,
                ('Not Grant', 'Partially Repaid'): 0.2,
                ('Not Grant', 'Not Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully Repaid'): 1.0,
                ('Grant', 'Partially Repaid'): 0.7,
                ('Grant', 'Not Repaid'): 0.5,
                ('Grant lower', 'Fully Repaid'): 0.5,
                ('Grant lower', 'Partially Repaid'): 0.8,
                ('Grant lower', 'Not Repaid'): 0.3,
                ('Not Grant', 'Fully Repaid'): 0.0,
                ('Not Grant', 'Partially Repaid'): 0.2,
                ('Not Grant', 'Not Repaid'): 0.6
            },
            'Regulatory': {
                ('Grant', 'Fully Repaid'): 1.0,
                ('Grant', 'Partially Repaid'): 0.5,
                ('Grant', 'Not Repaid'): 0.3,
                ('Grant lower', 'Fully Repaid'): 0.7,
                ('Grant lower', 'Partially Repaid'): 1,
                ('Grant lower', 'Not Repaid'): 0.2,
                ('Not Grant', 'Fully Repaid'): 0.3,
                ('Not Grant', 'Partially Repaid'): 0.5,
                ('Not Grant', 'Not Repaid'): 0.8
            }
        }
    }

    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def get_rewards(self, action, outcome, applicant_type, loan_amount, interest_rate):
        # Retrieve reward structure based on applicant type
        reward_structure = self.REWARD_STRUCTURES[applicant_type]
        
        # Base rewards from reward structure
        bank_reward = reward_structure['Bank'][(action, outcome)]
        applicant_reward = reward_structure['Applicant'][(action, outcome)]
        regulatory_reward = reward_structure['Regulatory'][(action, outcome)]
        
        # Apply adjustments based on loan and interest rate
        bank_reward, applicant_reward, regulatory_reward = self.adjust_rewards(
            bank_reward, applicant_reward, regulatory_reward, loan_amount, interest_rate
        )
        
        return bank_reward, applicant_reward, regulatory_reward

    def adjust_rewards(self, bank_reward, applicant_reward, regulatory_reward, loan_amount, interest_rate):
        loan_amount_factor = np.clip(loan_amount / 10000, 0.5, 1.5)
        interest_rate_factor = np.clip(interest_rate, 0.05, 0.25)
        
        # Adjust based on loan amount and interest rate
        bank_reward *= loan_amount_factor * (1 + interest_rate_factor)
        applicant_reward *= (2 - interest_rate_factor) * (1 - loan_amount_factor)
        regulatory_reward *= (1 - interest_rate_factor) * loan_amount_factor
        
        # Apply noise
        bank_reward += np.random.uniform(-self.noise_level, self.noise_level)
        applicant_reward += np.random.uniform(-self.noise_level, self.noise_level)
        regulatory_reward += np.random.uniform(-self.noise_level, self.noise_level)
        
        # Clip rewards to [0, 1]
        return np.clip(bank_reward, 0, 1), np.clip(applicant_reward, 0, 1), np.clip(regulatory_reward, 0, 1)

    def compute_rewards(self, df):
        # Apply get_rewards across DataFrame with lambda for vectorization
        rewards = df.apply(lambda row: self.get_rewards(
            row['Action'], row['Outcome'], row['Applicant Type'], row['Loan Amount'], row['Interest Rate']), axis=1
        )
        
        # Split rewards tuple into separate columns and assign them
        df[['Bank_reward', 'Applicant_reward', 'Regulatory_reward']] = pd.DataFrame(rewards.tolist(), index=df.index)
        return df
