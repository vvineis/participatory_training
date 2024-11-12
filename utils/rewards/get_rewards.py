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

    def __init__(self, reward_types, noise_level=0.05):
        self.noise_level = noise_level
        self.reward_types= reward_types
        self.reward_structures = RewardCalculator.REWARD_STRUCTURES 

    def get_rewards(self, action, outcome, applicant_type, loan_amount, interest_rate):
        # Retrieve reward structure based on applicant type
        reward_structure = self.REWARD_STRUCTURES[applicant_type]
        
        # Dynamically retrieve base rewards from the reward structure
        rewards = {
            reward_type: reward_structure[reward_type][(action, outcome)]
            for reward_type in self.reward_types
        }
        
        # Apply adjustments based on loan amount and interest rate
        adjusted_rewards = self.adjust_rewards(rewards, loan_amount, interest_rate)
        
        # Return rewards as a list in the same order as self.reward_types
        return [adjusted_rewards[reward_type] for reward_type in self.reward_types]


    def adjust_rewards(self, rewards, loan_amount, interest_rate):
        # Factors based on loan amount and interest rate
        loan_amount_factor = np.clip(loan_amount / 10000, 0.5, 1.5)
        interest_rate_factor = np.clip(interest_rate, 0.05, 0.25)

        # Adjust each reward type dynamically
        adjusted_rewards = {}
        for reward_type, reward_value in rewards.items():
            # Custom adjustments for each reward type (if needed)
            if reward_type == 'Bank':
                adjusted_reward = reward_value * loan_amount_factor * (1 + interest_rate_factor)
            elif reward_type == 'Applicant':
                adjusted_reward = reward_value * (2 - interest_rate_factor) * (1 - loan_amount_factor)
            elif reward_type == 'Regulatory':
                adjusted_reward = reward_value * (1 - interest_rate_factor) * loan_amount_factor
            else:
                # General adjustment for any additional reward types
                adjusted_reward = reward_value * loan_amount_factor * (1 + interest_rate_factor / 2)

            # Apply noise and clip
            adjusted_reward += np.random.uniform(-self.noise_level, self.noise_level)
            adjusted_rewards[reward_type] = np.clip(adjusted_reward, 0, 1)

        return adjusted_rewards


    def compute_rewards(self, df):
        # Apply get_rewards across DataFrame with lambda for vectorization
        rewards = df.apply(lambda row: self.get_rewards(
            row['Action'], row['Outcome'], row['Applicant Type'], row['Loan Amount'], row['Interest Rate']), axis=1
        )
        
        # Split rewards tuple into separate columns and assign them
        df[['Bank_reward', 'Applicant_reward', 'Regulatory_reward']] = pd.DataFrame(rewards.tolist(), index=df.index)
        return df
