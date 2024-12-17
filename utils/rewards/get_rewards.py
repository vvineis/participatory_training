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


class HealthRewardCalculator:
    def __init__(self, base_cost, recovery_factor, max_cost, alpha=0.5, noise_level=0.05, reward_types=None):
        """
        Initialize the reward calculator.
        :param base_cost: Base cost per treatment (dict).
        :param recovery_factor: Cost multiplier per week of recovery.
        :param max_cost: Maximum total cost for normalization.
        :param alpha: Weight for prioritizing recovery time in NHA's reward (0 <= alpha <= 1).
        :param noise_level: Random noise to add variability to rewards.
        :param reward_types: List of reward types to compute dynamically.
        """
        self.base_cost = base_cost
        self.recovery_factor = recovery_factor
        self.max_cost = max_cost
        self.alpha = alpha
        self.noise_level = noise_level
        self.reward_types = reward_types if reward_types else []

    def compute_cost(self, treatment, recovery_time):
        """ Compute the total cost based on treatment and recovery time. """
        return self.base_cost[treatment] + recovery_time * self.recovery_factor
    
    def get_rewards(self, action, outcome, *args):
        """
        Dynamically compute rewards for a given action and outcome.
        :param action: Treatment action (e.g., 'A', 'C').
        :param outcome: Recovery time in weeks.
        :param args: Additional arguments if required.
        :return: List of rewards for the specified reward_types.
        """
        # Compute cost based on the given action and outcome
        cost = self.compute_cost(action, outcome)

        # Calculate rewards for each reward type
        rewards = {}
        for reward_type in self.reward_types:
            if reward_type == "Patient":
                rewards[reward_type] = self._patient_reward(outcome)
            elif reward_type == "Hospital_Admin":
                rewards[reward_type] = self._hospital_reward(cost)
            elif reward_type == "NHA":
                rewards[reward_type] = self._nha_reward(outcome, cost)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

        # Return rewards in the order of reward_types
        return [rewards[reward_type] for reward_type in self.reward_types]

    def compute_rewards(self, df):
        """
        Compute rewards for all agents and add them to the DataFrame.
        :param df: DataFrame with columns ['Action', 'Outcome'].
        :return: DataFrame with rewards added dynamically.
        """
        # Check for required columns
        if 'Action' not in df.columns or 'Outcome' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Action' and 'Outcome' columns.")

        # Compute costs
        df['Cost'] = df.apply(lambda row: self.compute_cost(row['Action'], row['Outcome']), axis=1)

        # Dynamically compute rewards based on reward_types
        for reward_type in self.reward_types:
            if reward_type == "Patient":
                df[f"{reward_type}_reward"] = df['Outcome'].apply(self._patient_reward)
            elif reward_type == "Hospital_Admin":
                df[f"{reward_type}_reward"] = df['Cost'].apply(self._hospital_reward)
            elif reward_type == "NHA":
                df[f"{reward_type}_reward"] = df.apply(lambda row: self._nha_reward(row['Outcome'], row['Cost']), axis=1)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

        return df

    def _patient_reward(self, recovery_time):
        """ Patient's reward based on recovery time (faster is better). """
        reward = 1 - (recovery_time - 1) / (12 - 1)
        reward += np.random.uniform(-self.noise_level, self.noise_level)
        return np.clip(reward, 0, 1)

    def _hospital_reward(self, cost):
        """ Hospital's reward based on total cost (lower is better). """
        reward = 1 - cost / self.max_cost
        reward += np.random.uniform(-self.noise_level, self.noise_level)
        return np.clip(reward, 0, 1)

    def _nha_reward(self, recovery_time, cost):
        """ NHA's reward balancing recovery time and cost. """
        recovery_factor = self._patient_reward(recovery_time)
        cost_factor = self._hospital_reward(cost)
        reward = self.alpha * recovery_factor + (1 - self.alpha) * cost_factor
        reward += np.random.uniform(-self.noise_level, self.noise_level)
        return np.clip(reward, 0, 1)


'''if __name__ == "__main__":
    base_cost = {'A': 6000, 'C': 2000}
    recovery_factor = 400
    max_cost = 12000

    # Initialize the reward calculator
    reward_calculator = ContinuousHealthcareRewardCalculator(base_cost, recovery_factor, max_cost, alpha=0.6)

    # Compute rewards
    simulated_data = simulate_patient_data()
    rewarded_data = reward_calculator.compute_rewards(simulated_data)

    print(rewarded_data)'''