import numpy as np
import pandas as pd

class RewardCalculator:
    # Define base reward structures
    '''REWARD_STRUCTURES = {
        0: {  # Non-vulnerable applicants
            'Bank': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0,
                ('Not_Grant', 'Fully_Repaid'): 0.2,
                ('Not_Grant', 'Partially_Repaid'): 0.5,
                ('Not_Grant', 'Not_Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.3,
                ('Grant_lower', 'Fully_Repaid'): 0.7,
                ('Grant_lower', 'Partially_Repaid'): 0.8,
                ('Grant_lower', 'Not_Repaid'): 0.4,
                ('Not_Grant', 'Fully_Repaid'): 0.2,
                ('Not_Grant', 'Partially_Repaid'): 0.5,
                ('Not_Grant', 'Not_Repaid'): 0.7
            },
            'Regulatory': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.2,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.1,
                ('Not_Grant', 'Fully_Repaid'): 0.5,
                ('Not_Grant', 'Partially_Repaid'): 0.7,
                ('Not_Grant', 'Not_Repaid'): 1.0
            }
        },
        1: {  # Vulnerable applicants
            'Bank': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0,
                ('Not_Grant', 'Fully_Repaid'): 0.0,
                ('Not_Grant', 'Partially_Repaid'): 0.2,
                ('Not_Grant', 'Not_Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.7,
                ('Grant', 'Not_Repaid'): 0.5,
                ('Grant_lower', 'Fully_Repaid'): 0.5,
                ('Grant_lower', 'Partially_Repaid'): 0.8,
                ('Grant_lower', 'Not_Repaid'): 0.3,
                ('Not_Grant', 'Fully_Repaid'): 0.0,
                ('Not_Grant', 'Partially_Repaid'): 0.2,
                ('Not_Grant', 'Not_Repaid'): 0.6
            },
            'Regulatory': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.3,
                ('Grant_lower', 'Fully_Repaid'): 0.7,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.2,
                ('Not_Grant', 'Fully_Repaid'): 0.3,
                ('Not_Grant', 'Partially_Repaid'): 0.5,
                ('Not_Grant', 'Not_Repaid'): 0.8
            }
        }
    }'''

    '''REWARD_STRUCTURES = {
        0: {  # Non-vulnerable applicants
            'Bank': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.2,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.4,
                ('Grant_lower', 'Partially_Repaid'): 0.5,
                ('Grant_lower', 'Not_Repaid'): 0,
                ('Not_Grant', 'Fully_Repaid'): 0.2,
                ('Not_Grant', 'Partially_Repaid'): 0.5,
                ('Not_Grant', 'Not_Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.7,
                ('Grant', 'Not_Repaid'): 0.5,
                ('Grant_lower', 'Fully_Repaid'): 0.7,
                ('Grant_lower', 'Partially_Repaid'): 0.8,
                ('Grant_lower', 'Not_Repaid'): 0.6,
                ('Not_Grant', 'Fully_Repaid'): 0,
                ('Not_Grant', 'Partially_Repaid'): 0,
                ('Not_Grant', 'Not_Repaid'): 0
            },
            'Regulatory': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.2,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.1,
                ('Not_Grant', 'Fully_Repaid'): 0.5,
                ('Not_Grant', 'Partially_Repaid'): 0.7,
                ('Not_Grant', 'Not_Repaid'): 1.0
            }
        },
        1: {  # Vulnerable applicants
            'Bank': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.5,
                ('Grant_lower', 'Partially_Repaid'): 0.5,
                ('Grant_lower', 'Not_Repaid'): 0,
                ('Not_Grant', 'Fully_Repaid'): 0.0,
                ('Not_Grant', 'Partially_Repaid'): 0.2,
                ('Not_Grant', 'Not_Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.8,
                ('Grant', 'Not_Repaid'): 0.7,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.5,
                ('Not_Grant', 'Fully_Repaid'): 0.0,
                ('Not_Grant', 'Partially_Repaid'): 0,
                ('Not_Grant', 'Not_Repaid'): 0.2
            },
            'Regulatory': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.3,
                ('Grant_lower', 'Fully_Repaid'): 0.7,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.2,
                ('Not_Grant', 'Fully_Repaid'): 0.3,
                ('Not_Grant', 'Partially_Repaid'): 0.5,
                ('Not_Grant', 'Not_Repaid'): 0.8
            }
        }
    }'''

    REWARD_STRUCTURES = {
        0: {  # Non-vulnerable applicants
            'Bank': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 0,
                ('Grant_lower', 'Not_Repaid'): 0,
                ('Not_Grant', 'Fully_Repaid'): 0.5,
                ('Not_Grant', 'Partially_Repaid'): 1,
                ('Not_Grant', 'Not_Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.8,
                ('Grant', 'Not_Repaid'): 0.7,
                ('Grant_lower', 'Fully_Repaid'): 0.7,
                ('Grant_lower', 'Partially_Repaid'): 0.8,
                ('Grant_lower', 'Not_Repaid'): 0.6,
                ('Not_Grant', 'Fully_Repaid'): 0,
                ('Not_Grant', 'Partially_Repaid'): 0,
                ('Not_Grant', 'Not_Repaid'): 0
            },
            'Regulatory': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.2,
                ('Grant', 'Not_Repaid'): 0.0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.1,
                ('Not_Grant', 'Fully_Repaid'): 0.5,
                ('Not_Grant', 'Partially_Repaid'): 0.7,
                ('Not_Grant', 'Not_Repaid'): 1.0
            }
        },
        1: {  # Vulnerable applicants
            'Bank': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0,
                ('Grant', 'Not_Repaid'): 0,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 0,
                ('Grant_lower', 'Not_Repaid'): 0,
                ('Not_Grant', 'Fully_Repaid'): 0.5,
                ('Not_Grant', 'Partially_Repaid'): 1,
                ('Not_Grant', 'Not_Repaid'): 1.0
            },
            'Applicant': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.8,
                ('Grant', 'Not_Repaid'): 0.7,
                ('Grant_lower', 'Fully_Repaid'): 0.8,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.6,
                ('Not_Grant', 'Fully_Repaid'): 0.0,
                ('Not_Grant', 'Partially_Repaid'): 0,
                ('Not_Grant', 'Not_Repaid'): 0
            },
            'Regulatory': {
                ('Grant', 'Fully_Repaid'): 1.0,
                ('Grant', 'Partially_Repaid'): 0.5,
                ('Grant', 'Not_Repaid'): 0.3,
                ('Grant_lower', 'Fully_Repaid'): 0.7,
                ('Grant_lower', 'Partially_Repaid'): 1,
                ('Grant_lower', 'Not_Repaid'): 0.2,
                ('Not_Grant', 'Fully_Repaid'): 0.3,
                ('Not_Grant', 'Partially_Repaid'): 0.5,
                ('Not_Grant', 'Not_Repaid'): 0.8
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
            row['Action'], row['Outcome'], row['Applicant_Type'], row['Loan_Amount'], row['Interest_Rate']), axis=1
        )
        
        # Split rewards tuple into separate columns and assign them
        df[['Bank_reward', 'Applicant_reward', 'Regulatory_reward']] = pd.DataFrame(rewards.tolist(), index=df.index)
        return df
    
class HealthRewardCalculator:
    def __init__(self, alpha=0.7, beta=0.5, gamma=0.6, noise_level=0.05, fixed_cost=100, base_cost=None, reward_types=None):
        """
        Initialize the reward calculator.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_level = noise_level
        self.fixed_cost = fixed_cost
        self.base_cost = base_cost if base_cost else {'A': 1, 'C': 0}
        self.reward_types = reward_types if reward_types else ["Healthcare_Provider", "Policy_Maker", "Parent"]
        self.min_outcome = None
        self.max_outcome = None
        self.min_outcome_action = {}
        self.max_outcome_action = {}

    def compute_rewards(self, df):
        """
        Compute rewards for all stakeholders and add them to the DataFrame.
        """
        # Check for required columns
        required_cols = ['Action', 'Outcome', 'x23']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain the following columns: {required_cols}")

        # Convert Outcome to float
        df['Outcome'] = df['Outcome'].astype(float)

        # Compute min and max outcomes by Action
        self.min_outcome_action = df.groupby('Action')['Outcome'].min().to_dict()
        self.max_outcome_action = df.groupby('Action')['Outcome'].max().to_dict()

        # Dynamically compute rewards based on reward_types
        for reward_type in self.reward_types:
            if reward_type == "Healthcare_Provider":
                df[f"{reward_type}_reward"] = df.apply(
                    lambda row: self._healthcare_provider_reward(row['Outcome'], row['Action']), axis=1
                )
            elif reward_type == "Policy_Maker":
                df[f"{reward_type}_reward"] = df.apply(
                    lambda row: self._policy_maker_reward(row['Outcome'], row['x23'], row['Action']), axis=1
                )
            elif reward_type == "Parent":
                df[f"{reward_type}_reward"] = df['Outcome'].apply(self._parent_reward)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

        return df

    
    def get_rewards(self, action, outcome, x23):
        """
        Dynamically compute rewards for a given action, outcome, and group.
        :param action: Treatment action (e.g., 0 or 1).
        :param outcome: Observed outcome for the individual.
        :param x23: Demographic group identifier for the individual.
        :return: List of rewards for the specified reward_types.
        """
        # Use fixed cost since cost is constant in this context
        cost = self.fixed_cost

        # Calculate rewards for each reward type
        rewards = {}
        for reward_type in self.reward_types:
            if reward_type == "Parent":
                rewards[reward_type] = self._parent_reward(outcome)
            elif reward_type == "Policy_Maker":
                rewards[reward_type] = self._policy_maker_reward(outcome, x23,action)
            elif reward_type == "Healthcare_Provider":
                rewards[reward_type] = self._healthcare_provider_reward(outcome, action)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

        # Return rewards in the order of reward_types
        return [rewards[reward_type] for reward_type in self.reward_types]

    def _healthcare_provider_reward(self, outcome, action):
        # Get the baseline outcome 
        baseline_outcome = self.min_outcome_action.get('C', 2)
        # Compute outcome improvement
        outcome_improv = max(0, outcome - baseline_outcome)  # Ensure non-negative improvement

        # Get the cost associated with the action
        cost = self.base_cost.get(action, 0)
  
        # Reward calculation: emphasize improvement over cost
        # Scale improvement based on cost, but give more weight to improvement
        alpha = 0.8  # Weight for valuing improvement vs cost (0.8 gives 80% weight to improvement)
        max_possible_improv = self.max_outcome_action.get('A', 12) - baseline_outcome
        normalized_improv = outcome_improv / (max_possible_improv + 1e-10)  # Normalize improvement

        # Calculate reward
        reward = alpha * normalized_improv + (1 - alpha) * (1 - cost / max(self.base_cost.values()))
        
        # Add random noise for variability
        reward += np.random.uniform(-self.noise_level, self.noise_level)
        
        # Clip reward to [0, 1] range
        return np.clip(reward, 0, 1)

    def _policy_maker_reward(self, outcome, x23, action):
        """
        Reward for Policy-Makers, emphasizing outcome improvements over the minimum.
        """
        # Get min outcome for the specific treatment and category
        min_outcome = self.min_outcome_action.get(action, 0)

        # Normalize improvement above the minimum
        improvement = max(0, outcome - min_outcome)
        max_improvement = self.max_outcome_action.get(action, 1) - min_outcome
        normalized_improvement = improvement / (max_improvement + 1e-10)

        # Apply additional weight for fairness considerations
        demographic_weight = 1.0 + self.beta * (x23 - 0.5)  # Example weight adjustment for x23
        reward = normalized_improvement * demographic_weight
        reward += np.random.uniform(-self.noise_level, self.noise_level)
        return np.clip(reward, 0, 1)

    def _parent_reward(self, outcome):
        """
        Parent's reward as directly proportional to the normalized outcome.
        """
        max_outcome = max(self.max_outcome_action.values(), default=12)
        min_outcome = min(self.min_outcome_action.values(), default=0)
        reward = (outcome - min_outcome) / (max_outcome - min_outcome + 1e-10)
        return np.clip(reward, 0, 1)

    '''def _parent_reward(self, row, mean_outcome_action):
            """
            Updated reward for Parents with relative benefit and premium component.
            :param row: Row of the DataFrame for the individual.
            :param mean_outcome_action: Mean outcomes for each treatment group.
            :return: Normalized reward.
            """
            # Get min and max mean outcomes across groups
            min_mean_outcome = mean_outcome_action.min()
            max_mean_outcome = mean_outcome_action.max()

            # Compute the normalized relative benefit
            relative_benefit = (row['Outcome'] - min_mean_outcome) / (max_mean_outcome - min_mean_outcome + 1e-10)

            # Compute the premium component
            if row['Outcome'] > min_mean_outcome:
                premium = (row['Outcome'] - min_mean_outcome) / (max_mean_outcome - min_mean_outcome + 1e-10)
            else:
                premium = 0

            # Combine the relative benefit and premium
            reward = self.gamma * relative_benefit + (1 - self.gamma) * premium

            # Add noise and clip to [0, 1]
            reward += np.random.uniform(-self.noise_level, self.noise_level)
            return np.clip(reward, 0, 1)'''


'''class HealthRewardCalculator:
    def __init__(self, base_cost, recovery_factor, max_cost, alpha=0.7, noise_level=0.05, reward_types=None):
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
        return self.base_cost[treatment] + recovery_time *200
    
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
        return np.clip(reward, 0, 1)'''

