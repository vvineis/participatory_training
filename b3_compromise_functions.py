from abc import ABC, abstractmethod
import numpy as np
from math import prod

# Abstract Base Class for Solution Strategies
class SolutionStrategy(ABC):
    @abstractmethod
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        """Compute the suggested action and associated value for a strategy."""
        pass

class MaxIndividualReward(SolutionStrategy): 
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        best_actions = {actor: {'action': max(rewards, key=rewards.get), 'value': max(rewards.values())}
                        for actor, rewards in expected_rewards.items()}
        return best_actions

# Maximin Criterion Strategy
class MaximinCriterion(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        min_rewards = {
            action: min(actor_rewards.get(action, 0) for actor_rewards in expected_rewards.values())
            for action in all_actions
        }
        best_action = max(min_rewards, key=min_rewards.get)
        return best_action, min_rewards[best_action]

# Kalai-Smorodinsky Strategy
class KalaiSmorodinsky(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        gains = {
            action: min(
                (rewards.get(action, disagreement_point[actor]) - disagreement_point[actor]) /
                (ideal_point[actor] - disagreement_point[actor])
                for actor, rewards in expected_rewards.items() if ideal_point[actor] != disagreement_point[actor]
            )
            for action in all_actions
        }
        best_action = max(gains, key=gains.get)
        return best_action, gains[best_action]

# Nash Bargaining Solution Strategy
class NashBargainingSolution(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions):
        products = {
            action: prod(
                max(0, expected_rewards[actor].get(action, disagreement_point[actor]) - disagreement_point[actor])
                for actor in expected_rewards
            )
            for action in all_actions
            if all(expected_rewards[actor].get(action, disagreement_point[actor]) > disagreement_point[actor] for actor in expected_rewards)
        }
        best_action = max(products, key=products.get)
        return best_action, products[best_action]

# Nash Social Welfare Strategy
class NashSocialWelfare(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions, epsilon=1e-6):
        products = {
            action: prod(max(epsilon, actor_rewards.get(action, epsilon)) for actor_rewards in expected_rewards.values())
            for action in all_actions
        }
        best_action = max(products, key=products.get)
        return best_action, products[best_action]

# Compromise Programming Strategy
class CompromiseProgramming(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions, p=2):
        distances = {
            action: sum(
                abs(ideal_point[actor] - expected_rewards[actor].get(action, ideal_point[actor])) ** p
                for actor in expected_rewards
            ) ** (1 / p)
            for action in all_actions
        }
        best_action = min(distances, key=distances.get)
        return best_action, distances[best_action]

# Proportional Fairness Strategy
class ProportionalFairness(SolutionStrategy):
    def compute(self, expected_rewards, disagreement_point, ideal_point, all_actions, epsilon=1e-6):
        sums = {
            action: sum(np.log(max(epsilon, actor_rewards.get(action, epsilon))) for actor_rewards in expected_rewards.values())
            for action in all_actions
        }
        best_action = max(sums, key=sums.get)
        return best_action, sums[best_action]

# SuggestAction Class that uses the Strategies
class SuggestAction:
    def __init__(self, expected_rewards):
        self.expected_rewards = expected_rewards
        # Precompute values used across strategies
        self.disagreement_point = {actor: min(rewards.values()) for actor, rewards in expected_rewards.items()}
        print(f'disagreement point {self.disagreement_point}')
        self.ideal_point = {actor: max(rewards.values()) for actor, rewards in expected_rewards.items()}
        print(f'ideal point {self.ideal_point}')
        self.all_actions = set(action for rewards in expected_rewards.values() for action in rewards)
        
        # Register available strategies
        self.strategies = {
                'Max Individual Reward': MaxIndividualReward(),
                'Maximin': MaximinCriterion(),
                'Kalai-Smorodinsky': KalaiSmorodinsky(),
                'Nash Bargaining': NashBargainingSolution(),
                'Nash Social Welfare': NashSocialWelfare(),
                'Compromise Programming': CompromiseProgramming(),
                'Proportional Fairness': ProportionalFairness(),
            }
        print(self.strategies.keys())

    def compute_all_solutions(self):
        results = {}
        for name, strategy in self.strategies.items():
            action_value = strategy.compute(
                self.expected_rewards, self.disagreement_point, self.ideal_point, self.all_actions
            )
            # Adjust output format for MaxIndividualReward
            if name == 'Max Individual Reward':
                results[name] = action_value  # Store individual best actions directly
            else:
                results[name] = {'action': action_value[0], 'value': action_value[1]}
        return results