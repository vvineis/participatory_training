
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

class RewardModels:
    def __init__(self, regressor_class, reward_types, **regressor_params):
        """
        Initialize reward models dynamically for the specified reward types.

        Args:
            regressor_class: Regressor class to use for models.
            reward_types: List of reward types (e.g., ['Bank', 'Applicant', 'Regulatory']).
            regressor_params: Parameters for the regressor.
        """
        self.reward_types = reward_types
        self.regressors = {
            reward_type: clone(regressor_class).set_params(**regressor_params)
            for reward_type in reward_types
        }

    def train(self, X_train, y_train_rewards):
        # Ensure feature names are strings
        X_train.columns = X_train.columns.astype(str)

        trained_models = {}
        for reward_type, regressor in self.regressors.items():
            regressor.fit(X_train, y_train_rewards[reward_type])
            trained_models[reward_type] = regressor

        return trained_models

    def evaluate(self, X_val, y_val_rewards):
        # Ensure feature names are strings
        X_val.columns = X_val.columns.astype(str)

        mse_scores = {}
        for reward_type, regressor in self.regressors.items():
            y_pred_val = regressor.predict(X_val)
            mse_scores[f"{reward_type}_mse"] = mean_squared_error(y_val_rewards[reward_type], y_pred_val)
        
        print(mse_scores)
    
        return mse_scores
