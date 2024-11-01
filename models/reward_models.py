

from sklearn.metrics import mean_squared_error


class RewardModels:
    def __init__(self, regressor_class, **regressor_params):

        # Initialize separate models for each reward component
        self.bank_regressor = regressor_class(**regressor_params)
        self.applicant_regressor = regressor_class(**regressor_params)
        self.regulatory_regressor = regressor_class(**regressor_params)

    def train(self, X_train, y_train_bank, y_train_applicant, y_train_regulatory):

        # Ensure feature names are strings
        X_train.columns = X_train.columns.astype(str)

        # Train each reward model
        self.bank_regressor.fit(X_train, y_train_bank)
        self.applicant_regressor.fit(X_train, y_train_applicant)
        self.regulatory_regressor.fit(X_train, y_train_regulatory)

        return self.bank_regressor, self.applicant_regressor, self.regulatory_regressor

    def evaluate(self, X_val, y_val_bank, y_val_applicant, y_val_regulatory):

        # Ensure feature names are strings
        X_val.columns = X_val.columns.astype(str)

        # Predict and calculate MSE for each reward model
        y_pred_val_bank = self.bank_regressor.predict(X_val)
        y_pred_val_applicant = self.applicant_regressor.predict(X_val)
        y_pred_val_regulatory = self.regulatory_regressor.predict(X_val)

        mse_bank = mean_squared_error(y_val_bank, y_pred_val_bank)
        mse_applicant = mean_squared_error(y_val_applicant, y_pred_val_applicant)
        mse_regulatory = mean_squared_error(y_val_regulatory, y_pred_val_regulatory)

       # print(f"Bank Reward Prediction MSE (Validation Set): {mse_bank:.4f}")
       # print(f"Applicant Reward Prediction MSE (Validation Set): {mse_applicant:.4f}")
       # print(f"Regulatory Reward Prediction MSE (Validation Set): {mse_regulatory:.4f}")

        return {
            'bank_mse': mse_bank,
            'applicant_mse': mse_applicant,
            'regulatory_mse': mse_regulatory
        }
