
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_reward_models(X_train, y_train_bank, y_train_applicant, y_train_regulatory,
                        X_val=None, y_val_bank=None, y_val_applicant=None, y_val_regulatory=None,
                        **hyperparams):
    
    print(X_train.head())
    """
    Train the reward models (bank, applicant, regulatory) with specified hyperparameters.
    If no hyperparameters are provided, default values are used.
    """
    # Convert all column names to strings (if necessary)
    X_train.columns = X_train.columns.astype(str)
    if X_val is not None:
        X_val.columns = X_val.columns.astype(str)

    # Initialize models with the provided hyperparameters
    bank_regressor = RandomForestRegressor(**hyperparams)
    applicant_regressor = RandomForestRegressor(**hyperparams)
    regulatory_regressor = RandomForestRegressor(**hyperparams)

    # Fit models
    bank_regressor.fit(X_train, y_train_bank)
    applicant_regressor.fit(X_train, y_train_applicant)
    regulatory_regressor.fit(X_train, y_train_regulatory)

    return bank_regressor, applicant_regressor, regulatory_regressor


def evaluate_reward_models(bank_regressor, applicant_regressor, regulatory_regressor,
                           X_val, y_val_bank, y_val_applicant, y_val_regulatory):

    X_val.columns = X_val.columns.astype(str)

    y_pred_val_bank = bank_regressor.predict(X_val)
    y_pred_val_applicant = applicant_regressor.predict(X_val)
    y_pred_val_regulatory = regulatory_regressor.predict(X_val)


    mse_bank = mean_squared_error(y_val_bank, y_pred_val_bank)
    mse_applicant = mean_squared_error(y_val_applicant, y_pred_val_applicant)
    mse_regulatory = mean_squared_error(y_val_regulatory, y_pred_val_regulatory)

    print(f"Bank Reward Prediction MSE (Validation Set): {mse_bank:.4f}")
    print(f"Applicant Reward Prediction MSE (Validation Set): {mse_applicant:.4f}")
    print(f"Regulatory Reward Prediction MSE (Validation Set): {mse_regulatory:.4f}")

    return mse_bank, mse_applicant, mse_regulatory