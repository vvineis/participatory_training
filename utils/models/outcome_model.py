from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor
import numpy as np

class OutcomeModel:
    def __init__(self, classifier, use_smote=True, smote_k_neighbors=1, smote_random_state=42,  model_random_state=111):
        self.classifier = classifier
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_random_state = smote_random_state
        self.model_random_state= model_random_state

    def train(self, X_train, y_train, **hyperparams):
        if self.use_smote:
            # Apply SMOTE to balance the classes in the training set
            smote = SMOTE(k_neighbors=self.smote_k_neighbors, random_state=self.smote_random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train the classifier with the (resampled) training set
        if 'random_state' in self.classifier.get_params():
            hyperparams['random_state'] = self.model_random_state
        
        self.classifier.set_params(**hyperparams)
        self.classifier.fit(X_train_resampled, y_train_resampled)

        return self.classifier

    def evaluate(self, X_val, y_val):
        y_pred_val = self.classifier.predict(X_val)
        
        # Calculate accuracy for the validation set
        accuracy = accuracy_score(y_val, y_pred_val)
        print(f"Outcome Prediction Accuracy (Validation Set): {accuracy * 100:.2f}%")
        
        return accuracy


class CausalOutcomeModel:
    def __init__(self, learner=None, control_name='C', random_state=111):
        """
        Initialize the Causal Outcome Model with an X-Learner.
        :param learner: Base learner (e.g., XGBRegressor, RandomForestRegressor).
        :param control_name: The name of the control group in the treatment column.
        :param random_state: Random state for reproducibility.
        """
        self.learner = learner or XGBRegressor(random_state=random_state)
        self.control_name = control_name
        self.random_state = random_state
        self.model = BaseXRegressor(learner=self.learner, control_name=self.control_name)

    def train(self, X_train, treatment_train, y_train, **hyperparams):
        print("Unique values in treatment column:", treatment_train.unique())

        if 'random_state' in self.learner.get_params():
            hyperparams['random_state'] = self.random_state
        self.learner.set_params(**hyperparams)

        model=self.model.fit(X=X_train, treatment=treatment_train, y=y_train)

        return model

    def predict(self, X_test, treatment='A'):
        return self.model.predict(X=X_test, treatment=treatment)

    def predict_outcomes(self, X_test):
        predicted_outcomes_A = []
        predicted_outcomes_C = []

        for i in range(len(X_test)):
            # Predict baseline outcome for control group 'C'
            patient = X_test.iloc[i:i + 1]
            predicted_outcome_C = self.model.models_mu_c['A'].predict(patient)
            predicted_outcome_C = np.round(predicted_outcome_C, 0)
            predicted_outcomes_C.append(predicted_outcome_C)

            # Predict treatment effect (CATE) and calculate absolute outcome for 'A'
            cate = self.model.predict(patient, treatment='A')
            predicted_outcome_A = np.round(predicted_outcome_C + cate, 0)
            predicted_outcomes_A.append(predicted_outcome_A)

        return np.array(predicted_outcomes_A), np.array(predicted_outcomes_C)

    def evaluate(self, X_test, y_test, treatment_test):
        predicted_outcomes_A, predicted_outcomes_C = self.predict_outcomes(X_test)

        actual_outcomes = y_test.values if hasattr(y_test, 'values') else y_test
        actual_treatments = treatment_test.values if hasattr(treatment_test, 'values') else treatment_test

        # Compare predicted vs actual for each treatment
        treated_indices = (actual_treatments == 'A')
        control_indices = (actual_treatments == 'C')

        mse_treated = np.mean((predicted_outcomes_A[treated_indices] - actual_outcomes[treated_indices]) ** 2)
        mse_control = np.mean((predicted_outcomes_C[control_indices] - actual_outcomes[control_indices]) ** 2)

        print(f"Mean Squared Error (Treated): {mse_treated:.4f}")
        print(f"Mean Squared Error (Control): {mse_control:.4f}")

        return {'mse_treated': mse_treated, 'mse_control': mse_control}



