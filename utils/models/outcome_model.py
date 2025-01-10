from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor
#from sklearn.linear_model import Ridger
from sklearn.exceptions import ConvergenceWarning

class OutcomeModel:
    def __init__(self, classifier, use_smote=True, smote_k_neighbors=1, smote_random_state=42,  model_random_state=42):
        print(f"Initializing OutcomeModel with classifier: {classifier.__class__.__name__}")
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
    def __init__(self, learner=None, control_name='C', random_state=73):
        if learner is None:
            raise ValueError("Learner cannot be None for CausalOutcomeModel.")
        # Ensure learner is a class to instantiate dynamically
        self.learner_class = learner if isinstance(learner, type) else learner.__class__
        self.learner_instance = None  # Will be initialized in train

        self.control_name = control_name
        self.random_state = random_state
        self.model = None

    def train(self, X_train, treatment_train, y_train, **hyperparams): 
        warnings.filterwarnings("ignore", category=ConvergenceWarning)   
        # Create a fresh learner and model
        # Dynamically initialize the learner instance
        self.learner_instance = self.learner_class(random_state=self.random_state, **hyperparams)
        if isinstance(self.learner_instance, XGBRegressor):
            print("Using XGBRegressor as the learner.")
        else:
            print(f"Warning: The learner is not XGBRegressor. It is {type(self.learner_instance)}.")
        
        self.model = BaseXRegressor(learner=self.learner_instance, control_name=self.control_name)
        self.model.fit(X=X_train, treatment=treatment_train, y=y_train)
        return self.model


    def predict(self, X_test, treatment='A'):
        return self.model.predict(X=X_test, treatment=treatment)

    def predict_outcomes(self, X_test):
        predicted_outcomes_A = []
        predicted_outcomes_C = []

        for i in range(len(X_test)):
            # Predict baseline outcome for control group 'C'
            patient = X_test.iloc[i:i + 1]
            predicted_outcome_C = self.model.models_mu_c['A'].predict(patient)
            predicted_outcome_C = np.round(predicted_outcome_C, 1)
            predicted_outcomes_C.append(predicted_outcome_C)

            # Predict treatment effect (CATE) and calculate absolute outcome for 'A'
            cate = self.model.predict(patient, treatment='A')
            predicted_outcome_A = np.round(predicted_outcome_C + cate, 1)
            predicted_outcomes_A.append(predicted_outcome_A)

        #print(f'predicted_outcomes_C:{predicted_outcomes_C[0]}', f'predicted_outcomes_A:{predicted_outcomes_A[0]}')

        return np.array(predicted_outcomes_A), np.array(predicted_outcomes_C)

    def evaluate(self, X_test, treatment_test, y_test):
        # Predict outcomes
        predicted_outcomes_A, predicted_outcomes_C = self.predict_outcomes(X_test)
        #print(f'Evaluation phase:\npredicted_outcomes_C:{predicted_outcomes_C[0]}', f'predicted_outcomes_A:{predicted_outcomes_A[0]}')

        # Convert to numpy for indexing
        actual_outcomes = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        actual_treatments = treatment_test.values if hasattr(treatment_test, 'values') else np.array(treatment_test)
        # Identify treated and control indices
        treated_indices = (actual_treatments == 'A')
        control_indices = (actual_treatments == 'C')

        # Initialize MAE values
        mae_treated, mae_control = None, None

        # Calculate MAE for treated group
        if np.any(treated_indices):
            mae_treated = np.mean(np.abs(predicted_outcomes_A[treated_indices] - actual_outcomes[treated_indices]))
            print(f"Mean Absolute Error (Treated): {mae_treated:.4f}")
        else:
            print("Warning: No treated samples ('A') found in the test data.")
        #print sample of predicted and true outcomes: 
        # Print sample of predicted and true outcomes for the first 10 treated instances
        #print(f"Predicted outcomes: {predicted_outcomes_A[treated_indices]}, Actual outcomes: {actual_outcomes[treated_indices]}")


        # Calculate MAE for control group
        if np.any(control_indices):
            mae_control = np.mean(np.abs(predicted_outcomes_C[control_indices] - actual_outcomes[control_indices]))
            print(f"Mean Absolute Error (Control): {mae_control:.4f}")
        else:
            print("Warning: No control samples ('C') found in the test data.")

        # Compute the average MAE across both groups
        valid_maes = [mae for mae in [mae_treated, mae_control] if mae is not None]
        avg_mae = np.mean(valid_maes) if valid_maes else None

        print(f"Average Mean Absolute Error: {avg_mae:.4f}" if avg_mae is not None else "No valid samples for MAE computation.")
        print(f'mae_treated: {mae_treated}, mae_control: {mae_control}')
        return avg_mae



