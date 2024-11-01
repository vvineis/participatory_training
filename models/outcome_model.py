from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

class OutcomeModel:
    def __init__(self, classifier, use_smote=True, smote_k_neighbors=1, smote_random_state=42):
        self.classifier = classifier
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_random_state = smote_random_state

    def train(self, X_train, y_train, **hyperparams):
        if self.use_smote:
            # Apply SMOTE to balance the classes in the training set
            smote = SMOTE(k_neighbors=self.smote_k_neighbors, random_state=self.smote_random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train the classifier with the (resampled) training set
        self.classifier.set_params(**hyperparams)
        self.classifier.fit(X_train_resampled, y_train_resampled)

        return self.classifier

    def evaluate(self, X_val, y_val):
        y_pred_val = self.classifier.predict(X_val)
        
        # Calculate accuracy for the validation set
        accuracy = accuracy_score(y_val, y_pred_val)
        #print(f"Outcome Prediction Accuracy (Validation Set): {accuracy * 100:.2f}%")
        
        return accuracy


