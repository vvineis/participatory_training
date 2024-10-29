from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_outcome_model(X_train, y_train, use_smote=True, **hyperparams):

    if use_smote:
        # Apply SMOTE to balance the classes in the training set
        smote = SMOTE(k_neighbors=1, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        #print(f"Before SMOTE: {y_train.value_counts()}")
        #print(f"After SMOTE: {pd.Series(y_train_resampled).value_counts()}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Train the classifier with the (resampled) training set
    clf = RandomForestClassifier(**hyperparams)
    clf.fit(X_train_resampled, y_train_resampled)

    return clf

def evaluate_outcome_model(clf, X_val, y_val):
    y_pred_val = clf.predict(X_val)

    # Calculate accuracy for the validation set
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Outcome Prediction Accuracy (Validation Set): {accuracy * 100:.2f}%")

    return accuracy


