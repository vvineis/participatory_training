import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from get_rewards import RewardCalculator

class DataProcessor:
    def __init__(self, df, feature_columns, columns_to_display, categorical_columns, test_size=0.2, n_splits=5, random_split=True):
        self.df = df
        self.feature_columns = feature_columns
        self.columns_to_display = columns_to_display
        self.categorical_columns = categorical_columns
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_split = random_split
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
        self._split_data()
        self.reward_calculator = RewardCalculator()


    def _split_data(self):
        # Split data into training and testing sets
        if self.random_split:
            self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=42)
        else:
            split_index = int(len(self.df) * (1 - self.test_size))
            self.train_df = self.df.iloc[:split_index]
            self.test_df = self.df.iloc[split_index:]
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

    def prepare_folds(self):
        # Generate training-validation folds
        for train_index, val_index in self.kf.split(self.train_df):
            train_fold_df = self.train_df.iloc[train_index]
            val_fold_df = self.train_df.iloc[val_index]
            yield self.prepare_for_training(train_fold_df, val_fold_df)

    def prepare_for_training(self, train_df, val_df):
        # Process train and validation data for outcome and reward prediction
        train_df = self.scale_features(train_df)
        val_df = self.scale_features(val_df, fit=False)

        dict_for_training = {
            'train_outcome': self.prepare_for_outcome_prediction(train_df),
            'val_outcome': self.prepare_for_outcome_prediction(val_df),
            'train_reward': self.prepare_for_reward_prediction(self.augment_train_for_reward(train_df)),
            'val_reward': self.prepare_for_reward_prediction(val_df),
            'val_set': val_df[self.columns_to_display].copy(),
            'unscaled_val_set': val_df[self.columns_to_display].copy(),
            'scaler': self.scaler,
            'onehot_encoder': self.onehot_encoder
        }
        return dict_for_training

    def scale_features(self, df, fit=True):
        # Scale features in the DataFrame
        if fit:
            df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        else:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df

    def one_hot_encode(self, train_df, val_df):
        # One-hot encode categorical columns
        self.onehot_encoder.fit(train_df[self.categorical_columns])
        X_train_encoded = pd.DataFrame(self.onehot_encoder.transform(train_df[self.categorical_columns]), index=train_df.index)
        X_val_encoded = pd.DataFrame(self.onehot_encoder.transform(val_df[self.categorical_columns]), index=val_df.index)
        return X_train_encoded, X_val_encoded

    def augment_train_for_reward(self, df):
        # Augment the training set for reward prediction
        augmented_rows = []
        for _, row in df.iterrows():
            augmented_rows.append(row.copy())
            duplicated_row = row.copy()
            duplicated_row['Action'] = 'Not Grant'
            bank_reward, applicant_reward, regulatory_reward =self.reward_calculator.get_rewards(
                duplicated_row['Action'], duplicated_row['Outcome'],
                duplicated_row['Applicant Type'], duplicated_row['Loan Amount'], duplicated_row['Interest Rate']
            )
            duplicated_row['Bank_reward'] = bank_reward
            duplicated_row['Applicant_reward'] = applicant_reward
            duplicated_row['Regulatory_reward'] = regulatory_reward
            augmented_rows.append(duplicated_row)
        return pd.DataFrame(augmented_rows).reset_index(drop=True)

    def prepare_for_outcome_prediction(self, df):
        # Prepare data for outcome prediction
        X = df[self.feature_columns]
        y = df['Outcome']
        return X, y

    def prepare_for_reward_prediction(self, df):
        # Prepare data for reward prediction
        reward_features = ['Income', 'Credit Score', 'Loan Amount', 'Interest Rate', 'Action', 'Outcome']
        X = df[reward_features]
        y_bank = df['Bank_reward']
        y_applicant = df['Applicant_reward']
        y_regulatory = df['Regulatory_reward']
        return X, y_bank, y_applicant, y_regulatory
    
    def process(self):
        # This method returns the train-validation folds generator, the full training set, and test set
        return self.prepare_folds(), self.train_df, self.test_df
