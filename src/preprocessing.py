import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.rewards.get_rewards import RewardCalculator
from hydra.utils import instantiate



class DataProcessor:
    def __init__(self, df, cfg, random_split=True):
        self.df = df
        self.feature_columns = cfg.context.feature_columns
        self.columns_to_display = cfg.context.columns_to_display
        self.categorical_columns = cfg.categorical_columns
        self.test_size = cfg.test_size
        self.reward_types= cfg.actors.reward_types
        self.n_splits = cfg.cv_splits
        self.random_split = random_split
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
        self._split_data()
        self.reward_calculator = instantiate(cfg.reward_calculator) 
        self.augmentation_params = cfg.augmentation_for_rewards.get("augmentation_parameters", {})
        

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
        unscaled_val_or_test_df = val_df[self.columns_to_display].copy()
        
    
        #print(f"train_df_shape{train_df.shape}")
        val_or_test_df = self.scale_features(val_df, fit=False)

        # Prepare outcome prediction data
        X_train_outcome, y_train_outcome = self.prepare_for_outcome_prediction(train_df)
        X_val_or_test_outcome, y_val_or_test_outcome = val_or_test_df[self.feature_columns], val_or_test_df['Outcome']

        # Prepare reward prediction data
        augmented_train_df = self.augment_train_for_reward(train_df)

        X_train_reward, y_train_rewards = self.prepare_for_reward_prediction(augmented_train_df)

        X_train_encoded, X_val_encoded = self.one_hot_encode(X_train_reward, val_or_test_df)

        # Combine features and encoded data for train and validation sets
        X_train_reward_combined = pd.concat(
            [X_train_reward[self.feature_columns].reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1
        )
        X_val_or_test_reward_combined = pd.concat(
            [val_or_test_df[self.feature_columns].reset_index(drop=True), X_val_encoded.reset_index(drop=True)], axis=1
        )

        # Dynamically retrieve validation rewards
        y_val_or_test_rewards = {
            reward_type: val_or_test_df[f'{reward_type}_reward']
            for reward_type in self.reward_types
        }

        # Dictionary to store training and validation data for each reward type
        dict_for_training = {
            'train_outcome': (X_train_outcome, y_train_outcome),
            'val_or_test_outcome': (X_val_or_test_outcome, y_val_or_test_outcome),
            'train_reward': (X_train_reward_combined, y_train_rewards), #*y_train_rewards.values()),
            'val_or_test_reward': (X_val_or_test_reward_combined, y_val_or_test_rewards), #*y_val_or_test_rewards.values()),
            'val_or_test_set': val_or_test_df,
            'unscaled_val_or_test_set': unscaled_val_or_test_df,
            'scaler': self.scaler,
            'onehot_encoder': self.onehot_encoder
        }

        return dict_for_training

    def scale_features(self, df, fit=True):
        if fit:
            df.loc[:, self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        else:
            df.loc[:, self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df

    def one_hot_encode(self, train_df, val_df):
        # Fit OneHotEncoder on the training data
        self.onehot_encoder.fit(train_df[self.categorical_columns])

        # Transform the train and validation sets
        X_train_encoded = self.onehot_encoder.transform(train_df[self.categorical_columns])
        X_val_encoded = self.onehot_encoder.transform(val_df[self.categorical_columns])

        # Retrieve descriptive column names from the encoder
        encoded_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_columns)

        # Convert encoded arrays to DataFrames with descriptive column names
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=train_df.index)
        X_val_encoded = pd.DataFrame(X_val_encoded, columns=encoded_feature_names, index=val_df.index)

        return X_train_encoded, X_val_encoded
    
    def augment_train_for_reward(self, df):
        """
        Augment training data for reward calculation with a unified strategy.
        Each row is augmented with all possible actions from 'actions_set',
        excluding the existing action in the row.
        """
        augmented_rows = []

        # Retrieve all possible actions from the configuration
        possible_actions = self.augmentation_params.get("actions_set", ["A", "C"])  # Default actions if not provided

        for _, row in df.iterrows():
            # Include the original row
            augmented_rows.append(row.copy())

            # Generate rows for all actions except the current action
            current_action = row['Action']
            actions_to_add = [action for action in possible_actions if action != current_action]

            for action in actions_to_add:
                duplicated_row = row.copy()
                duplicated_row['Action'] = action  # Replace with the new action

                # Dynamically extract additional arguments (if defined in the config)
                additional_arguments = [
                    duplicated_row[arg] for arg in self.augmentation_params.get("additional_arguments", [])
                ]

                # Calculate rewards for the current action
                rewards = self.reward_calculator.get_rewards(
                    duplicated_row['Action'], duplicated_row['Outcome'], *additional_arguments
                )

                # Add reward columns dynamically
                for reward_type, reward_value in zip(self.reward_types, rewards):
                    duplicated_row[f'{reward_type}_reward'] = reward_value

                # Append the augmented row
                augmented_rows.append(duplicated_row)

        # Return the augmented DataFrame
        return pd.DataFrame(augmented_rows).reset_index(drop=True)


    def prepare_for_outcome_prediction(self, df):
        X = df[self.feature_columns]
        y = df['Outcome']

        return X, y

    def prepare_for_reward_prediction(self, df):
        reward_features = self.feature_columns + self.categorical_columns #['Income', 'Credit Score', 'Loan Amount', 'Interest Rate', 'Action', 'Outcome']
        X = df[reward_features]
        
        # Dynamically retrieve reward columns based on self.reward_types
        y_rewards = {reward_type: df[f'{reward_type}_reward'] for reward_type in self.reward_types}
        
        return X, y_rewards
    
    def process(self):
        return self.prepare_folds(), self.train_df, self.test_df
    
