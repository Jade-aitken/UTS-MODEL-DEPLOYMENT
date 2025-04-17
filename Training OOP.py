import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class DataLoader:
    def __init__(self, file_path, encoders):
        self.file_path = file_path
        self.encoders = encoders
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        # benerin value yang salah tadi
        self.data['person_gender'].replace({'Male': 'male', 'fe male': 'female'}, inplace=True)

        

    def apply_encodings(self):
        for col, enc_file in self.encoders.items():
            with open(enc_file, 'rb') as f:
                encoder_dict = pickle.load(f)

            # Ambil dictionary-nya (karena disimpan dalam bentuk {'colname': {dict}})
            if isinstance(encoder_dict, dict) and col in encoder_dict:
                mapping_dict = encoder_dict[col]
            else:
                mapping_dict = encoder_dict

            # Map dan pastikan hasilnya numerik
            self.data[f"{col}_encoded"] = self.data[col].map(mapping_dict).astype(float)
            self.data.drop(columns=[col], inplace=True)


    def create_input_output(self, target_col):
        self.y = self.data[target_col]
        self.X = self.data.drop(columns=[target_col])


class XGBoostTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

         # Handle missing values setelah split (hanya pakai mean dari X_train)
        mean_income = self.X_train['person_income'].mean()
        self.X_train['person_income'].fillna(mean_income, inplace=True)
        self.X_test['person_income'].fillna(mean_income, inplace=True)

    def tuning_parameters(self):
        parameters = {
            'max_depth': [3,4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100,150, 200,300]
        }

        XGB = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        grid_search = GridSearchCV(
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            param_grid=parameters,
            scoring='accuracy',
            cv=5,
            error_score='raise'
        )

        grid_search.fit(self.X_train, self.y_train)

        print("Tuned Hyperparameters:", grid_search.best_params_)
        print("Best Accuracy Score:", grid_search.best_score_)

        # Create model with best params
        self.model = XGBClassifier(
            **grid_search.best_params_,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        matrix = confusion_matrix(self.y_test, predictions)

        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(matrix)

    def save_model(self, filename='best_xgb_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == '__main__':
    encoders = {
        'person_education': 'education_encode.pkl',
        'person_gender': 'gender_encode.pkl',
        'person_home_ownership': 'home_own_encode.pkl',
        'loan_intent': 'loan_intent_encode.pkl',
        'previous_loan_defaults_on_file': 'prev_defaults_encode.pkl'
    }

    data_loader = DataLoader('Dataset_A_loan.csv', encoders)
    data_loader.load_data()
    data_loader.apply_encodings()
    data_loader.create_input_output(target_col='loan_status')

    trainer = XGBoostTrainer(data_loader.X, data_loader.y)
    trainer.split_data()
    trainer.tuning_parameters()
    trainer.evaluate_model()
    trainer.save_model()