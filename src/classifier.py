import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.vectorizer import TextVectorizer 
class ReviewClassifier:
    def __init__(self, data_path="data/processed/clean_data.csv"):
        self.data_path = data_path
        self.x_train_text, self.x_test_text = None, None
        self.y_train, self.y_test = None, None
        self.x_train_vec, self.x_test_vec = None, None
        
        self.vectorizer = None
        self.nb_model = None
        self.lr_model = None

    def prepare_data(self, test_size=0.2, random_state=42):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Missing feature data at: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df.dropna(subset=['Cleaned Review', 'Target Label'], inplace=True)
        
        x_raw = df['Cleaned Review'].values.astype('U')
        y_raw = df['Target Label'].values

        self.x_train_text, self.x_test_text, self.y_train, self.y_test = train_test_split(
            x_raw, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
        )
        return self.x_train_text, self.x_test_text, self.y_train, self.y_test

    def extract_features(self, max_features=5000):
        if self.x_train_text is None:
            self.prepare_data()

        self.vectorizer = TextVectorizer(max_features=max_features)
        self.x_train_vec = self.vectorizer.fit_transform(self.x_train_text)
        self.x_test_vec = self.vectorizer.transform(self.x_test_text)
        self.vectorizer.save()
        return self.x_train_vec, self.x_test_vec

    def train(self):
        if self.x_train_vec is None:
            self.extract_features()

        print("[INFO] Tuning optimization weights for classifiers...")
        self.nb_model = MultinomialNB(alpha=1.0)
        self.nb_model.fit(self.x_train_vec, self.y_train)

        self.lr_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        self.lr_model.fit(self.x_train_vec, self.y_train)
        print("[SUCCESS] Training pipeline completed successfully.")

    def evaluate(self):
        models = {"NAIVE_BAYES": self.nb_model, "LOGISTIC_REGRESSION": self.lr_model}
        for name, model in models.items():
            if model is None:
                continue
            y_pred = model.predict(self.x_test_vec)
            print(f"\n===== {name} REPORT =====")
            print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
            print("Metrics Summary:\n", classification_report(self.y_test, y_pred))

    def explain(self, n_top=15):
        feature_names = self.vectorizer.get_features()

        if self.nb_model is not None:
            print(f"\n--- TOP {n_top} CRITICAL TOKENS (NAIVE BAYES) ---")
            indices = self.nb_model.feature_log_prob_[0].argsort()[::-1]
            for idx in indices[:n_top]:
                print(f" -> {feature_names[idx]}")

        if self.lr_model is not None:
            print(f"\n--- TOP {n_top} RISK WEIGHTS (LOGISTIC REGRESSION) ---")
            coefs = self.lr_model.coef_[0]
            indices = coefs.argsort()
            for idx in indices[:n_top]:
                print(f" -> {feature_names[idx]} ({coefs[idx]:.4f})")

    def save(self, lr_path="models/neural_network_model.joblib", nb_path="models/naive_bayes_model.joblib"):
        joblib.dump(self.lr_model, lr_path)
        joblib.dump(self.nb_model, nb_path)
        print("[EXPORT] Serialized models are packed into production environment.")