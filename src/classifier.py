import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
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
        self.mlp_model = None 
        self.y_pred = None    

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

    def extract_features(self, max_features=40000):
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

        self.lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42)
        self.lr_model.fit(self.x_train_vec, self.y_train)
        
        print("[INFO] Training Deep Architecture: Multi-layer Perceptron topology...")
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.05,                   
            early_stopping=True,          
            validation_fraction=0.1,       
            n_iter_no_change=8,           
            max_iter=150,
            random_state=42,
            verbose=True
        )
        self.mlp_model.fit(self.x_train_vec, self.y_train)
        
        self.export_loss_telemetry(self.mlp_model.loss_curve_)
        print("[SUCCESS] Training pipeline completed successfully.")

    def export_loss_telemetry(self, loss_history: list, export_path="models/learning_curve.json"):
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        telemetry_data = {
            "iterations": list(range(1, len(loss_history) + 1)),
            "loss_values": [float(loss) for loss in loss_history]
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_data, f, indent=4, ensure_ascii=False)
        print(f"[EXPORT] Extracted learning curve history packed at: {export_path}")

    def evaluate(self):
        models = {
            "NAIVE_BAYES": self.nb_model, 
            "LOGISTIC_REGRESSION": self.lr_model,
            "MULTI_LAYER_PERCEPTRON": self.mlp_model
        }
        
        for name, model in models.items():
            if model is None:
                continue
            y_pred_local = model.predict(self.x_test_vec)
            print(f"\n===== {name} REPORT =====")
            print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred_local))
            print("Metrics Summary:\n", classification_report(self.y_test, y_pred_local))
            
            if name == "MULTI_LAYER_PERCEPTRON":
                self.y_pred = y_pred_local

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

        if self.mlp_model is not None:
            print(f"\n--- TOP {n_top} PERMUTATION IMPORTANCES (MULTI-LAYER PERCEPTRON) ---")
            x_sample = self.x_test_vec[:200].toarray()
            y_sample = self.y_test[:200]
            
            result = permutation_importance(
                self.mlp_model, x_sample, y_sample, 
                n_repeats=1, max_samples=1.0, random_state=42, n_jobs=-1
            )
            indices = result.importances_mean.argsort()[::-1]
            for idx in indices[:n_top]:
                print(f" -> {feature_names[idx]} ({result.importances_mean[idx]:.4f})")

    def save(self, mlp_path="models/neural_network_model.joblib", lr_path="models/logistic_regression_model.joblib", nb_path="models/naive_bayes_model.joblib"):
        os.makedirs(os.path.dirname(mlp_path), exist_ok=True)
        joblib.dump(self.mlp_model, mlp_path)
        joblib.dump(self.lr_model, lr_path)
        joblib.dump(self.nb_model, nb_path)
        print("[EXPORT] Serialized models are packed into production environment.")