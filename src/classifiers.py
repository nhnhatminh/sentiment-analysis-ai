from sklearn.linear_model import LogisticRegression
import pickle
import os

class SentimentClassifier:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)

    def train(self, X_train, y_train):
        """Huấn luyện mô hình"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
        
    def save_model(self, file_path="model/model.pkl"):
        """Lưu mô hình ra file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[SUCCESS] Model saved to {file_path} - classifiers.py:21")