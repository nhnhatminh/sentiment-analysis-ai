from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class DataProcessor:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, file_path="model/vectorizer.pkl"):
        """Lưu bộ vector hóa ra file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"[SUCCESS] Vectorizer saved to {file_path} - data_preprocessing.py:20")