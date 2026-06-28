import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))

    def fit_transform(self, train_texts):
        return self.vectorizer.fit_transform(train_texts)

    def transform(self, target_texts):
        return self.vectorizer.transform(target_texts)

    def save(self, file_path="models/tfidf_vectorizer.joblib"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.vectorizer, file_path)
        print(f"[SUCCESS] Vectorizer model exported to: {file_path}")

    def get_features(self):
        return self.vectorizer.get_feature_names_out()