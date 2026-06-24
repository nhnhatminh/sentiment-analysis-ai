import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class ReviewVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, 
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            smooth_idf=True
        )

    def fit_transform_data(self, train_texts):
        print(f"[INFO] Fitting vocabulary and transforming text data with max_features={self.max_features}...")
        sparse_matrix = self.vectorizer.fit_transform(train_texts)
        print(f"[SUCCESS] Vectorization completed. Sparse matrix shape: {sparse_matrix.shape}")
        return sparse_matrix

    def transform_data(self, new_texts):
        return self.vectorizer.transform(new_texts)

    def save_vectorizer_model(self, file_path="models/tfidf_vectorizer.joblib"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.vectorizer, file_path)
        print(f"[EXPORT] TF-IDF Vectorizer model successfully saved to: {file_path}")

    def load_vectorizer_model(self, file_path="models/tfidf_vectorizer.joblib"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] Serialized vectorizer model not found at: {file_path}")
        self.vectorizer = joblib.load(file_path)
        print(f"[IMPORT] TF-IDF Vectorizer model successfully loaded from: {file_path}")
        return self.vectorizer