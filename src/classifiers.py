import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.vectorizer import ReviewVectorizer

class SentimentClassifier:
    def __init__(self, data_path="data/processed/clean_data.csv"):
        self.data_path = data_path
        
        self.x_train_text = None
        self.x_test_text = None
        self.y_train = None
        self.y_test = None
        
        self.x_train_vectorized = None
        self.x_test_vectorized = None

    def load_and_split_data(self, test_size=0.2, random_state=42):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Cleaned dataset file not found at: {self.data_path}")

        print(f"[INFO] Loading clean dataset from local directory: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        df.dropna(subset=['Review Text', 'Target Label'], inplace=True)
        
        x_raw = df['Review Text'].values.astype('U')
        y_raw = df['Target Label'].values

        print(f"[INFO] Splitting dataset into train (80%) and test (20%) partitions...")
        
        self.x_train_text, self.x_test_text, self.y_train, self.y_test = train_test_split(
            x_raw, y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw
        )

        print(f"[SUCCESS] Data split completed successfully.")
        print(f"[SUCCESS] Training samples count: {len(self.x_train_text)}")
        print(f"[SUCCESS] Testing samples count: {len(self.x_test_text)}")
        
        return self.x_train_text, self.x_test_text, self.y_train, self.y_test

    def prepare_features(self, max_features=5000):
        if self.x_train_text is None:
            self.load_and_split_data()

        print("[INFO] Kicking off isolated feature extraction pipeline...")
        vectorizer_manager = ReviewVectorizer(max_features=max_features)
        
        self.x_train_vectorized = vectorizer_manager.fit_transform_data(self.x_train_text)
        
        self.x_test_vectorized = vectorizer_manager.transform_data(self.x_test_text)
        
        vectorizer_manager.save_vectorizer_model()
        print("[SUCCESS] Feature matrices are fully prepared for modeling phase.")
        
        return self.x_train_vectorized, self.x_test_vectorized