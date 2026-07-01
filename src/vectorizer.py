import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Khởi tạo bộ số hóa văn bản sử dụng phương pháp TF-IDF.
        ngram_range=(1, 2) giúp giữ lại các cụm 2 từ (bigrams) như 'not bad', 'never buy'
        để giải quyết bài toán phủ định trong phân tích cảm xúc.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=r'(?u)\b\w+\b'  # Cho phép từ có 1 chữ cái (ví dụ 'a', 'i', 'no')
        )

    def fit(self, texts):
        """
        Huấn luyện bộ Vectorizer trên danh sách các văn bản sạch.
        """
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        """
        Số hóa các văn bản mới thành ma trận thưa TF-IDF.
        """
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """
        Vừa huấn luyện vừa số hóa văn bản.
        """
        return self.vectorizer.fit_transform(texts)

    def save(self, filepath):
        """
        Đóng gói và lưu trạng thái của bộ Vectorizer xuống ổ đĩa cục bộ.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[SERIALIZE] TextVectorizer state successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Tải trạng thái của bộ Vectorizer từ tệp tĩnh.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[ERROR] Serialized vectorizer file not found at: {filepath}")
        return joblib.load(filepath)

if __name__ == "__main__":
    # Script export tệp tĩnh bộ từ vựng
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(project_dir, "data", "processed", "clean_sentiment_data.csv")
    
    if not os.path.exists(processed_data_path):
        print(f"[ERROR] Clean data file not found at: {processed_data_path}")
        print("[INFO] Please run 'python -m src.data_pipeline' first to generate clean data.")
    else:
        print(f"[INFO] Reading cleaned dataset from: {processed_data_path}")
        df = pd.read_csv(processed_data_path)
        
        # Đảm bảo Review Text không chứa NaN
        df['Review Text'] = df['Review Text'].fillna("")
        
        print("[INFO] Initializing and fitting TextVectorizer...")
        vectorizer = TextVectorizer(max_features=5000, ngram_range=(1, 2))
        X_sparse = vectorizer.fit_transform(df['Review Text'])
        
        print("[SUCCESS] Vocabulary fitted.")
        print(f"[INFO] Sparse Matrix Type: {type(X_sparse)}")
        print(f"[INFO] Matrix Shape: {X_sparse.shape} (instances: {X_sparse.shape[0]}, features/vocabulary: {X_sparse.shape[1]})")
        print(f"[INFO] Non-zero elements: {X_sparse.nnz}")
        
        # Lưu vào thư mục models/vectorizer.pkl
        models_dir = os.path.join(project_dir, "models")
        export_path = os.path.join(models_dir, "vectorizer.pkl")
        vectorizer.save(export_path)
        
        # Đồng thời lưu một bản ở thư mục dự án gốc để dễ bàn giao
        root_export_path = os.path.join(project_dir, "vectorizer.pkl")
        vectorizer.save(root_export_path)
