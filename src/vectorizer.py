import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        """
        Khởi tạo TextVectorizer với cấu hình tối ưu hóa từ vựng (max_features) và n-gram.
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, texts):
        """
        Huấn luyện TF-IDF Vectorizer và biến đổi chuỗi văn bản sạch thành ma trận thưa (scipy.sparse).
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Biến đổi chuỗi văn bản mới thành ma trận thưa dựa trên từ vựng đã học.
        """
        return self.vectorizer.transform(texts)

    def save(self, file_path="vectorizer.pkl"):
        """
        Đóng gói và lưu trữ trạng thái của Vectorizer thành tệp tĩnh trên đĩa local.
        """
        # Đảm bảo thư mục cha tồn tại
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        joblib.dump(self.vectorizer, file_path)
        print(f"[EXPORT] Vectorizer state successfully saved to: {file_path}")

    def load(self, file_path="vectorizer.pkl"):
        """
        Tải trạng thái của Vectorizer từ tệp tĩnh lên RAM.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] Vectorizer file not found at: {file_path}")
            
        self.vectorizer = joblib.load(file_path)
        print(f"[IMPORT] Vectorizer state successfully loaded from: {file_path}")
        return self
