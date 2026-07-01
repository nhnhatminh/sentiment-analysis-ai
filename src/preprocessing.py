import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class TextCleaner:
    def __init__(self):
        """
        Khởi tạo TextCleaner với danh sách stop words tiếng Anh tiêu chuẩn từ scikit-learn.
        """
        self.stop_words = set(ENGLISH_STOP_WORDS)

    def clean(self, text):
        """
        Làm sạch chuỗi văn bản thô:
        1. Chuyển text về lowercase.
        2. Dùng Regex loại bỏ dấu câu, ký tự đặc biệt và chữ số.
        3. Lọc bỏ danh sách Stopwords tiếng Anh tiêu chuẩn.
        """
        if not isinstance(text, str):
            return ""

        # Chuyển text về lowercase
        text = text.lower()

        # Regex loại bỏ dấu câu, ký tự đặc biệt và chữ số (chỉ giữ lại chữ cái a-z và khoảng trắng)
        text = re.sub(r'[^a-z\s]', '', text)

        # Tách từ và loại bỏ Stopwords
        words = text.split()
        cleaned_words = [word for word in words if word not in self.stop_words]

        # Trả về chuỗi văn bản sạch
        return " ".join(cleaned_words)

    def __call__(self, text):
        """
        Cho phép gọi trực tiếp instance như một hàm.
        """
        return self.clean(text)
