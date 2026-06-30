import re

class TextCleaner:
    def __init__(self):
        # Tập hợp các từ dừng (stop words) cơ bản tiếng Anh
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'it'}

    def clean_text(self, text):
        """Làm sạch văn bản đầu vào"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower() # Chuyển chữ thường
        text = re.sub(r'[^a-z\s]', '', text) # Xóa dấu câu, số, ký tự đặc biệt
        
        # Xóa stop words
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        
        return " ".join(words)