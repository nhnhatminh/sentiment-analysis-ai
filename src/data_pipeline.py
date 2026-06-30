import os
import pandas as pd

class DataIngestionPipeline:
    def __init__(self, raw_data_path):
        """
        Khởi tạo đường ống nạp dữ liệu với đường dẫn tệp tin CSV gốc.
        """
        self.raw_data_path = raw_data_path
        self.raw_df = None
        self.processed_df = None

    def load_raw_dataset(self):
        """
        Đọc tệp tin dữ liệu từ thư mục cục bộ và xử lý các dòng bị khuyết thiếu.
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"[ERROR] Raw data file not found at: {self.raw_data_path}")
            
        print("[INFO] Loading raw e-commerce reviews dataset...")
        self.raw_df = pd.read_csv(self.raw_data_path)
        
        initial_rows = len(self.raw_df)
        self.raw_df.dropna(subset=['Review Text'], inplace=True)
        dropped_rows = initial_rows - len(self.raw_df)
        print(f"[SUCCESS] Dataset loaded. Removed {dropped_rows} rows due to missing text data.")
        
        return self.raw_df

    def transform_and_label_data(self):
        """
        Lọc bỏ điểm đánh giá trung lập 3 sao và thực hiện gán nhãn nhị phân toán học.
        Rating 4-5 sao -> Nhãn 1 (Positive)
        Rating 1-2 sao -> Nhãn 0 (Negative)
        """
        if self.raw_df is None:
            self.load_raw_dataset()
            
        print("[INFO] Transforming ratings and executing auto-labeling strategy...")
        
        working_df = self.raw_df[self.raw_df['Rating'] != 3].copy()
        
        working_df['Target Label'] = working_df['Rating'].apply(lambda score: 1 if score >= 4 else 0)
        
        self.processed_df = working_df[['Review Text', 'Target Label']].reset_index(drop=True)
        print(f"[SUCCESS] Labeling completed. Total processed instances: {len(self.processed_df)}")
        
        return self.processed_df

    def save_processed_dataset(self, output_dir="data/processed"):
        """
        Lưu trữ tập dữ liệu sạch sau xử lý thành tệp tin tĩnh trên ổ đĩa local.
        """
        if self.processed_df is None:
            self.transform_and_label_data()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "clean_sentiment_data.csv")
        
        self.processed_df.to_csv(output_file_path, index=False)
        print(f"[EXPORT] Clean dataset successfully saved to local directory: {output_file_path}")
        return output_file_path