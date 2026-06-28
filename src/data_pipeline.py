import os
import csv
import io
import pandas as pd
from text_cleaning import TextCleaner

class DataIngestionPipeline:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.raw_df = None
        self.processed_df = None
        self.text_cleaner = TextCleaner()

    def load_raw_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"[ERROR] Tập dữ liệu thô không tồn tại tại: {self.raw_data_path}")
            
        print("[INFO] Launching high-performance data ingestion engine...")
        try:
            self.raw_df = pd.read_csv(self.raw_data_path, encoding='utf-8')
        except (pd.errors.ParserError, Exception) as error:
            print(f"[WARN] C Engine parser bottleneck detected ({str(error)}). Falling back to safe Python Engine...")
            self.raw_df = pd.read_csv(
                self.raw_data_path, 
                engine='python',            
                on_bad_lines='skip',        
                quoting=csv.QUOTE_MINIMAL,  
                encoding='utf-8'            
            )
        
        initial_rows = len(self.raw_df)
        self.raw_df.dropna(subset=['Review Text'], inplace=True)
        print(f"[SUCCESS] Raw data loaded. Screened out {initial_rows - len(self.raw_df)} rows due to null text fields.")

        self.raw_df['Rating'] = self.raw_df['Rating'].astype(str).str.extract(r'(\d+)', expand=False)
        self.raw_df['Rating'] = pd.to_numeric(self.raw_df['Rating'], errors='coerce')
        self.raw_df.dropna(subset=['Rating'], inplace=True)
        self.raw_df['Rating'] = self.raw_df['Rating'].astype(int)
        
        return self.raw_df

    def transform_and_label_data(self) -> pd.DataFrame:
        if self.raw_df is None:
            self.load_raw_dataset()
            
        print("[INFO] Eliminating neutral 3-star signals and executing binary labeling strategy...")
        working_df = self.raw_df[self.raw_df['Rating'] != 3].copy()

        working_df['Target Label'] = working_df['Rating'].apply(lambda score: 1 if score >= 4 else 0)

        print("[INFO] Executing optimized text cleaning loop across data matrix...")
        working_df['Cleaned Review'] = working_df['Review Text'].apply(self.text_cleaner.clean_review_text)

        working_df = working_df[working_df['Cleaned Review'].str.strip() != ""]
        working_df.dropna(subset=['Cleaned Review'], inplace=True)
        
        self.processed_df = working_df[['Review Text', 'Cleaned Review', 'Target Label']].reset_index(drop=True)
        print(f"[SUCCESS] Processing pipeline finished. Total structured samples: {len(self.processed_df)}")
        
        return self.processed_df

    def save_processed_dataset(self, output_dir: str = "data/processed") -> str:
        if self.processed_df is None:
            self.transform_and_label_data()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "clean_data.csv")
        
        self.processed_df.to_csv(output_file_path, index=False)
        print(f"[EXPORT] Refactored clean dataset successfully saved to: {output_file_path}")
        return output_file_path