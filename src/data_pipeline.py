import os
import csv
import pandas as pd
from src.text_cleaning import TextCleaner

class DataIngestionPipeline:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.raw_df = None
        self.processed_df = None

    def load_raw_dataset(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"[ERROR] Raw data file not found at: {self.raw_data_path}")
            
        print("[INFO] Loading raw e-commerce reviews dataset...")
        
        self.raw_df = pd.read_csv(
            self.raw_data_path, 
            engine='python',            
            on_bad_lines='skip',        
            quoting=csv.QUOTE_MINIMAL,  
            encoding='utf-8'            
        )
        
        initial_rows = len(self.raw_df)
        self.raw_df.dropna(subset=['Review Text'], inplace=True)
        dropped_rows = initial_rows - len(self.raw_df)
        print(f"[SUCCESS] Dataset loaded. Removed {dropped_rows} rows due to missing text data.")
        
        return self.raw_df

    def transform_and_label_data(self):
        if self.raw_df is None:
            self.load_raw_dataset()
            
        print("[INFO] Transforming ratings and executing auto-labeling strategy...")
        
        working_df = self.raw_df[self.raw_df['Rating'] != 'Rated 3 out of 5 stars'].copy()

        def parse_and_label_rating(rating_value):
            rating_str = str(rating_value).lower().strip()
            if "rated 5" in rating_str or "rated 4" in rating_str:
                return 1
            if rating_str in ["5", "4", "5.0", "4.0"]:
                return 1
            return 0

        working_df['Target Label'] = working_df['Rating'].apply(parse_and_label_rating)

        print("[INFO] Executing text cleaning and advanced noise removal...")

        cleaner = TextCleaner()
        working_df['Review Text'] = working_df['Review Text'].apply(cleaner.clean_review_text)

        working_df = working_df[working_df['Review Text'].str.strip() != ""]
        working_df.dropna(subset=['Review Text'], inplace=True)
        
        self.processed_df = working_df[['Review Text', 'Target Label']].reset_index(drop=True)
        print(f"[SUCCESS] Labeling completed. Total processed instances: {len(self.processed_df)}")
        
        return self.processed_df

    def save_processed_dataset(self, output_dir="data/processed"):
        if self.processed_df is None:
            self.transform_and_label_data()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "clean_data.csv")
        
        self.processed_df.to_csv(output_file_path, index=False)
        print(f"[EXPORT] Clean dataset successfully saved to local directory: {output_file_path}")
        return output_file_path