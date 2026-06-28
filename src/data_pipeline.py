import os
import csv
import pandas as pd
from src.data_cleaner import TextCleaner  

class DataPipeline:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.raw_df = None
        self.processed_df = None
        self.cleaner = TextCleaner()

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"[ERROR] Raw data file missing at: {self.raw_data_path}")
            
        print("[INFO] Loading dataset from target block...")
        try:
            self.raw_df = pd.read_csv(self.raw_data_path, encoding='utf-8')
        except Exception as e:
            print(f"[WARN] Performance parser fallbacked due to: {str(e)}")
            self.raw_df = pd.read_csv(
                self.raw_data_path, 
                engine='python',            
                on_bad_lines='skip',        
                quoting=csv.QUOTE_MINIMAL,  
                encoding='utf-8'            
            )
        
        self.raw_df.dropna(subset=['Review Text'], inplace=True)
        
        self.raw_df['Rating'] = self.raw_df['Rating'].astype(str).str.extract(r'(\d+)', expand=False)
        self.raw_df['Rating'] = pd.to_numeric(self.raw_df['Rating'], errors='coerce')
        self.raw_df.dropna(subset=['Rating'], inplace=True)
        self.raw_df['Rating'] = self.raw_df['Rating'].astype(int)
        
        return self.raw_df

    def process(self) -> pd.DataFrame:
        if self.raw_df is None:
            self.load()
            
        print("[INFO] Screening data matrix and transforming target labels...")
        working_df = self.raw_df[self.raw_df['Rating'] != 3].copy()
        working_df['Target Label'] = working_df['Rating'].apply(lambda score: 1 if score >= 4 else 0)

        print("[INFO] Inverting noisy sequences into processed text arrays...")
        working_df['Cleaned Review'] = working_df['Review Text'].apply(self.cleaner.clean)

        working_df = working_df[working_df['Cleaned Review'].str.strip() != ""]
        working_df.dropna(subset=['Cleaned Review'], inplace=True)
        
        self.processed_df = working_df[['Review Text', 'Cleaned Review', 'Target Label']].reset_index(drop=True)
        return self.processed_df

    def save(self, output_dir: str = "data/processed") -> str:
        if self.processed_df is None:
            self.process()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "clean_data.csv")
        self.processed_df.to_csv(output_file_path, index=False)
        print(f"[SUCCESS] Clean dataset stored at: {output_file_path}")
        return output_file_path