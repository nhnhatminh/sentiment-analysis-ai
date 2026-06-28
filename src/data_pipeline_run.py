import time
from data_pipeline import DataIngestionPipeline

def run_data_infrastructure_test():
    raw_dataset_path = "data/raw/Amazon_Reviews.csv"
    
    print("="*15 + " STARTING DATA INFRASTRUCTURE PHASE " + "="*15)
    start_time = time.time()
    
    try:
        pipeline = DataIngestionPipeline(raw_data_path=raw_dataset_path)
        pipeline.save_processed_dataset()
        
        execution_duration = time.time() - start_time
        print(f"[PERFORMANCE] Total pipeline pipeline runtime: {execution_duration:.2f} seconds.")
        print("="*20 + " PIPELINE PHASE COMPLETED " + "="*20 + "\n")
        
    except Exception as error:
        print(f"\n[CRITICAL] Data infrastructure pipeline crashed due to: {str(error)}")

if __name__ == "__main__":
    run_data_infrastructure_test()