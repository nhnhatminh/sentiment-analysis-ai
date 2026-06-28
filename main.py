import time
from src.data_pipeline import DataPipeline
from src.classifier import ReviewClassifier

def main():
    print("=================== STARTING MACHINE LEARNING PIPELINE ===================")
    start_time = time.time()
    
    try:
        pipeline = DataPipeline(raw_data_path="data/raw/Amazon_Reviews.csv")
        clean_path = pipeline.save()
        
        model = ReviewClassifier(data_path=clean_path)
        model.prepare_data()
        model.extract_features(max_features=5000)
        model.train()
        
        model.evaluate()
        model.explain(n_top=15)
        
        model.save()
        print(f"\n[SUCCESS] Pipeline finished in: {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline terminated due to: {str(e)}")

if __name__ == "__main__":
    main()