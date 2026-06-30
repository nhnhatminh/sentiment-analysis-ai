import pandas as pd
import time
import os
from src.text_cleaner import TextCleaner
from src.data_preprocessing import DataProcessor
from src.classifiers import SentimentClassifier
from src.models import ModelEvaluator

def main():
    print("=== BẮT ĐẦU CHẠY PIPELINE MÁY HỌC === - run_pipeline.py:10")
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # 1. Tạo tập dữ liệu thô giả lập (Raw Data)
    print("\n1. Tạo file data/raw.csv... - run_pipeline.py:15")
    data = [
        ("I love this product, it is absolutely amazing and works perfectly.", "Positive"),
        ("This is the worst item I have ever bought. It broke on day one.", "Negative"),
        ("Great quality, fast shipping, highly recommend this seller.", "Positive"),
        ("Terrible customer service and the product is defective.", "Negative"),
        ("I am so happy with this purchase! Five stars.", "Positive"),
        ("Do not buy this. Waste of money and time.", "Negative")
    ] * 50
    raw_df = pd.DataFrame(data, columns=['text', 'label'])
    raw_df.to_csv("data/raw.csv", index=False)

    # 2. Làm sạch dữ liệu (Text Cleaning)
    print("2. Đang làm sạch văn bản và tạo data/clean.csv... - run_pipeline.py:28")
    cleaner = TextCleaner()
    # Chuyển nhãn Positive->1, Negative->0 cho model học
    raw_df['Target Label'] = raw_df['label'].apply(lambda x: 1 if x == 'Positive' else 0)
    raw_df['clean_text'] = raw_df['text'].apply(cleaner.clean_text)
    
    clean_df = raw_df[['clean_text', 'label', 'Target Label']].copy()
    clean_df.rename(columns={'clean_text': 'text'}, inplace=True)
    clean_df.to_csv("data/clean.csv", index=False)

    # 3. Vector hóa (Data Processing)
    print("3. Vector hóa văn bản... - run_pipeline.py:39")
    processor = DataProcessor(max_features=500)
    X = processor.fit_transform(clean_df['text'])
    y = clean_df['Target Label']
    processor.save_vectorizer()

    # 4. Huấn luyện (Classification)
    print("4. Huấn luyện mô hình... - run_pipeline.py:46")
    start_train = time.time()
    classifier = SentimentClassifier()
    classifier.train(X, y)
    classifier.save_model()
    train_latency = time.time() - start_train

    # 5. Đánh giá (Evaluation)
    print("5. Đánh giá mô hình và lưu metrics.json... - run_pipeline.py:54")
    y_pred = classifier.predict(X)
    ModelEvaluator.evaluate_and_save(y, y_pred, train_latency)

    print("\n=== HOÀN TẤT PIPELINE! BẠN CÓ THỂ MỞ DASHBOARD. === - run_pipeline.py:58")

if __name__ == "__main__":
    main()