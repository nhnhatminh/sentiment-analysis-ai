import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split

# Thêm thư mục 'src' vào sys.path để có thể import các module nội bộ độc lập với thư mục chạy lệnh
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_pipeline import DataIngestionPipeline
from preprocessing import TextCleaner
from vectorizer import TextVectorizer
from models import train_and_evaluate_model

def main():
    print("="*60)
    print(" STARTING MACHINE LEARNING TRAINING & EVALUATION PIPELINE ")
    print("="*60)

    # Định nghĩa các đường dẫn tệp tin động dựa trên project root
    project_root = os.path.dirname(src_dir)
    raw_data_path = os.path.join(project_root, "data", "raw", "ecom_reviews.csv")
    
    classifier_export_path = os.path.join(project_root, "classifier.pkl")
    vectorizer_export_path = os.path.join(project_root, "vectorizer.pkl")
    metrics_export_path = os.path.join(project_root, "metrics.json")

    # --- BƯỚC 1: Đọc và gán nhãn tập dữ liệu (Mảng 1) ---
    print("\n[STEP 1] Data Ingestion & Scoring Normalization...")
    if not os.path.exists(raw_data_path):
        print(f"[WARNING] Raw data not found at {raw_data_path}. Creating sample CSV first...")
        # (Nếu vì lý do gì đó file data chưa được ghi, chương trình sẽ báo lỗi hoặc tự tạo)
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")

    ingestion_pipeline = DataIngestionPipeline(raw_data_path)
    # Tải và chuẩn hóa nhãn (Giữ 4-5 sao -> 1, 1-2 sao -> 0, drop 3 sao)
    processed_df = ingestion_pipeline.transform_and_label_data()

    # --- BƯỚC 2: Làm sạch chuỗi văn bản (Mảng 1) ---
    print("\n[STEP 2] Text Cleaning (Text Purifying)...")
    cleaner = TextCleaner()
    processed_df['Cleaned Text'] = processed_df['Review Text'].apply(cleaner.clean)
    
    # Loại bỏ các dòng trống sau khi làm sạch (nếu có)
    processed_df = processed_df[processed_df['Cleaned Text'].str.strip() != ''].reset_index(drop=True)
    print(f"[SUCCESS] Cleaned text dataset size: {len(processed_df)} instances.")

    # --- BƯỚC 3: Đặc trưng & Số hóa văn bản (Mảng 2) ---
    print("\n[STEP 3] Text Vectorization (TF-IDF sparse matrix representation)...")
    vectorizer = TextVectorizer(max_features=2000, ngram_range=(1, 2))
    X_sparse = vectorizer.fit_transform(processed_df['Cleaned Text'])
    y = processed_df['Target Label'].values

    # Lưu Vectorizer state thành tệp tĩnh vectorizer.pkl
    vectorizer.save(vectorizer_export_path)

    # --- BƯỚC 4: Chia tập dữ liệu Train/Test 80/20 (Mảng 3) ---
    print("\n[STEP 4] Splitting Dataset into Train/Test Sets (80/20)...")
    # Sử dụng stratify=y để giữ nguyên tỉ lệ nhãn trong cả 2 tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"-> Train set shape: {X_train.shape}")
    print(f"-> Test set shape: {X_test.shape}")

    # Lấy danh sách từ vựng từ bộ Vectorizer phục vụ giải trình thuộc tính mô hình
    feature_names = vectorizer.vectorizer.get_feature_names_out()

    # --- BƯỚC 5: Huấn luyện và Đánh giá các Mô hình (Mảng 3) ---
    print("\n[STEP 5] Training & Evaluating Classifiers...")
    candidate_models = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]
    evaluation_results = {}
    best_f1 = -1.0
    best_model_data = None

    for model_name in candidate_models:
        print(f"\nTraining {model_name}...")
        result = train_and_evaluate_model(
            model_name, X_train, y_train, X_test, y_test, feature_names
        )
        
        metrics = result["metrics"]
        print(f"   Accuracy : {metrics['accuracy']:.4f}")
        print(f"   F1-Score : {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall   : {metrics['recall']:.4f}")
        
        # Lưu kết quả đánh giá (ngoại trừ đối tượng mô hình thô) vào dict để ghi log
        evaluation_results[model_name] = {
            "metrics": result["metrics"],
            "top_features": result["top_features"]
        }

        # Lựa chọn mô hình tốt nhất dựa trên F1-Score
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_data = result

    # --- BƯỚC 6: Xuất báo cáo metrics.json và classifier.pkl (Mảng 3) ---
    print("\n[STEP 6] Saving Logs and Exporting Best Classifier...")
    
    # Xuất metrics.json chứa toàn bộ thông tin hiệu năng và giải trình cơ chế từ vựng
    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    print(f"[EXPORT] Detailed metrics saved to: {metrics_export_path}")

    # Xuất mô hình có F1-Score cao nhất thành classifier.pkl
    best_model_name = best_model_data["model_name"]
    best_model_obj = best_model_data["model_object"]
    
    joblib.dump(best_model_obj, classifier_export_path)
    print(f"[EXPORT] Best Classifier ({best_model_name}) serialized to: {classifier_export_path}")

    # In kết luận tóm tắt trực quan
    print("\n" + "="*60)
    print(f" TRAINING PIPELINE SUCCESSFULLY COMPLETED ")
    print(f" Best Model Selected: {best_model_name}")
    print(f" Best Test F1-Score : {best_f1:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
