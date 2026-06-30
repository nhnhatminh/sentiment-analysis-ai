import json
import pickle
import time
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    @staticmethod
    def evaluate_and_save(y_true, y_pred, latency, file_path="model/metrics.json"):
        """Tính toán các chỉ số và lưu ra metrics.json"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'prediction_speed': round(latency, 4),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"[SUCCESS] Metrics saved to {file_path} - models.py:24")
        return metrics

# ==========================================
# HÀM CORE INFERENCE DÀNH CHO DASHBOARD GỌI
# ==========================================
def run_inference(text, model_path="model/model.pkl", vectorizer_path="model/vectorizer.pkl"):
    """
    Hàm này được thiết kế để file app.py (Dashboard) gọi trực tiếp.
    Trả về: (Nhãn cảm xúc, Độ tin cậy, Độ trễ)
    """
    start_time = time.time()
    try:
        # Load Model & Vectorizer
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Trích xuất đặc trưng
        X_input = vectorizer.transform([text])

        # Suy luận
        prediction = model.predict(X_input)[0]
        # Mapping 1 -> Positive, 0 -> Negative
        label = "Positive" if prediction == 1 else "Negative"

        # Tính độ tin cậy
        if hasattr(model, 'predict_proba'):
            prob = float(np.max(model.predict_proba(X_input)[0]))
        else:
            prob = 1.0

        latency = time.time() - start_time
        return label, prob, latency
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e} - models.py:61")
        return None, None, None