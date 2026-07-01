import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, feature_names):
    """
    Huấn luyện và đánh giá mô hình học máy theo tên được yêu cầu.
    Trích xuất top các từ vựng có ảnh hưởng mạnh nhất đến quyết định phân lớp.
    """
    # Khởi tạo mô hình tương ứng
    if model_name == "Multinomial Naive Bayes":
        model = MultinomialNB()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Linear SVM":
        # dual=False khuyến nghị khi n_samples > n_features hoặc sử dụng dual='auto' trong sklearn 1.5.0
        model = LinearSVC(dual=False, random_state=42, max_iter=1000)
    else:
        raise ValueError(f"[ERROR] Unknown model name: {model_name}")

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm thử
    y_pred = model.predict(X_test)

    # Tính toán các chỉ số hiệu năng
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)

    # Trích xuất đặc trưng có trọng số ảnh hưởng mạnh nhất
    top_positive_words = []
    top_negative_words = []

    if model_name == "Multinomial Naive Bayes":
        # Với Naive Bayes, sử dụng log xác suất điều kiện: feature_log_prob_
        # Shape: (n_classes, n_features) -> class 0: Negative, class 1: Positive
        log_prob_neg = model.feature_log_prob_[0]
        log_prob_pos = model.feature_log_prob_[1]
        
        # Log-likelihood ratio để tìm từ phân biệt tốt nhất
        # Trị số dương lớn -> chỉ thị Positive; Trị số âm lớn -> chỉ thị Negative
        ratio = log_prob_pos - log_prob_neg
        sorted_indices = np.argsort(ratio)
        
        # 10 từ có tỉ lệ Positive/Negative cao nhất (chỉ thị Tích cực)
        top_pos_indices = sorted_indices[-10:][::-1]
        for idx in top_pos_indices:
            top_positive_words.append({"word": feature_names[idx], "weight": float(ratio[idx])})
            
        # 10 từ có tỉ lệ Negative/Positive cao nhất (chỉ thị Tiêu cực)
        top_neg_indices = sorted_indices[:10]
        for idx in top_neg_indices:
            top_negative_words.append({"word": feature_names[idx], "weight": float(ratio[idx])})

    elif model_name in ["Logistic Regression", "Linear SVM"]:
        # Với Logistic Regression và SVM, sử dụng coef_
        # Shape: (1, n_features) cho phân loại nhị phân
        coef = model.coef_[0]
        sorted_indices = np.argsort(coef)
        
        # Hệ số dương lớn nhất chỉ thị Tích cực (Positive)
        top_pos_indices = sorted_indices[-10:][::-1]
        for idx in top_pos_indices:
            top_positive_words.append({"word": feature_names[idx], "weight": float(coef[idx])})
            
        # Hệ số âm nhỏ nhất chỉ thị Tiêu cực (Negative)
        top_neg_indices = sorted_indices[:10]
        for idx in top_neg_indices:
            top_negative_words.append({"word": feature_names[idx], "weight": float(coef[idx])})

    # Đóng gói kết quả
    result = {
        "model_name": model_name,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()  # Chuyển numpy array thành list để ghi vào JSON
        },
        "top_features": {
            "positive": top_positive_words,
            "negative": top_negative_words
        },
        "model_object": model
    }
    
    return result
