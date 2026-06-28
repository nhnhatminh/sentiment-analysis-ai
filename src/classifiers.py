import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class ExplainableSentimentPipeline:
    def __init__(self, data_path="data/processed/clean_data.csv"):
        self.data_path = data_path
        
        self.x_train_text = None
        self.x_test_text = None
        self.y_train = None
        self.y_test = None
        
        self.x_train_vectorized = None
        self.x_test_vectorized = None

        self.vectorizer = None
        self.nb_model = None
        self.lr_model = None

    def load_and_split_data(self, test_size=0.2, random_state=42):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Dataset không tồn tại tại: {self.data_path}")

        print(f"[INFO] Đang nạp tập dữ liệu sạch từ đường dẫn: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        df.dropna(subset=['Review Text', 'Target Label'], inplace=True)
        
        x_raw = df['Review Text'].values.astype('U')
        y_raw = df['Target Label'].values

        print(f"[INFO] Đang thực hiện phân rã dữ liệu theo tỷ lệ Train 80% / Test 20%...")
        
        self.x_train_text, self.x_test_text, self.y_train, self.y_test = train_test_split(
            x_raw, y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw
        )
        
        return self.x_train_text, self.x_test_text, self.y_train, self.y_test

    def prepare_features(self, max_features=5000):
        if self.x_train_text is None:
            self.load_and_split_data()

        print(f"[INFO] Đang khởi tạo bộ chuyển đổi TF-IDF với {max_features} đặc trưng...")
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        
        self.x_train_vectorized = self.vectorizer.fit_transform(self.x_train_text)
        self.x_test_vectorized = self.vectorizer.transform(self.x_test_text)
        
        print("[SUCCESS] Quá trình trích xuất ma trận đặc trưng hoàn tất.")
        return self.x_train_vectorized, self.x_test_vectorized

    def train_naive_bayes(self):
        if self.x_train_vectorized is None:
            self.prepare_features()

        print("[INFO] Đang cấu hình và khớp trọng số cho mô hình Naive Bayes...")
        self.nb_model = MultinomialNB(alpha=1.0)
        self.nb_model.fit(self.x_train_vectorized, self.y_train)
        print("[SUCCESS] Huấn luyện Naive Bayes hoàn tất.")
        return self.nb_model

    def train_logistic_regression(self):
        if self.x_train_vectorized is None:
            self.prepare_features()

        print("[INFO] Đang tối ưu hóa hàm chi phí cho mô hình Logistic Regression...")
        self.lr_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        self.lr_model.fit(self.x_train_vectorized, self.y_train)
        print("[SUCCESS] Huấn luyện Logistic Regression hoàn tất.")
        return self.lr_model

    def evaluate_model(self, model_type="nb"):
        if model_type == "nb":
            current_model = self.nb_model
            name = "MULTINOMIAL NAIVE BAYES"
        else:
            current_model = self.lr_model
            name = "LOGISTIC REGRESSION"

        if current_model is None:
            raise ValueError(f"[ERROR] Mô hình {model_type} chưa được huấn luyện.")

        y_pred = current_model.predict(self.x_test_vectorized)
        conf_mat = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'])

        print("\n" + "="*20 + f" {name} REPORT " + "="*20)
        print("\n--- Confusion Matrix ---")
        print(conf_mat)
        print("\n--- Detailed Classification Metrics ---")
        print(class_report)
        print("="*60 + "\n")
        return conf_mat, class_report

    def explain_model_features(self, n_top=10):
        if self.vectorizer is None:
            raise ValueError("[ERROR] Bộ vectorizer chưa được khởi tạo.")
            
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        if self.nb_model is not None:
            print("\n" + "-"*15 + " NAIVE BAYES INTERPRETABILITY " + "-"*15)
            neg_log_prob = self.nb_model.feature_log_prob_[0].argsort()[::-1]
            pos_log_prob = self.nb_model.feature_log_prob_[1].argsort()[::-1]
            
            print(f"[NB] Top {n_top} từ khóa dẫn dắt lớp Tiêu cực (Nguy cơ khủng hoảng):")
            for idx in neg_log_prob[:n_top]:
                print(f"  -> {feature_names[idx]}")
                
            print(f"[NB] Top {n_top} từ khóa dẫn dắt lớp Tích cực:")
            for idx in pos_log_prob[:n_top]:
                print(f"  -> {feature_names[idx]}")

        if self.lr_model is not None:
            print("\n" + "-"*15 + " LOGISTIC REGRESSION INTERPRETABILITY " + "-"*15)
            coefficients = self.lr_model.coef_[0]
            sorted_coeff_indices = coefficients.argsort()

            print(f"[LR] Top {n_top} từ khóa có trọng số Tiêu cực mạnh nhất:")
            for idx in sorted_coeff_indices[:n_top]:
                print(f"  -> {feature_names[idx]} (Weight: {coefficients[idx]:.4f})")

            print(f"[LR] Top {n_top} từ khóa có trọng số Tích cực mạnh nhất:")
            for idx in sorted_coeff_indices[::-1][:n_top]:
                print(f"  -> {feature_names[idx]} (Weight: {coefficients[idx]:.4f})")

    def save_explainable_artifacts(self):
        os.makedirs("models", exist_ok=True)
        if self.vectorizer:
            joblib.dump(self.vectorizer, "models/tfidf_vectorizer.joblib")
        if self.nb_model:
            joblib.dump(self.nb_model, "models/naive_bayes_model.joblib")
        if self.lr_model:
            joblib.dump(self.lr_model, "models/neural_network_model.joblib") 
        print("[EXPORT] Toàn bộ Artifacts giải thích được đã xuất kho thành công.")