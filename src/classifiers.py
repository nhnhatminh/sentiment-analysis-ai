import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.vectorizer import ReviewVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

class SentimentClassifier:
    def __init__(self, data_path="data/processed/clean_data.csv"):
        self.data_path = data_path
        
        self.x_train_text = None
        self.x_test_text = None
        self.y_train = None
        self.y_test = None
        
        self.x_train_vectorized = None
        self.x_test_vectorized = None

        self.model = None

    def load_and_split_data(self, test_size=0.2, random_state=42):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Cleaned dataset file not found at: {self.data_path}")

        print(f"[INFO] Loading clean dataset from local directory: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        df.dropna(subset=['Review Text', 'Target Label'], inplace=True)
        
        x_raw = df['Review Text'].values.astype('U')
        y_raw = df['Target Label'].values

        print(f"[INFO] Splitting dataset into train (80%) and test (20%) partitions...")
        
        self.x_train_text, self.x_test_text, self.y_train, self.y_test = train_test_split(
            x_raw, y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw
        )

        print(f"[SUCCESS] Data split completed successfully.")
        print(f"[SUCCESS] Training samples count: {len(self.x_train_text)}")
        print(f"[SUCCESS] Testing samples count: {len(self.x_test_text)}")
        
        return self.x_train_text, self.x_test_text, self.y_train, self.y_test

    def prepare_features(self, max_features=5000):
        if self.x_train_text is None:
            self.load_and_split_data()

        print("[INFO] Kicking off isolated feature extraction pipeline...")
        vectorizer_manager = ReviewVectorizer(max_features=max_features)
        
        self.x_train_vectorized = vectorizer_manager.fit_transform_data(self.x_train_text)
        self.x_test_vectorized = vectorizer_manager.transform_data(self.x_test_text)
        
        vectorizer_manager.save_vectorizer_model()
        print("[SUCCESS] Feature matrices are fully prepared for modeling phase.")
        
        return self.x_train_vectorized, self.x_test_vectorized
    
    def train_multi_layer_perceptron(self, max_iter=20):
        if self.x_train_vectorized is None:
            self.prepare_features()

        print("[INFO] Initializing Multi-Layer Perceptron network architecture...")
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  
            activation='relu',            
            solver='adam',                
            batch_size=128,               
            max_iter=max_iter,            
            early_stopping=True,          
            validation_fraction=0.1,      
            n_iter_no_change=5,           
            verbose=True,                 
            random_state=42
        )
        
        print(f"[INFO] Fitting Deep Neural Network weights with Early Stopping protection...")
        self.model.fit(self.x_train_vectorized, self.y_train)
        print("[SUCCESS] Multi-Layer Perceptron training pipeline completed.")
        return self.model

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("[ERROR] Neural Network has not been trained yet.")

        print("[INFO] Generating network predictions on 3,920 test instances...")
        y_pred = self.model.predict(self.x_test_vectorized)
        
        conf_mat = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'])
        
        print("\n" + "="*20 + " MULTI-LAYER PERPCEPTRON REPORT " + "="*20)
        print("\n--- Confusion Matrix ---")
        print(conf_mat)
        print("\n--- Detailed Neural Network Classification Metrics ---")
        print(class_report)
        print("="*66 + "\n")
        
        return conf_mat, class_report

    def save_classifier_model(self, file_path="models/neural_network_model.joblib"):
        if self.model is None:
            raise ValueError("[ERROR] No trained neural network found to export.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)
        print(f"[EXPORT] Trained MLP Neural Network successfully saved to: {file_path}")