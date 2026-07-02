import os
import time
import joblib
from src.data_cleaner import DataCleaner 

def run_inference(text: str, model_path: str, vectorizer_path: str) -> tuple:
    """
    Execute real-time inference for a single review text string.
    """
    start_time = time.perf_counter()
    
    if not isinstance(text, str) or not text.strip():
        return "Negative", 0.0, 0.0

    cleaner = DataCleaner()

    cleaned_text = cleaner.clean_text(text)

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Production model or vectorizer file not found.")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = float(max(probabilities))
    
    label = "Positive" if str(prediction) in ("1", "positive", "pos") else "Negative"
    
    latency = time.perf_counter() - start_time
    
    return label, confidence, latency