import os
import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.text_cleaning import TextCleaner

app = FastAPI(title="ReviewClassifyAI - Core Backend Engine")

app.mount("/static", StaticFiles(directory="static"), name="static")

VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
MODEL_PATH = "models/neural_network_model.joblib"

if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
    raise RuntimeError("[CRITICAL] Serialized model files are missing! Please execute test_run.py first.")

loaded_vectorizer = joblib.load(VECTORIZER_PATH)
loaded_mlp_model = joblib.load(MODEL_PATH)
text_cleaner = TextCleaner()


class SingleReviewRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard_interface():
    template_file_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_file_path):
        return HTMLResponse(
            content="<h1>Dashboard Template Not Found</h1><p>Please build templates/index.html file.</p>", 
            status_code=404
        )
    
    with open(template_file_path, "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.post("/predict/single")
async def predict_single_review(payload: SingleReviewRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input review text cannot be blank.")

    cleaned_vocal = text_cleaner.clean_review_text(payload.text)
    
    vectorized_input = loaded_vectorizer.transform([cleaned_vocal])
    
    prediction_label = int(loaded_mlp_model.predict(vectorized_input)[0])
    
    probability_matrix = loaded_mlp_model.predict_proba(vectorized_input)[0]
    confidence_score = float(probability_matrix[prediction_label])

    return {
        "status": "success",
        "raw_text": payload.text,
        "cleaned_text": cleaned_vocal,
        "prediction": prediction_label,
        "confidence": round(confidence_score, 4)
    }


@app.post("/predict/batch")
async def predict_batch_csv_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Unsupported file extensions. Only CSV files are accepted.")

    try:
        file_bytes = await file.read()
        dataframe = pd.read_csv(
            io.BytesIO(file_bytes),
            engine='python',
            on_bad_lines='skip'
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file structures: {str(error)}")

    if 'Review Text' not in dataframe.columns:
        raise HTTPException(status_code=400, detail="Required column 'Review Text' is missing in the uploaded file.")

    dataframe.dropna(subset=['Review Text'], inplace=True)
    if dataframe.empty:
        raise HTTPException(status_code=400, detail="The provided dataset contains no valid row entities.")

    dataframe['Cleaned Text'] = dataframe['Review Text'].apply(lambda item: text_cleaner.clean_review_text(str(item)))
    dataframe = dataframe[dataframe['Cleaned Text'].str.strip() != ""]
    
    if dataframe.empty:
        raise HTTPException(status_code=400, detail="All records were fully eliminated after the noise removal phase.")

    batch_vectorized = loaded_vectorizer.transform(dataframe['Cleaned Text'].values.astype('U'))

    dataframe['Prediction'] = loaded_mlp_model.predict(batch_vectorized).astype(int)
    batch_probabilities = loaded_mlp_model.predict_proba(batch_vectorized)

    dataframe['Confidence'] = [float(batch_probabilities[index][pred]) for index, pred in enumerate(dataframe['Prediction'])]

    total_processed = len(dataframe)
    negative_instances = int((dataframe['Prediction'] == 0).sum())
    positive_instances = int((dataframe['Prediction'] == 1).sum())

    positive_ratio = round((positive_instances / total_processed) * 100, 2)
    negative_ratio = round((negative_instances / total_processed) * 100, 2)

    crisis_condition = (dataframe['Prediction'] == 0) & (dataframe['Confidence'] >= 0.95)
    crisis_dataframe = dataframe[crisis_condition]
    
    crisis_ordered = crisis_dataframe.sort_values(by='Confidence', ascending=False)
    crisis_output_list = crisis_ordered[['Review Text', 'Confidence']].head(50).to_dict('records')

    return {
        "status": "success",
        "telemetry": {
            "total_records": total_processed,
            "positive_count": positive_instances,
            "negative_count": negative_instances,
            "positive_percentage": positive_ratio,
            "negative_percentage": negative_ratio
        },
        "crisis_queue": crisis_output_list
    }