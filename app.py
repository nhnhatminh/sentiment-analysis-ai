import os
import io
import re
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Response

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


class AspectTagger:
    def __init__(self):
        self.patterns = {
            "Logistics": re.compile(
                r"\b(deliver(y|ed|s|er)?|ship(ping|ment|ped|s)?|arrive(d|s|al)?|late|delay(ed|s)?|track(ing)?|lost|stolen|carrier|courier|mailbox|box|package)\b|mail\s*room|next\s*day|door\s*step|wrong\s*address|returned\s*to\s*sender|delivery\s*window", 
                re.IGNORECASE
            ),
            "Finance": re.compile(
                r"\b(refund(s|ed|ing)?|money|bank|charge(d|s|ng)?|price|cost(s)?|fee(s)?|pay(ment|ed|s|ing)?|bill(ed|s|ing)?|debit|credit|overcharge|scam(s)?|fraud|theft)\b|credit\s*card|gift\s*card|double\s*billed|bank\s*account|small\s*claims|billing\s*department", 
                re.IGNORECASE
            ),
            "Customer Support": re.compile(
                r"\b(rude|attitude|support|service|staff|agent(s)?|manager(s)?|supervisor(s)?|representative(s)?|ignor(ed|es|ing)|empathy|complaint(s)?)\b|customer\s*service|call\s*center|help\s*line|hung\s*up|automated\s*response|talk\s*to\s*manager", 
                re.IGNORECASE
            ),
            "Product Quality": re.compile(
                r"\b(broken|junk|defective|garbage|fail(ed|s|ure)?|tear|rust(y|ed)?|crack(ed)?|counterfeit|useless|cheap|faulty|scratch(ed|es)?)\b|stopped\s*working|incorrect\s*disc|quality\s*control|piece\s*of\s*junk|waste\s*of\s*money|dead\s*pixel", 
                re.IGNORECASE
            )
        }

    def tag_review_aspects(self, text: str) -> list:
        matched_aspects = []
        lower_text = text.lower()

        if "sd card" in lower_text or "memory card" in lower_text or "hardware" in lower_text:
            matched_aspects.append("Product Quality")
            lower_text = lower_text.replace("sd card", "").replace("memory card", "")

        for aspect, pattern in self.patterns.items():
            if aspect in matched_aspects:
                continue
            if pattern.search(lower_text):
                matched_aspects.append(aspect)
                
        return matched_aspects if matched_aspects else ["General_Issue"]

aspect_tagger = AspectTagger()


class SingleReviewRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard_interface():
    template_file_path = os.path.join("dashboard", "index.html")
    if not os.path.exists(template_file_path):
        return HTMLResponse(
            content="<h1>Dashboard Template Not Found</h1><p>Please build dashboard/index.html file.</p>", 
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

    aspect_list = aspect_tagger.tag_review_aspects(payload.text) if prediction_label == 0 else ["General_Issue"]

    return {
        "status": "success",
        "raw_text": payload.text,
        "cleaned_text": cleaned_vocal,
        "prediction": prediction_label,
        "confidence": round(confidence_score, 4),
        "aspect": aspect_list
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
    crisis_dataframe = dataframe[crisis_condition].copy()
    
    crisis_ordered = crisis_dataframe.sort_values(by='Confidence', ascending=False).head(50)
    
    crisis_output_list = []
    for _, row in crisis_ordered.iterrows():
        raw_review_content = str(row['Review Text'])
        crisis_output_list.append({
            "review_text": raw_review_content,
            "confidence": round(row['Confidence'], 4),
            "aspect": aspect_tagger.tag_review_aspects(raw_review_content)
        })

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

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)