import os
import sys
import pathlib
import time
import json
import pandas as pd
import streamlit as st
import joblib
from src.data_cleaner import TextCleaner

# Thêm đường dẫn gốc hệ thống
ROOT = pathlib.Path(os.getcwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui_styles import inject_css
from src.ui_pages import (
    render_sidebar,
    render_header,
    render_overview,
    render_single_analysis,
    render_history,
    render_batch,
    render_visualization,
    render_performance,
    render_data_info,
)

# Khai báo đường dẫn tài nguyên
DATA_RAW = ROOT / "data" / "raw" / "Amazon_Reviews.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "clean_data.csv"
MODEL_PATH = ROOT / "models" / "neural_network_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

GLOBAL_CLEANER = TextCleaner()

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="ReviewClassifyAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Khởi tạo session state mặc định
defaults = {
    "dark_mode": True,
    "history": [],
    "active_page": "Tổng quan",
    "batch_results": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data(show_spinner=False)
def load_dataset(path):
    # Nạp tập dữ liệu có cache
    if not path.exists():
        return None

    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return None


@st.cache_data(show_spinner=False)
def load_metrics(path):
    # Nạp tệp chỉ số hiệu năng
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_production_models(mp, vp):
    # Nạp mô hình vào RAM
    mlp_model = joblib.load(mp) if mp.exists() else None
    vectorizer = joblib.load(vp) if vp.exists() else None

    lr_path = mp.parent / "logistic_regression_model.joblib"
    lr_model = joblib.load(lr_path) if lr_path.exists() else None

    return mlp_model, vectorizer, lr_model


def predict_single(text, model, vectorizer, lr_model=None):
    # Dự đoán cho một câu văn
    if not model or not vectorizer:
        return "Negative", 0.0, 0.0, []

    if not isinstance(text, str) or not text.strip():
        return "Negative", 0.0, 0.0, []

    cleaner = TextCleaner()
    cleaned_text = (
        cleaner.clean(text)
        if hasattr(cleaner, "clean")
        else text.lower().strip()
    )

    start_time = time.perf_counter()
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = float(max(probabilities))
    label = "Positive" if str(prediction) == "1" else "Negative"
    latency = time.perf_counter() - start_time

    word_contributions = []

    if lr_model is not None:
        feature_names = vectorizer.get_feature_names_out()
        words_in_text = cleaned_text.split()

        for word in set(words_in_text):
            if word in feature_names:
                idx = list(feature_names).index(word)
                weight = float(lr_model.coef_[0][idx])

                if abs(weight) > 0.1:
                    word_contributions.append({
                        "word": word,
                        "weight": weight,
                    })

        word_contributions = sorted(
            word_contributions,
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )

    return label, confidence, latency, word_contributions


def predict_batch(texts, model, vectorizer):
    # Dự đoán theo lô hàng loạt
    if not model or not vectorizer:
        return [("Negative", 0.0, 0.0)] * len(texts)

    cleaned_texts = [
        GLOBAL_CLEANER.clean(text)
        if isinstance(text, str) and text.strip()
        else ""
        for text in texts
    ]

    start_time = time.perf_counter()
    vectorized_matrices = vectorizer.transform(cleaned_texts)

    probabilities = model.predict_proba(vectorized_matrices)

    total_latency = time.perf_counter() - start_time
    average_latency = round(
        total_latency / max(1, len(texts)),
        4,
    )

    OPTIMAL_THRESHOLD = 0.75
    results = []

    for proba in probabilities:
        pos_prob = float(proba[1])
        neg_prob = float(proba[0])

        if pos_prob >= OPTIMAL_THRESHOLD:
            label = "Positive"
            confidence = pos_prob
        else:
            label = "Negative"
            confidence = neg_prob if neg_prob > pos_prob else pos_prob

        results.append((label, confidence, average_latency))

    return results


def main():
    # Hàm khởi chạy ứng dụng
    inject_css()

    raw_df = load_dataset(DATA_RAW)
    clean_df = load_dataset(DATA_CLEAN)
    metrics = load_metrics(METRICS_PATH)

    mlp_model, vectorizer, lr_model = load_production_models(
        MODEL_PATH,
        VECTORIZER_PATH,
    )

    render_sidebar()
    render_header()

    page = st.session_state.active_page

    if page == "Tổng quan":
        render_overview(
            raw_df,
            clean_df,
            mlp_model,
            vectorizer,
            metrics,
        )

    elif page == "Phân tích Trực tiếp":
        render_single_analysis(
            mlp_model,
            vectorizer,
            lambda t, m, v: predict_single(
                t,
                m,
                v,
                lr_model,
            ),
        )
        st.markdown(
            '<div class="dv"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sh">Lịch sử Phân tích</div>',
            unsafe_allow_html=True,
        )
        render_history()

    elif page == "Phân tích Hàng loạt":
        render_batch(
            mlp_model,
            vectorizer,
            predict_batch,
        )

    elif page == "Trực quan hóa Dữ liệu":
        render_visualization(
            raw_df,
            clean_df,
        )

    elif page == "Hiệu suất Mô hình":
        render_performance(metrics)

    elif page == "Thông tin Dữ liệu":
        render_data_info(
            raw_df,
            clean_df,
        )

    st.markdown(
        '<div class="ft">ReviewClassifyAI Platform &nbsp;·&nbsp; '
        '<strong>Sentiment Agent</strong></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()