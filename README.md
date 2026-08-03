# E-Commerce Review Sentiment Analysis System

An end-to-end Machine Learning system that automatically classifies customer reviews from e-commerce platforms (such as Amazon) into **Positive** or **Negative** sentiments. Built with Python, Scikit-learn, and Streamlit, this project features a full data pipeline and an interactive web dashboard for real-time analysis.

---

## 📌 Project Overview

Customer feedback is critical for online businesses. However, reading thousands of reviews manually is difficult and time-consuming. This system automates the process by:
- Cleaning and preprocessing raw review text.
- Transforming text into numerical feature vectors using **TF-IDF Vectorization**.
- Classifying sentiment using an optimized **Multi-Layer Perceptron (MLP)** Neural Network alongside baseline classifiers.
- Displaying interactive metrics, batch analysis, and data visualizations on a **Streamlit Web Dashboard**.

---

## ✨ Key Features

- **Automated Text Preprocessing**: Handles noise removal, contraction expansion, stopword filtering, and negation context processing.
- **Feature Extraction**: Converts text into high-dimensional TF-IDF vectors with N-gram support.
- **Neural Network Classification**: Utilizes an MLP model tuned for high accuracy on text datasets.
- **Interactive Streamlit Dashboard**:
  - **Overview**: Displays system status, datasets, and speed metrics.
  - **Single Inference**: Instant sentiment prediction with confidence scores and latency tracking.
  - **Batch Processing**: Upload CSV files to analyze thousands of customer reviews at once.
  - **Data Visualization**: Explore label distributions, top word frequencies, and word clouds.
  - **Model Performance**: View the interactive confusion matrix and metrics (Accuracy, Precision, Recall, F1-Score).

---

## 📂 Project Structure

```text
review-classify-ai/
├── data/
│   ├── processed/          # Cleaned dataset (clean_data.csv, matrix_sample.csv)
│   └── raw/                # Raw datasets (Amazon_Reviews.csv, test samples)
├── models/                 # Saved model weights, vectorizer, and metrics JSON
├── notebooks/              # Jupyter notebook for Exploratory Data Analysis (EDA)
├── src/                    # Source code modules
│   ├── classifier.py       # Model training, evaluation, and feature extraction
│   ├── data_cleaner.py     # Text cleaning and negation logic
│   ├── data_pipeline.py    # Data loading and preprocessing pipeline
│   ├── models.py           # Real-time inference functions
│   ├── ui_charts.py        # Plotly chart components for Streamlit
│   ├── ui_pages.py         # Dashboard page views and layouts
│   ├── ui_styles.py        # Custom CSS styling
│   └── vectorizer.py       # TF-IDF feature extraction module
├── app.py                  # Streamlit web application entry point
├── main.py                 # Pipeline execution and model training script
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies list
```

---

## 🚀 Quick Start Guide

Follow these steps to set up and run the project locally.

### 1. Prerequisites
Ensure you have **Python 3.10** or higher installed on your computer.

### 2. Clone the Repository
```bash
git clone https://github.com/nhnhatminh/review-classify-ai.git
cd review-classify-ai
```

### 3. Create and Activate a Virtual Environment
```bash
# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Train the Model (Run Pipeline)
Run `main.py` to process raw data, train the model, evaluate performance, and save the artifacts in the `models/` directory:
```bash
python main.py
```

### 6. Launch the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
Open your browser and go to `http://localhost:8501`.

---

## 📊 Model Performance

The trained Multi-Layer Perceptron (MLP) classifier achieves strong performance across test metrics:
- **Accuracy**: ~94%
- **Precision**: High precision in identifying positive reviews correctly.
- **Recall**: Effective detection of negative customer complaints.
- **F1-Score**: High balance between precision and recall.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, NumPy, Pandas, Joblib
- **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **Web Framework**: Streamlit

## 👤 Author

**Nguyen Huynh Nhat Minh**
* **GitHub:** [@nhnhatminh](https://github.com/nhnhatminh)