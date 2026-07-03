import os, sys, json, time, pickle, pathlib, warnings, re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import joblib
from src.data_cleaner import TextCleaner

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

ROOT            = pathlib.Path(os.getcwd())
DATA_RAW        = ROOT / "data"  / "raw" / "Amazon_Reviews.csv"
DATA_CLEAN      = ROOT / "data"  / "processed" / "clean_data.csv"
MODEL_PATH      = ROOT / "models" / "neural_network_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
METRICS_PATH    = ROOT / "models" / "metrics.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="ReviewClassifyAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "dark_mode":     False,
    "history":       [],
    "active_page":   "Tổng quan",
    "batch_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
def T():
    if st.session_state.dark_mode:
        return dict(
            bg="#0F172A", surface="#1E293B", surface2="#263348",
            border="#334155", text="#F1F5F9", muted="#94A3B8",
            primary="#3B82F6", secondary="#6366F1",
            success="#22C55E", danger="#EF4444",
            shadow="0 4px 24px rgba(0,0,0,.45)",
            nav_bg="#1E293B", nav_hover="rgba(59,130,246,.15)",
        )
    return dict(
        bg="#F0F4F8", surface="#FFFFFF", surface2="#F8FAFC",
        border="#E2E8F0", text="#1E293B", muted="#64748B",
        primary="#2563EB", secondary="#4F46E5",
        success="#22C55E", danger="#EF4444",
        shadow="0 2px 12px rgba(0,0,0,.06)",
        nav_bg="#FFFFFF", nav_hover="rgba(37,99,235,.08)",
    )

def get_icon(name, size=16, color="currentColor"):
    icons = {
        "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
        "mouse-pointer-click": '<path d="M14 4.1 12 6"/><path d="m5.1 8-2.9-.8"/><path d="m6 12-1.9 2"/><path d="M7.2 2.2 8 5.1"/><path d="M9.037 9.69a.498.498 0 0 1 .653-.653l11 4.5a.5.5 0 0 1-.074.949l-4.349 1.041a1 1 0 0 0-.74.739l-1.04 4.35a.5.5 0 0 1-.95.074z"/>',
        "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
        "bar-chart-2": '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
        "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
        "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "x-circle": '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
        "brain": '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>',
        "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    }
    path = icons.get(name, "")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; flex-shrink:0;">{path}</svg>'

def get_icon_data_uri(name, size=18, color="%236B7280"):
    raw_svg = get_icon(name, size, color="COLORPLACEHOLDER")
    svg = raw_svg.replace("COLORPLACEHOLDER", color)
    svg = svg.replace("#", "%23").replace('"', "'")
    return f"data:image/svg+xml,{svg}"

NAV_PAGES = {
    "Tổng quan": ("overview", "layout-dashboard"),
    "Phân tích đơn lẻ": ("single", "mouse-pointer-click"),
    "Phân tích hàng loạt": ("batch", "layers"),
    "Trực quan dữ liệu": ("viz", "bar-chart-2"),
    "Hiệu suất mô hình": ("performance", "activity"),
    "Thông tin dữ liệu": ("info", "file-text"),
}

def inject_css():
    t = T()
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 2rem 2.5rem 4rem !important; max-width: 1280px; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {t['nav_bg']} !important;
        border-right: 1px solid {t['border']};
        min-width: 260px !important;
    }}
    section[data-testid="stSidebar"] > div {{ padding: 1.2rem 1rem; }}

    /* ── Header ── */
    .hdr {{
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 55%, #4F46E5 100%);
        border-radius: 20px; padding: 2rem 2.6rem; margin-bottom: 2rem;
        box-shadow: 0 16px 50px rgba(37,99,235,.22);
        position: relative; overflow: hidden;
    }}
    .hdr::before {{
        content: ''; position: absolute; top: -60px; right: -60px;
        width: 200px; height: 200px; border-radius: 50%;
        background: rgba(255,255,255,.07); pointer-events: none;
    }}
    .hdr-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
        color: #fff; font-size: .66rem; font-weight: 600; letter-spacing: 1px;
        text-transform: uppercase; padding: 4px 12px;
        border-radius: 99px; margin-bottom: .85rem;
    }}
    .hdr-title {{ font-size: 2rem; font-weight: 800; color: #fff; margin: 0 0 .25rem; line-height: 1.2; }}
    .hdr-sub   {{ font-size: .95rem; color: rgba(255,255,255,.85); margin: 0; }}

    /* ── Sidebar Logo ── */
    .sidebar-logo {{
        display: flex; align-items: center; gap: .75rem;
        padding: .5rem .5rem 1.2rem; border-bottom: 1px solid {t['border']}; margin-bottom: 1.2rem;
    }}
    .sidebar-logo-icon {{
        width: 38px; height: 38px; border-radius: 10px;
        background: linear-gradient(135deg, #2563EB, #4F46E5);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }}
    .sidebar-logo-text {{ font-size: .95rem; font-weight: 700; color: {t['text']}; line-height: 1.2; }}
    .sidebar-logo-sub  {{ font-size: .7rem; color: {t['muted']}; }}

    /* ── Nav Sections ── */
    .nav-section-label {{
        font-size: .65rem; font-weight: 700; letter-spacing: 1.3px;
        text-transform: uppercase; color: {t['muted']};
        padding: 0 .5rem; margin: 0 0 .6rem;
    }}

    /* ── THIẾT KẾ LẠI NÚT ĐIỀU HƯỚNG TỐI ƯU HƠN ── */
    section[data-testid="stSidebar"] div[data-testid="stButton"] {{
        margin-bottom: .2rem !important; 
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: .5rem 1rem .5rem 2.6rem !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        transition: all .2s ease !important;
        font-size: .85rem !important;
        font-weight: 500 !important;
        color: {t['text']} !important;
        background-color: transparent !important;
        background-repeat: no-repeat !important;
        background-position: .8rem center !important;
        background-size: 16px 16px !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        width: 100% !important;
        text-align: left !important;
        height: auto !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
        background-color: {t['nav_hover']} !important;
        color: {t['primary']} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus:not(:active) {{
        box-shadow: none !important;
    }}

    /* ── Info Banners ── */
    .ib {{ background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB;
           border-radius: 10px; padding: .8rem 1rem; font-size: .85rem; color: #1E40AF; margin-bottom: 1rem; }}
    .wb {{ background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B;
           border-radius: 10px; padding: .8rem 1rem; font-size: .85rem; color: #92400E; margin-bottom: 1rem; }}

    /* ── Headings ── */
    .sh {{
        font-size: .85rem; font-weight: 700; color: {t['text']};
        text-transform: uppercase; letter-spacing: 1.1px;
        margin: 0 0 1rem; padding-bottom: .5rem;
        border-bottom: 2px solid {t['border']};
    }}

    /* ── Overview Cards ── */
    .card {{
        background: {t['surface']}; border-radius: 14px; padding: 1.25rem 1.3rem;
        box-shadow: {t['shadow']}; border: 1px solid {t['border']};
        transition: transform .25s, box-shadow .25s; height: 100%;
    }}
    .card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 28px rgba(37,99,235,.11); }}
    .card-lbl {{ font-size: .65rem; font-weight: 600; letter-spacing: 1px;
                 text-transform: uppercase; color: {t['muted']}; margin-bottom: .4rem; display: flex; align-items: center; gap: 6px; }}
    .card-val {{ font-size: 1.1rem; font-weight: 700; color: {t['text']}; line-height: 1.3; word-break: break-all; }}

    /* ── Metric Cards ── */
    .mc {{
        background: {t['surface']}; border-radius: 14px; padding: 1.4rem 1rem;
        box-shadow: {t['shadow']}; border: 1px solid {t['border']};
        text-align: center; transition: transform .25s;
    }}
    .mc:hover {{ transform: translateY(-3px); box-shadow: 0 10px 26px rgba(37,99,235,.1); }}
    .mc-name {{ font-size: .65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: {t['muted']}; margin-bottom: .5rem; }}
    .mc-val  {{ font-size: 1.9rem; font-weight: 800; background: linear-gradient(135deg,#2563EB,#4F46E5);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }}

    /* ── Predictions ── */
    @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(10px) }} to {{ opacity:1; transform:translateY(0) }} }}
    .res       {{ border-radius: 14px; padding: 1.5rem 1.6rem; animation: fadeUp .35s ease; border: 1.5px solid transparent; }}
    .res.pos   {{ background: linear-gradient(135deg,#F0FDF4,#DCFCE7); border-color: #86EFAC; box-shadow: 0 4px 18px rgba(34,197,94,.13); }}
    .res.neg   {{ background: linear-gradient(135deg,#FFF5F5,#FEE2E2); border-color: #FCA5A5; box-shadow: 0 4px 18px rgba(239,68,68,.13); }}
    .res-lbl   {{ font-size: .65rem; font-weight: 600; letter-spacing: 1.1px; text-transform: uppercase; margin-bottom: .4rem; display: flex; align-items: center; gap: 5px; }}
    .res-lbl.pos {{ color: #16A34A; }} .res-lbl.neg {{ color: #DC2626; }}
    .res-val   {{ font-size: 1.5rem; font-weight: 800; line-height: 1.1; }}
    .res-val.pos {{ color: #15803D; }} .res-val.neg {{ color: #B91C1C; }}
    .conf-wrap {{ margin-top: 1rem; background: rgba(0,0,0,.08); border-radius: 99px; height: 6px; overflow: hidden; }}
    .conf-fill {{ height: 100%; border-radius: 99px; transition: width .6s ease; }}
    .conf-fill.pos {{ background: linear-gradient(90deg,#22C55E,#16A34A); }}
    .conf-fill.neg {{ background: linear-gradient(90deg,#EF4444,#B91C1C); }}

    /* ── Tables & Charts ── */
    .hw {{ background: {t['surface']}; border-radius: 14px; padding: 1.4rem; box-shadow: {t['shadow']}; border: 1px solid {t['border']}; }}
    table.ht {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
    table.ht thead th  {{ border-bottom: 2px solid {t['border']}; padding: .6rem .4rem; text-align: left; font-size: .65rem; text-transform: uppercase; letter-spacing: .8px; color: {t['muted']}; font-weight: 600; }}
    table.ht tbody tr  {{ border-bottom: 1px solid {t['border']}; transition: background .12s; }}
    table.ht tbody tr:hover {{ background: {t['surface2']}; }}
    table.ht tbody td  {{ padding: .7rem .4rem; color: {t['text']}; vertical-align: middle; }}
    .bp {{ background:#DCFCE7; color:#15803D; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; display: inline-flex; align-items: center; gap: 4px; }}
    .bn {{ background:#FEE2E2; color:#B91C1C; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; display: inline-flex; align-items: center; gap: 4px; }}

    /* ── Summary & Data Info ── */
    .sb {{ display: flex; gap: 1rem; flex-wrap: wrap; background: {t['surface2']}; border-radius: 12px; padding: 1rem 1.4rem; border: 1px solid {t['border']}; margin-bottom: 1.2rem; }}
    .si {{ flex: 1; min-width: 90px; }}
    .sl {{ font-size: .65rem; text-transform: uppercase; letter-spacing: .9px; color: {t['muted']}; font-weight: 600; margin-bottom: .25rem; display: flex; align-items: center; gap: 4px; }}
    .sv {{ font-size: 1.25rem; font-weight: 700; color: {t['text']}; }}

    .cc {{ background: {t['surface']}; border-radius: 14px; padding: 1.4rem; box-shadow: {t['shadow']}; border: 1px solid {t['border']}; margin-bottom: 1.2rem; }}
    .ct {{ font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: {t['text']}; margin-bottom: 1rem; border-bottom: 1px solid {t['border']}; padding-bottom: .5rem; }}

    .dv {{ height: 1px; background: {t['border']}; margin: 2rem 0; }}
    .ft {{ margin-top: 3rem; padding: 1.2rem 0; border-top: 1px solid {t['border']}; text-align: center; font-size: .8rem; color: {t['muted']}; }}

    /* ── Streamlit Overrides ── */
    .main div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg,#2563EB 0%,#4F46E5 100%) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        padding: .6rem 1.6rem !important; font-size: .86rem !important; font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,.28) !important;
        transition: opacity .2s, transform .15s, box-shadow .2s !important; width: 100%;
    }}
    .main div[data-testid="stButton"] > button:hover {{ opacity: .9 !important; transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(37,99,235,.36) !important; }}
    div[data-testid="stDownloadButton"] > button {{ background: linear-gradient(135deg,#2563EB 0%,#4F46E5 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: .86rem !important; box-shadow: 0 4px 12px rgba(37,99,235,.28) !important; }}
    div[data-testid="stTextArea"] textarea {{ border-radius: 10px !important; border: 1.5px solid {t['border']} !important; font-family: 'Inter', sans-serif !important; font-size: .9rem !important; color: {t['text']} !important; background: {t['surface2']} !important; padding: 1rem !important; transition: border-color .2s, box-shadow .2s !important; }}
    div[data-testid="stTextArea"] textarea:focus {{ border-color: {t['primary']} !important; box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important; background: {t['surface']} !important; outline: none !important; }}
    div[data-testid="stTabs"] button {{ font-family: 'Inter', sans-serif !important; font-size: .88rem !important; font-weight: 500 !important; color: {t['muted']} !important; background: transparent !important; border: none !important; box-shadow: none !important; transform: none !important; padding: .6rem 1rem !important; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {t['primary']} !important; font-weight: 700 !important; border-bottom: 2px solid {t['primary']} !important; box-shadow: none !important; }}
    div[data-testid="stToggle"] label {{ color: {t['text']} !important; font-size: .86rem !important; font-weight: 500 !important; }}

    /* ── Sidebar Session Box ── */
    .session-box {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1rem; margin-top: .5rem; box-shadow: 0 2px 8px rgba(0,0,0,.04); }}
    .session-row {{ display: flex; justify-content: space-between; align-items: center; font-size: .82rem; color: {t['muted']}; font-weight: 500; }}
    .session-val {{ font-weight: 700; color: {t['text']}; font-size: .9rem; }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_dataset(path: pathlib.Path):
    if not path.exists(): return None
    try: return pd.read_csv(path)
    except Exception: return None

@st.cache_data(show_spinner=False)
def load_metrics(path: pathlib.Path):
    if not path.exists(): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return None

@st.cache_resource(show_spinner=False)
def load_model(mp: pathlib.Path, vp: pathlib.Path):
    if not mp.exists() or not vp.exists(): 
        return None, None
    try:
        model      = joblib.load(mp)
        vectorizer = joblib.load(vp)
        return model, vectorizer
    except Exception: 
        return None, None

try:
    from src.models import run_inference as _core_inference
    _INFERENCE_OK = True
except ImportError:
    _INFERENCE_OK = False

def predict_single(text: str, model, vectorizer) -> tuple:
    if _INFERENCE_OK:
        label, prob, lat = _core_inference(text, model_path=str(MODEL_PATH), vectorizer_path=str(VECTORIZER_PATH))
        if label is not None:
            return label, float(prob), round(float(lat), 4)
            
    if model and vectorizer:
        t0 = time.perf_counter()
        
        cleaner = TextCleaner()
        if hasattr(cleaner, 'clean_text'):
            cleaned_text = cleaner.clean_text(text)
        elif hasattr(cleaner, 'clean'):
            cleaned_text = cleaner.clean(text)
        else:
            cleaned_text = text.lower()
            
        vec = vectorizer.transform([cleaned_text])
        proba = model.predict_proba(vec)[0]
        pred  = model.predict(vec)[0]
        lat   = time.perf_counter() - t0
        
        label = "Positive" if str(pred).lower() in ("1", "positive", "pos") else "Negative"
        return label, float(max(proba)), round(lat, 4)
        
    return "N/A", 0.0, 0.0

def predict_batch(texts: list, model, vectorizer) -> list:
    results = []
    for txt in texts:
        if not isinstance(txt, str) or not txt.strip():
            results.append(("N/A", 0.0, 0.0))
        else:
            results.append(predict_single(txt, model, vectorizer))
    return results

STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","is","it","this","that","was","are","be","with","as","by","from","they","we","i","my","your","me","he","she","his","her","its","our","have","had","has","do","did","not","so","if","up","all","also","will","just","can","more","been","than","then","there","their","out","would","could","what","which","who","how","no","one","about","when","into","very","too","am","were","being"}

def detect_text_col(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == object and df[col].dropna().str.len().mean() > 12: return col
    return None

def detect_label_col(df: pd.DataFrame):
    priority = ("label","sentiment","target","target label","class")
    for col in df.columns:
        if col.lower() in priority and df[col].nunique() <= 5: return col
    for col in df.columns:
        if df[col].nunique() <= 4 and df[col].dtype in (object, "int64", int): return col
    return None

def _base_layout(h=320):
    t = T()
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=t["text"]), height=h,
        margin=dict(l=14, r=14, t=14, b=14),
    )

def _axis_style():
    t = T()
    return dict(gridcolor=t["border"], tickfont=dict(size=11, color=t["muted"]))

def chart_distribution(df: pd.DataFrame):
    lc = detect_label_col(df)
    if not lc: return None
    vc = df[lc].value_counts()
    norm   = lambda x: "Positive" if str(x).lower() in ("1","positive","pos") else "Negative"
    labels = [norm(l) for l in vc.index]
    colors = ["#22C55E" if l == "Positive" else "#EF4444" for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=vc.values, marker_color=colors, marker_line_width=0,
        text=vc.values, textposition="outside", textfont=dict(size=12, color=T()["text"]),
        hovertemplate="<b>%{x}</b><br>Số lượng: %{y:,}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(300))
    fig.update_xaxes(**_axis_style(), showgrid=False)
    fig.update_yaxes(**_axis_style())
    return fig

def chart_word_freq(df: pd.DataFrame):
    tc = detect_text_col(df)
    if not tc: return None
    words = []
    for row in df[tc].dropna():
        words.extend(w for w in re.findall(r"\b[a-z]{3,}\b", str(row).lower()) if w not in STOP)
    if not words: return None
    top20 = Counter(words).most_common(20)
    lbls  = [w for w, _ in top20][::-1]
    vals  = [c for _, c in top20][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=lbls, orientation="h",
        marker=dict(color=vals, colorscale=[[0,"#BFDBFE"],[1,"#1E40AF"]], showscale=False),
        marker_line_width=0, text=vals, textposition="outside", textfont=dict(size=10, color=T()["text"]),
        hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(500))
    fig.update_xaxes(**_axis_style())
    fig.update_yaxes(**_axis_style(), showgrid=False)
    return fig

def chart_conf_hist(history: list):
    if not history: return None
    fig = go.Figure()
    for lbl, clr in [("Positive", "#22C55E"), ("Negative", "#EF4444")]:
        confs = [h["confidence"] for h in history if h["label"] == lbl]
        if confs:
            fig.add_trace(go.Histogram(
                x=confs, name=lbl, nbinsx=10, marker_color=clr, opacity=.75,
                hovertemplate="Confidence: %{x:.0%}<br>Số: %{y}<extra></extra>",
            ))
    fig.update_layout(**_base_layout(280), barmode="overlay", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=11)))
    fig.update_xaxes(**_axis_style(), tickformat=".0%")
    fig.update_yaxes(**_axis_style())
    return fig

def chart_trend(history: list):
    if len(history) < 2: return None
    p = n = 0
    pr, nr = [], []
    for h in history:
        if h["label"] == "Positive": p += 1
        else: n += 1
        pr.append(p); nr.append(n)
    x = list(range(1, len(history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pr, name="Positive", mode="lines+markers", line=dict(color="#22C55E", width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x, y=nr, name="Negative", mode="lines+markers", line=dict(color="#EF4444", width=2.5), marker=dict(size=5)))
    fig.update_layout(**_base_layout(280), legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=11)))
    fig.update_xaxes(**_axis_style(), title="Lần dự đoán")
    fig.update_yaxes(**_axis_style(), title="Tích lũy")
    return fig

def chart_confusion(cm):
    z   = np.array(cm)
    x_labels = ["Dự đoán Tiêu cực", "Dự đoán Tích cực"]
    y_labels = ["Thực tế Tiêu cực",  "Thực tế Tích cực"]
    
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[[0,"#EFF6FF"],[0.5,"#2563EB"],[1,"#1E40AF"]], showscale=False,
        hovertemplate="Thực tế: %{y}<br>Dự đoán: %{x}<br>Số: %{z}<extra></extra>",
    ))
    
    max_val = z.max()
    annotations = []
    for i in range(2):
        for j in range(2):
            val = z[i, j]

            color = "white" if val > max_val * 0.2 else "#1E293B"
            annotations.append(dict(
                x=x_labels[j], y=y_labels[i],
                text=f"{val:,}",
                font=dict(family="Inter", size=18, color=color, weight="bold"),
                showarrow=False,
                xref="x1", yref="y1"
            ))
            
    fig.update_layout(**_base_layout(320))
    fig.update_layout(annotations=annotations)
    
    fig.update_xaxes(side="bottom", showgrid=False, tickfont=dict(size=12, family="Inter", color=T()["muted"]))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, family="Inter", color=T()["muted"]))
    return fig

def chart_wordcloud(df: pd.DataFrame):
    try: from wordcloud import WordCloud
    except ImportError: return None
    tc = detect_text_col(df)
    if not tc: return None
    corpus = " ".join(df[tc].dropna().astype(str).tolist())
    if not corpus.strip(): return None

    font_path = None
    if os.name == "nt":
        for f in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/calibri.ttf"]:
            if os.path.exists(f):
                font_path = f
                break
                
    if font_path is None:
        try:
            import matplotlib.font_manager as fm
            font_path = fm.findfont(fm.FontProperties(family="sans-serif"))
        except:
            pass

    try:
        wc = WordCloud(width=840, height=320, background_color="white", font_path=font_path,
                       colormap="Blues", max_words=120, prefer_horizontal=.85).generate(corpus)
    except Exception: 
        try:
            wc = WordCloud(width=840, height=320, background_color="white",
                           colormap="Blues", max_words=120, prefer_horizontal=.85).generate(corpus)
        except Exception:
            return None

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0)
    return fig

def chart_batch_pie(df_out: pd.DataFrame):
    vc     = df_out["Dự đoán"].value_counts()
    colors = ["#22C55E" if l == "Positive" else "#EF4444" for l in vc.index]
    fig = go.Figure(go.Pie(
        labels=vc.index, values=vc.values,
        marker=dict(colors=colors, line=dict(color=T()["surface"], width=2)),
        textinfo="label+percent", textfont=dict(size=12, family="Inter", color="#fff"),
        hole=.45, hovertemplate="<b>%{label}</b><br>%{value} đánh giá<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(280), showlegend=False)
    return fig

def render_sidebar():
    t = T()
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">
                {get_icon('brain', 22, 'white')}
            </div>
            <div>
                <div class="sidebar-logo-text">ReviewClassifyAI</div>
                <div class="sidebar-logo-sub">Sentiment Analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-section-label">ĐIỀU HƯỚNG</div>', unsafe_allow_html=True)
        
        for page, (slug, icon_name) in NAV_PAGES.items():
            clicked = st.button(page, key=f"nav_{slug}")
            if clicked and st.session_state.active_page != page:
                st.session_state.active_page = page
                st.rerun()

        active_page = st.session_state.active_page
        active_slug = NAV_PAGES[active_page][0]
        icon_rules = []
        for page, (slug, icon_name) in NAV_PAGES.items():
            is_active = page == active_page
            icon_color = "%23FFFFFF" if is_active else "%2364748B"
            icon_uri = get_icon_data_uri(icon_name, 16, icon_color)
            icon_rules.append(f"""
            section[data-testid="stSidebar"] div[data-testid="stButton"]:has(button#nav_{slug}) > button {{
                background-image: url("{icon_uri}") !important;
            }}""")
            
        t2 = T()
        st.markdown(f"""
        <style>
        {''.join(icon_rules)}
        section[data-testid="stSidebar"] div[data-testid="stButton"]:has(button#nav_{active_slug}) > button {{
            background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
            color: #FFFFFF !important;
            border-color: transparent !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 10px rgba(37,99,235,0.3) !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="height:1px;background:{t["border"]};margin:1.2rem 0;"></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="nav-section-label">TÙY CHỈNH</div>', unsafe_allow_html=True)
        dark = st.toggle("Chế độ tối", value=st.session_state.dark_mode, key="tgl_dark")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.markdown(f'<div style="height:1px;background:{t["border"]};margin:1.2rem 0;"></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="nav-section-label">PHIÊN LÀM VIỆC</div>', unsafe_allow_html=True)
        h = st.session_state.history
        total = len(h)
        pos   = sum(1 for x in h if x["label"] == "Positive")
        neg   = total - pos
        st.markdown(f"""
        <div class="session-box">
            <div class="session-row">
                <span>Tổng dự đoán</span>
                <span class="session-val">{total}</span>
            </div>
            <div class="session-row" style="margin-top:.6rem;">
                <span style="display:flex;align-items:center;gap:6px;">{get_icon('check-circle', 14, '#22C55E')} Tích cực</span>
                <span class="session-val" style="color:#22C55E;">{pos}</span>
            </div>
            <div class="session-row" style="margin-top:.4rem;">
                <span style="display:flex;align-items:center;gap:6px;">{get_icon('x-circle', 14, '#EF4444')} Tiêu cực</span>
                <span class="session-val" style="color:#EF4444;">{neg}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_header():
    st.markdown(f"""
    <div class="hdr">
        <div class="hdr-badge"> {get_icon('brain', 14, 'white')} Hệ thống Phân loại Cảm xúc</div>
        <div class="hdr-title">ReviewClassifyAI</div>
        <div class="hdr-sub">Bảng điều khiển phân loại cảm xúc đánh giá sản phẩm thương mại điện tử</div>
    </div>""", unsafe_allow_html=True)

def render_overview(raw_df, clean_df, model, vectorizer, metrics):
    st.markdown('<div class="sh">Tổng quan hệ thống</div>', unsafe_allow_html=True)
    
    active_df = raw_df if raw_df is not None else clean_df
    ds = f"{len(active_df):,} dòng" if active_df is not None else "Không tìm thấy"
    
    mn = type(model).__name__          if model       is not None else "Chưa tải"
    vn = type(vectorizer).__name__     if vectorizer  is not None else "Chưa tải"
    ps = f"~{metrics.get('prediction_speed','N/A')}s"   if metrics else "N/A"

    for col, (lbl, val, ic) in zip(st.columns(4, gap="medium"), [
        ("Tập dữ liệu", ds, "database"), ("Mô hình", mn, "brain"),
        ("Vector hóa", vn, "cpu"), ("Tốc độ dự đoán", ps, "zap")
    ]):
        with col:
            st.markdown(f'<div class="card"><div class="card-lbl">{get_icon(ic, 14)} {lbl}</div><div class="card-val">{val}</div></div>', unsafe_allow_html=True)

def render_single_analysis(model, vectorizer):
    st.markdown('<div class="sh">Phân tích cảm xúc trực tuyến</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        txt = st.text_area("", placeholder="Nhập đánh giá sản phẩm bằng tiếng Anh...", height=160, label_visibility="collapsed", key="single_input")
        c1, c2 = st.columns([2, 1], gap="small")
        with c1: run = st.button("Phân tích đánh giá", use_container_width=True, key="btn_run")
        with c2: clr = st.button("Xóa lịch sử",        use_container_width=True, key="btn_clr")
        if clr:
            st.session_state.history = []
            st.rerun()
    with right:
        if run:
            if not txt.strip():
                st.markdown('<div class="ib">Vui lòng nhập nội dung đánh giá trước.</div>', unsafe_allow_html=True)
            elif model is None:
                st.markdown('<div class="wb">Chưa tải được mô hình. Kiểm tra file model.pkl và vectorizer.pkl trong thư mục model/.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Đang phân tích..."):
                    label, prob, lat = predict_single(txt, model, vectorizer)
                    st.session_state.history.append({
                        "text":       txt[:80] + ("…" if len(txt) > 80 else ""),
                        "label":      label, "confidence": prob, "latency":    lat,
                        "timestamp":  time.strftime("%H:%M:%S"),
                    })
                    sc     = "pos" if label == "Positive" else "neg"
                    lbl_vi = "Tích cực" if label == "Positive" else "Tiêu cực"
                    ic_res = "check-circle" if label == "Positive" else "x-circle"
                    st.markdown(f"""
                    <div class="res {sc}">
                        <div class="res-lbl {sc}">Kết quả dự đoán</div>
                        <div class="res-val {sc}" style="display:flex; align-items:center; gap:8px;">{get_icon(ic_res, 28)} {lbl_vi} ({label})</div>
                        <div class="conf-wrap"><div class="conf-fill {sc}" style="width:{int(prob*100)}%;"></div></div>
                    </div>""", unsafe_allow_html=True)
                    ca, cb = st.columns(2)
                    with ca: st.markdown(f'<div class="card" style="margin-top:.8rem; padding: 1rem;"><div class="card-lbl">Độ tin cậy</div><div class="card-val">{prob:.1%}</div></div>', unsafe_allow_html=True)
                    with cb: st.markdown(f'<div class="card" style="margin-top:.8rem; padding: 1rem;"><div class="card-lbl">Độ trễ</div><div class="card-val">{lat:.4f}s</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ib">Nhập đánh giá bên trái rồi nhấn "Phân tích đánh giá".</div>', unsafe_allow_html=True)

def render_history():
    h = st.session_state.history
    if not h:
        st.markdown('<div class="ib">Chưa có dự đoán nào. Hãy phân tích một đánh giá để xem lịch sử.</div>', unsafe_allow_html=True)
        return
    total = len(h)
    pos   = sum(1 for x in h if x["label"] == "Positive")
    avg_c = sum(x["confidence"] for x in h) / total
    avg_l = sum(x["latency"]    for x in h) / total
    st.markdown(f"""
    <div class="sb">
        <div class="si"><div class="sl">{get_icon('layers',12)} Tổng số</div><div class="sv">{total}</div></div>
        <div class="si"><div class="sl">{get_icon('check-circle',12)} Tích cực</div><div class="sv" style="color:#22C55E;">{pos}</div></div>
        <div class="si"><div class="sl">{get_icon('x-circle',12)} Tiêu cực</div><div class="sv" style="color:#EF4444;">{total-pos}</div></div>
        <div class="si"><div class="sl">{get_icon('activity',12)} TB Tin cậy</div><div class="sv">{avg_c:.1%}</div></div>
        <div class="si"><div class="sl">{get_icon('zap',12)} TB Độ trễ</div><div class="sv">{avg_l:.4f}s</div></div>
    </div>""", unsafe_allow_html=True)

    hc, lt = chart_conf_hist(h), chart_trend(h)
    if hc or lt:
        col1, col2 = st.columns(2, gap="large")
        if hc:
            with col1:
                st.markdown('<div class="cc"><div class="ct">Phân bố độ tin cậy</div>', unsafe_allow_html=True)
                st.plotly_chart(hc, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
        if lt:
            with col2:
                st.markdown('<div class="cc"><div class="ct">Xu hướng tích lũy</div>', unsafe_allow_html=True)
                st.plotly_chart(lt, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

    rows = "".join(f"""<tr>
        <td>{x['timestamp']}</td>
        <td style="max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{x['text']}</td>
        <td><span class="{'bp' if x['label']=='Positive' else 'bn'}">{get_icon('check-circle' if x['label']=='Positive' else 'x-circle', 12)}
        {'Tích cực' if x['label']=='Positive' else 'Tiêu cực'}</span></td>
        <td>{x['confidence']:.1%}</td>
        <td>{x['latency']:.4f}s</td>
    </tr>""" for x in reversed(h))
    st.markdown(f'<div class="hw"><table class="ht"><thead><tr><th>Thời gian</th><th>Đánh giá</th><th>Nhãn</th><th>Độ tin cậy</th><th>Độ trễ</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

def render_batch(model, vectorizer):
    st.markdown('<div class="sh">Phân tích hàng loạt</div>', unsafe_allow_html=True)
    if model is None:
        st.markdown('<div class="wb">Chưa tải được mô hình. Kiểm tra file model.pkl và vectorizer.pkl trong thư mục model/.</div>', unsafe_allow_html=True)
        return
    up = st.file_uploader("Tải lên file CSV chứa cột đánh giá", type=["csv"], label_visibility="collapsed", key="batch_up")
    if up is None:
        st.markdown('<div class="ib">Tải lên file CSV chứa các đánh giá sản phẩm để phân tích hàng loạt.</div>', unsafe_allow_html=True)
        return
    try: df_up = pd.read_csv(up)
    except Exception as e:
        st.markdown(f'<div class="wb">Không đọc được file CSV: {e}</div>', unsafe_allow_html=True)
        return
    tc = detect_text_col(df_up)
    if tc is None:
        st.markdown('<div class="wb">Không phát hiện được cột văn bản. Đảm bảo CSV có cột chứa nội dung đánh giá.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="ib">Phát hiện cột văn bản: <strong>{tc}</strong> — {len(df_up):,} dòng</div>', unsafe_allow_html=True)
    if st.button("Chạy phân tích hàng loạt", key="btn_batch"):
        with st.spinner(f"Đang phân tích {len(df_up):,} đánh giá..."):
            res    = predict_batch(df_up[tc].tolist(), model, vectorizer)
            df_out = df_up.copy()
            df_out["Dự đoán"]    = [r[0] for r in res]
            df_out["Độ tin cậy"] = [round(r[1], 4) for r in res]
            df_out["Độ trễ (s)"] = [r[2] for r in res]
            st.session_state.batch_results = df_out

    if st.session_state.batch_results is not None:
        df_out = st.session_state.batch_results
        total  = len(df_out)
        pos    = (df_out["Dự đoán"] == "Positive").sum()
        avg_c  = df_out["Độ tin cậy"].mean()
        st.markdown(f"""
        <div class="sb">
            <div class="si"><div class="sl">{get_icon('layers',12)} Tổng số dòng</div><div class="sv">{total:,}</div></div>
            <div class="si"><div class="sl">{get_icon('check-circle',12)} Tích cực</div><div class="sv" style="color:#22C55E;">{pos:,}</div></div>
            <div class="si"><div class="sl">{get_icon('x-circle',12)} Tiêu cực</div><div class="sv" style="color:#EF4444;">{total-pos:,}</div></div>
            <div class="si"><div class="sl">{get_icon('activity',12)} TB Độ tin cậy</div><div class="sv">{avg_c:.1%}</div></div>
        </div>""", unsafe_allow_html=True)
        col1, col2 = st.columns([0.4, 0.6], gap="large")
        with col1:
            pie = chart_batch_pie(df_out)
            if pie:
                st.markdown('<div class="cc"><div class="ct">Phân bố nhãn</div>', unsafe_allow_html=True)
                st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            fh = go.Figure()
            for lbl, clr in [("Positive","#22C55E"),("Negative","#EF4444")]:
                sub = df_out[df_out["Dự đoán"] == lbl]["Độ tin cậy"]
                if not sub.empty: fh.add_trace(go.Histogram(x=sub, name=lbl, nbinsx=15, marker_color=clr, opacity=.72))
            fh.update_layout(**_base_layout(280), barmode="overlay", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=11)))
            fh.update_xaxes(**_axis_style(), tickformat=".0%")
            fh.update_yaxes(**_axis_style())
            st.markdown('<div class="cc"><div class="ct">Phân bố độ tin cậy</div>', unsafe_allow_html=True)
            st.plotly_chart(fh, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:1.4rem;">Xem trước (50 dòng đầu)</div>', unsafe_allow_html=True)
        st.dataframe(df_out.head(50), use_container_width=True, height=270)
        st.download_button("Xuất kết quả CSV", data=df_out.to_csv(index=False).encode("utf-8"), file_name="ket_qua_phan_tich.csv", mime="text/csv", key="dl_batch")

def render_visualization(raw_df, clean_df):
    st.markdown('<div class="sh">Trực quan hóa dữ liệu</div>', unsafe_allow_html=True)
    df = clean_df if clean_df is not None else raw_df
    if df is None:
        st.markdown('<div class="wb">Không tìm thấy tập dữ liệu. Đặt file raw.csv / clean.csv vào thư mục data/.</div>', unsafe_allow_html=True)
        return
    t1, t2, t3 = st.tabs(["Phân bố nhãn", "Tần suất từ vựng", "Word Cloud"])
    with t1:
        fig = chart_distribution(df)
        if fig:
            st.markdown('<div class="cc">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="wb">Không phát hiện cột nhãn.</div>', unsafe_allow_html=True)
    with t2:
        fig = chart_word_freq(df)
        if fig:
            st.markdown('<div class="cc">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="wb">Không phát hiện cột văn bản.</div>', unsafe_allow_html=True)
    with t3:
        wc = chart_wordcloud(df)
        if wc:
            st.markdown('<div class="cc">', unsafe_allow_html=True)
            st.pyplot(wc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="ib">Word Cloud không khả dụng (thiếu font chữ tương thích hoặc dữ liệu trống).</div>', unsafe_allow_html=True)

def render_performance(metrics):
    st.markdown('<div class="sh">Hiệu suất mô hình</div>', unsafe_allow_html=True)
    if metrics is None:
        st.markdown('<div class="wb">Không tìm thấy file metrics.json. Đặt file vào thư mục model/.</div>', unsafe_allow_html=True)
        return
    for col, (k, name) in zip(st.columns(4, gap="medium"), [("accuracy","Độ chính xác"),("precision","Precision"),("recall","Recall"),("f1","F1 Score")]):
        v = metrics.get(k)
        with col:
            pct  = f"{v*100:.1f}%" if isinstance(v, (int, float)) else "N/A"
            disp = f"{v:.4f}"      if isinstance(v, (int, float)) else "N/A"
            st.markdown(f'<div class="mc"><div class="mc-name">{name}</div><div class="mc-val">{disp}</div><div class="mc-pct">{pct}</div></div>', unsafe_allow_html=True)
    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, _ = st.columns([0.55, 0.45], gap="large")
        with c1:
            st.markdown('<div class="cc"><div class="ct">Ma trận nhầm lẫn (Confusion Matrix)</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_confusion(cm), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

def render_data_info(raw_df, clean_df):
    st.markdown('<div class="sh">Thông tin tập dữ liệu</div>', unsafe_allow_html=True)
    df = clean_df if clean_df is not None else raw_df
    if df is None:
        st.markdown('<div class="wb">Không tìm thấy tập dữ liệu. Đặt file raw.csv / clean.csv vào thư mục data/.</div>', unsafe_allow_html=True)
        return
    lc = detect_label_col(df)
    pos = neg = "N/A"
    if lc:
        for k, v in df[lc].value_counts().items():
            if str(k).lower() in ("1","positive","pos"): pos = f"{v:,}"
            else: neg = f"{v:,}"
    vocab = "N/A"
    tc = detect_text_col(df)
    if tc:
        wds = set()
        for row in df[tc].dropna(): wds.update(re.findall(r"\b[a-z]{2,}\b", str(row).lower()))
        vocab = f"{len(wds):,}"

    st.markdown(f"""
    <style>
    .di-card {{ background: {T()['surface']}; border-radius: 12px; padding: 1.5rem; box-shadow: {T()['shadow']}; border: 1px solid {T()['border']}; display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; transition: transform .2s; }}
    .di-card:hover {{ transform: translateY(-2px); }}
    .di-icon {{ width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .di-lbl {{ font-size: .75rem; font-weight: 600; color: {T()['muted']}; text-transform: uppercase; letter-spacing: .5px; margin-bottom: .2rem; }}
    .di-val {{ font-size: 1.4rem; font-weight: 800; color: {T()['text']}; }}
    </style>
    """, unsafe_allow_html=True)

    data_points = [
        ("Tổng số dòng", f"{len(df):,}", "layers", "#EFF6FF", "#2563EB"),
        ("Tổng số cột", str(len(df.columns)), "layout-dashboard", "#F5F3FF", "#D97706"),
        ("Mẫu tích cực", pos, "check-circle", "#DCFCE7", "#16A34A"),
        ("Mẫu tiêu cực", neg, "x-circle", "#FEE2E2", "#DC2626"),
        ("Giá trị bị thiếu", f"{int(df.isnull().sum().sum()):,}", "activity", "#F3F4F6", "#4B5563"),
        ("Kích thước từ vựng", vocab, "file-text", "#FCE7F3", "#7C3AED"),
    ]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3, c1, c2, c3]
    for i, (lbl, val, icon, bg, color) in enumerate(data_points):
        with cols[i]:
            st.markdown(f'''
            <div class="di-card">
                <div class="di-icon" style="background:{bg}; color:{color};">
                    {get_icon(icon, 20, color)}
                </div>
                <div>
                    <div class="di-lbl">{lbl}</div>
                    <div class="di-val">{val}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

def main():
    inject_css()
    raw_df            = load_dataset(DATA_RAW)
    clean_df          = load_dataset(DATA_CLEAN)
    metrics           = load_metrics(METRICS_PATH)
    model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)

    render_sidebar()
    render_header()

    page = st.session_state.active_page
    if   page == "Tổng quan":           
      render_overview(raw_df, clean_df, model, vectorizer, metrics)
    elif page == "Phân tích đơn lẻ":
        render_single_analysis(model, vectorizer)
        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">Lịch sử dự đoán</div>', unsafe_allow_html=True)
        render_history()
    elif page == "Phân tích hàng loạt": render_batch(model, vectorizer)
    elif page == "Trực quan dữ liệu":   render_visualization(raw_df, clean_df)
    elif page == "Hiệu suất mô hình":   render_performance(metrics)
    elif page == "Thông tin dữ liệu":   render_data_info(raw_df, clean_df)

    st.markdown('<div class="ft">Project &nbsp;·&nbsp; <strong>ReviewClassifyAI</strong></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()