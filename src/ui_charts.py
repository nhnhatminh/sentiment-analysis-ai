import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from src.ui_styles import T, STOP

# Thiết lập cấu hình đồ họa
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["JetBrains Mono", "Inter", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#334155",  
    "axes.linewidth": 0.8,
    "grid.color": "#334155",      
    "grid.linewidth": 0.5,
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "axes.labelcolor": "#F1F5F9",
    "text.color": "#F1F5F9"
})

def _apply_academic_style(ax, show_grid=True):
    # Áp dụng phong cách tối giản
    ax.set_facecolor("none")
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(labelsize=9)

def chart_distribution(df, label_col):
    # Biểu đồ phân bố nhãn
    if not label_col or df is None or df.empty: 
        return None
        
    vc = df[label_col].value_counts()
    norm = lambda x: "Đánh giá Tiêu cực" if str(x).lower() in ("0", "negative", "neg") else "Đánh giá Tích cực"
    labels = [norm(l) for l in vc.index]
    
    colors = ["#FDA4AF" if "Tiêu cực" in l else "#A7F3D0" for l in labels]
    edge_colors = ["#E11D48" if "Tiêu cực" in l else "#059669" for l in labels]
    
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(labels, vc.values, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.38, zorder=3)
    
    _apply_academic_style(ax, show_grid=True)
    ax.set_ylabel("Số lượng đánh giá", fontsize=10, weight="bold")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,} mẫu",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, weight="bold", color="#F1F5F9")
                    
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_word_freq(df, text_col):
    # Biểu đồ tần suất từ
    if not text_col or df is None or df.empty: 
        return None
        
    words = []
    for row in df[text_col].dropna():
        words.extend(w for w in re.findall(r"\b[a-z]{3,}\b", str(row).lower()) if w not in STOP)
    if not words: 
        return None
        
    top20 = Counter(words).most_common(20)
    lbls  = [w for w, _ in top20][::-1]
    vals  = [c for _, c in top20][::-1]
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.barh(lbls, vals, color="#3B82F6", edgecolor="#60A5FA", linewidth=1, height=0.55, zorder=3)
    
    _apply_academic_style(ax, show_grid=True)
    ax.set_xlabel("Tần suất xuất hiện", fontsize=10, weight="bold")
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f" {width:,}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9, color="#F1F5F9", weight="bold")
                    
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_conf_hist(history):
    # Biểu đồ phân bố tin cậy
    if not history: 
        return None
        
    fig, ax = plt.subplots(figsize=(6, 3.8))
    
    for lbl, clr, edge_clr in [("Positive", "#22C55E", "#4ADE80"), ("Negative", "#EF4444", "#FCA5A5")]:
        confs = [h["confidence"] for h in history if h["label"] == lbl]
        if confs:
            ax.hist(confs, bins=12, alpha=0.8, color=clr, edgecolor=edge_clr, linewidth=0.8, label=lbl, zorder=3)
            
    _apply_academic_style(ax, show_grid=True)
    ax.set_xlabel("Độ tin cậy (Confidence)", fontsize=10)
    ax.set_ylabel("Số lượng đánh giá", fontsize=10)
    legend = ax.legend(frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color("#F1F5F9")
    
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_trend(history):
    # Biểu đồ xu hướng tích lũy
    if len(history) < 2: 
        return None
        
    p = n = 0
    pr, nr = [], []
    for h in history:
        if h["label"] == "Positive": p += 1
        else: n += 1
        pr.append(p); nr.append(n)
    x = list(range(1, len(history) + 1))
    
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(x, pr, label="Tích cực tích lũy", color="#10B981", linewidth=2.2, marker='o', markersize=4, zorder=3)
    ax.plot(x, nr, label="Tiêu cực tích lũy", color="#F43F5E", linewidth=2.2, marker='s', markersize=4, zorder=3)
    
    _apply_academic_style(ax, show_grid=True)
    ax.set_xlabel("Lượt phân tích", fontsize=10)
    ax.set_ylabel("Số lượng tích lũy", fontsize=10)
    legend = ax.legend(frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color("#F1F5F9")
    
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_confusion(cm):
    # Vẽ ma trận nhầm lẫn
    z = np.array(cm)
    x_labels = ["Dự đoán Tiêu cực", "Dự đoán Tích cực"]
    y_labels = ["Thực tế Tiêu cực",  "Thực tế Tích cực"]
    
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(z, cmap="Blues", interpolation="nearest")
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, fontsize=10, color="#F1F5F9")
    ax.set_yticklabels(y_labels, fontsize=10, color="#F1F5F9")
    
    for i in range(len(y_labels) + 1):
        ax.axhline(i - 0.5, color="#334155", linewidth=1.2)
    for j in range(len(x_labels) + 1):
        ax.axvline(j - 0.5, color="#334155", linewidth=1.2)
        
    max_val = z.max()
    for i in range(2):
        for j in range(2):
            val = z[i, j]
            color = "#F1F5F9" if val > max_val * 0.4 else "#94A3B8"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=15, weight="bold")
            
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_wordcloud(df, text_col):
    # Tạo đám mây từ vựng
    try: 
        from wordcloud import WordCloud
    except ImportError: 
        return None
        
    corpus = " ".join(df[text_col].dropna().astype(str).tolist())
    if not corpus.strip(): 
        return None
        
    font_path = None
    if os.name == "nt":
        for f in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
            if os.path.exists(f): 
                font_path = f
                break
    try: 
        wc = WordCloud(width=840, height=320, background_color="#0F172A", font_path=font_path, colormap="Blues", max_words=120).generate(corpus)
    except Exception: 
        return None
        
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")
    fig.tight_layout(pad=0)
    return fig

def chart_batch_pie(df_out):
    # Biểu đồ tròn phân bố
    if "Dự đoán" not in df_out.columns or df_out.empty:
        return None
        
    vc = df_out["Dự đoán"].value_counts()
    total_samples = len(df_out)
    
    labels = list(vc.index)
    colors = ["#10B981" if l == "Positive" else "#F43F5E" for l in labels]
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        vc.values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.32, edgecolor='#1E293B', linewidth=2),
        textprops=dict(fontsize=10, weight="bold", color="#F1F5F9")
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        
    ax.text(0, 0, f"TỔNG SỐ\n{total_samples:,}", ha="center", va="center", fontsize=11, weight="bold", color="#94A3B8")
    
    ax.axis("equal")
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_batch_hist(df_out):
    # Biểu đồ tần suất tin cậy
    if "Độ tin cậy" not in df_out.columns or "Dự đoán" not in df_out.columns or df_out.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for lbl, clr, b_clr in [("Positive", "#22C55E", "#4ADE80"), ("Negative", "#EF4444", "#FCA5A5")]:
        subset = df_out[df_out["Dự đoán"] == lbl]["Độ tin cậy"]
        if not subset.empty:
            ax.hist(subset, bins=20, alpha=0.82, color=clr, edgecolor=b_clr, linewidth=0.8, label=f"{lbl} Confidence", zorder=3)
            
    _apply_academic_style(ax, show_grid=True)
    ax.set_xlabel("Mức độ tự tin (Confidence Score)", fontsize=10)
    ax.set_ylabel("Tần suất mẫu", fontsize=10)
    legend = ax.legend(frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=9)
    for text in legend.get_texts():
        text.set_color("#F1F5F9")
    
    fig.patch.set_facecolor("none")
    fig.tight_layout()
    return fig

def chart_learning_curve(lc_path="models/learning_curve.json"):
    # Biểu đồ tiến trình hội tụ
    if not os.path.exists(lc_path): 
        return None
    try:
        with open(lc_path, "r", encoding="utf-8") as f: 
            data = json.load(f)
            
        iterations = data.get("iterations", [])
        loss_values = data.get("loss_values", [])
        
        if not iterations:
            return None
            
        fig, ax = plt.subplots(figsize=(6.5, 4))
        
        ax.fill_between(iterations, loss_values, color="#6366F1", alpha=0.15, zorder=2)
        ax.plot(iterations, loss_values, color="#818CF8", linewidth=2.5, zorder=3, label="Loss Value")
        
        ax.scatter(iterations[-1], loss_values[-1], color="#EF4444", s=50, edgecolors="white", linewidths=1.5, zorder=4, label="Convergence")
        ax.annotate(f" Hội tụ\n (Epoch {iterations[-1]})",
                    xy=(iterations[-1], loss_values[-1]),
                    xytext=(12, 12), textcoords="offset points",
                    color="#FCA5A5", weight="bold", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#EF4444", lw=0.8))
                    
        _apply_academic_style(ax, show_grid=True)
        ax.set_xlabel("Kỷ nguyên huấn luyện (Epochs)", fontsize=10)
        ax.set_ylabel("Độ hao hụt (Loss Value)", fontsize=10)
        
        fig.patch.set_facecolor("none")
        fig.tight_layout()
        return fig
    except Exception: 
        return None