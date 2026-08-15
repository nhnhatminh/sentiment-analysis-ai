import time
import re
import streamlit as st
import pandas as pd
from src.ui_styles import T, get_icon, get_icon_data_uri, NAV_PAGES
from src.ui_charts import (
    chart_distribution, chart_word_freq, chart_conf_hist, 
    chart_trend, chart_confusion, chart_wordcloud, chart_batch_pie, chart_batch_hist, chart_learning_curve
)

def detect_text_col(df):
    for col in df.columns:
        if df[col].dtype == object and df[col].dropna().str.len().mean() > 12: 
            return col
    return None

def detect_label_col(df):
    priority = ("label", "sentiment", "target", "target label", "class")
    for col in df.columns:
        if col.lower() in priority and df[col].nunique() <= 5: 
            return col
    for col in df.columns:
        if df[col].nunique() <= 4 and df[col].dtype in (object, "int64", int): 
            return col
    return None

def render_sidebar():
    t = T()
    with st.sidebar:
        st.markdown(f'<div class="sidebar-logo"><div class="sidebar-logo-icon">{get_icon("brain", 22, "white")}</div><div><div class="sidebar-logo-text">ReviewClassifyAI</div><div class="sidebar-logo-sub">Sentiment Agent</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-section-label">ĐIỀU HƯỚNG</div>', unsafe_allow_html=True)
        
        for page, (slug, icon_name) in NAV_PAGES.items():
            if st.button(page, key=f"nav_{slug}") and st.session_state.active_page != page:
                st.session_state.active_page = page
                st.rerun()

        active_page = st.session_state.active_page
        active_slug = NAV_PAGES[active_page][0]
        icon_rules = [f'section[data-testid="stSidebar"] div[data-testid="stButton"]:has(button#nav_{slug}) > button {{ background-image: url("{get_icon_data_uri(ic, 16, "%23FFFFFF" if page==active_page else "%2364748B")}") !important; }}' for page, (slug, ic) in NAV_PAGES.items()]
        st.markdown(f'<style>{"".join(icon_rules)} section[data-testid="stSidebar"] div[data-testid="stButton"]:has(button#nav_{active_slug}) > button {{ background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important; color: #FFFFFF !important; font-weight: 700 !important; box-shadow: 0 4px 10px rgba(37,99,235,0.3) !important; }}</style>', unsafe_allow_html=True)
        
        st.markdown(f'<div style="height:1px;background:{t["border"]};margin:1.2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-section-label">PHIÊN LÀM VIỆC</div>', unsafe_allow_html=True)
        h = st.session_state.history
        total = len(h)
        pos = sum(1 for x in h if x["label"] == "Positive")
        st.markdown(f'<div class="session-box"><div class="session-row"><span>Tổng dự đoán</span><span class="session-val">{total}</span></div><div class="session-row" style="margin-top:.6rem;"><span style="display:flex;align-items:center;gap:6px;">{get_icon("check-circle", 14, "#22C55E")} Tích cực</span><span class="session-val" style="color:#22C55E;">{pos}</span></div><div class="session-row" style="margin-top:.4rem;"><span style="display:flex;align-items:center;gap:6px;">{get_icon("x-circle", 14, "#EF4444")} Tiêu cực</span><span class="session-val" style="color:#EF4444;">{total-pos}</span></div></div>', unsafe_allow_html=True)

def render_header():
    st.markdown(f'<div class="hdr"><div class="hdr-badge"> {get_icon("brain", 14, "white")} Hệ thống Học sâu giải trình</div><div class="hdr-title">ReviewClassifyAI</div><div class="hdr-sub">Kiến trúc Tác tử phân loại cảm xúc sản phẩm thương mại điện tử sử dụng MLP</div></div>', unsafe_allow_html=True)

def render_overview(raw_df, clean_df, model, vectorizer, metrics):
    st.markdown('<div class="sh">Tổng quan hệ thống</div>', unsafe_allow_html=True)
    active_df = raw_df if raw_df is not None else clean_df
    ds = f"{len(active_df):,} dòng" if active_df is not None else "Không tìm thấy"
    mn = type(model).__name__ if model is not None else "Chưa tải"
    vn = type(vectorizer).__name__ if vectorizer is not None else "Chưa tải"
    ps = f"~{metrics.get('prediction_speed','N/A')}s" if metrics else "N/A"
    
    cols = st.columns(4, gap="medium")
    for col, (lbl, val, ic) in zip(cols, [("Tập dữ liệu", ds, "database"), ("Mô hình Agent", mn, "brain"), ("Kiến trúc Vector", vn, "cpu"), ("Tốc độ phản hồi", ps, "zap")]):
        with col: 
            st.markdown(f'<div class="card"><div class="card-lbl">{get_icon(ic, 14)} {lbl}</div><div class="card-val">{val}</div></div>', unsafe_allow_html=True)

def render_single_analysis(model, vectorizer, predict_func):
    st.markdown('<div class="sh">Phân tích cảm xúc trực tuyến</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        txt = st.text_area("", placeholder="Nhập đánh giá sản phẩm bằng tiếng Anh...", height=160, label_visibility="collapsed", key="single_input")
        c1, c2 = st.columns([2, 1], gap="small")
        with c1: 
            run = st.button("Phân tích đánh giá", use_container_width=True, key="btn_run")
        with c2: 
            if st.button("Xóa lịch sử", use_container_width=True, key="btn_clr"):
                st.session_state.history = []
                st.rerun()
    with right:
        if run:
            if not txt.strip(): 
                st.markdown('<div class="ib">Vui lòng nhập nội dung đánh giá trước.</div>', unsafe_allow_html=True)
            elif model is None: 
                st.markdown('<div class="wb">Chưa tải được mô hình tĩnh mạng Neural.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Đang suy luận thực tế..."):
                    label, prob, lat, contributions = predict_func(txt, model, vectorizer)
                    st.session_state.history.append({"text": txt[:80] + ("…" if len(txt) > 80 else ""), "label": label, "confidence": prob, "latency": lat, "timestamp": time.strftime("%H:%M:%S")})
                    
                    sc = "pos" if label == "Positive" else "neg"
                    st.markdown(f'<div class="res {sc}"><div class="res-lbl {sc}">Kết quả phân lớp</div><div class="res-val {sc}" style="display:flex; align-items:center; gap:8px;">{get_icon("check-circle" if label=="Positive" else "x-circle", 28)} {"Tích cực" if label=="Positive" else "Tiêu cực"}</div><div class="conf-wrap"><div class="conf-fill {sc}" style="width:{int(prob*100)}%;"></div></div></div>', unsafe_allow_html=True)
                    
                    ca, cb = st.columns(2)
                    with ca: 
                        st.markdown(f'<div class="card" style="margin-top:.8rem; padding: 1rem;"><div class="card-lbl">Độ tin cậy toán học</div><div class="card-val">{prob:.1%}</div></div>', unsafe_allow_html=True)
                    with cb: 
                        st.markdown(f'<div class="card" style="margin-top:.8rem; padding: 1rem;"><div class="card-lbl">Độ trễ suy luận</div><div class="card-val">{lat:.4f}s</div></div>', unsafe_allow_html=True)
                    
                    if contributions:
                        st.markdown('<br><div style="font-size:0.8rem; font-weight:700; color:#94A3B8; letter-spacing:0.5px;">CƠ SỞ TOÁN HỌC DỰA TRÊN TỪ KHÓA ĐÓNG GÓP:</div>', unsafe_allow_html=True)
                        for item in contributions:
                            w_word = item["word"]
                            w_coef = item["weight"]
                            w_color = "#4ADE80" if w_coef > 0 else "#FCA5A5"
                            w_dir = "Tích cực" if w_coef > 0 else "Tiêu cực"
                            st.markdown(f'<div style="font-size:0.85rem; margin-bottom:0.2rem; color:{T()["text"]};">• Từ <strong>"{w_word}"</strong>: tác động <span style="color:{w_color}; font-weight:700;">{w_coef:+.2f}</span> về hướng {w_dir}</div>', unsafe_allow_html=True)
        else: 
            st.markdown('<div class="ib">Nhập đánh giá bên trái rồi nhấn "Phân tích đánh giá".</div>', unsafe_allow_html=True)

def render_history():
    h = st.session_state.history
    if not h:
        st.markdown('<div class="ib">Chưa có bản ghi nhật ký.</div>', unsafe_allow_html=True)
        return
    total = len(h)
    pos = sum(1 for x in h if x["label"] == "Positive")
    st.markdown(f'<div class="sb"><div class="si"><div class="sl">{get_icon("layers",12)} Tổng số</div><div class="sv">{total}</div></div><div class="si"><div class="sl">{get_icon("check-circle",12)} Tích cực</div><div class="sv" style="color:#22C55E;">{pos}</div></div><div class="si"><div class="sl">{get_icon("x-circle",12)} Tiêu cực</div><div class="sv" style="color:#EF4444;">{total-pos}</div></div><div class="si"><div class="sl">{get_icon("activity",12)} TB Tin cậy</div><div class="sv">{sum(x["confidence"] for x in h)/total:.1%}</div></div><div class="si"><div class="sl">{get_icon("zap",12)} TB Độ trễ</div><div class="sv">{sum(x["latency"] for x in h)/total:.4f}s</div></div></div>', unsafe_allow_html=True)
    
    hc, lt = chart_conf_hist(h), chart_trend(h)
    if hc or lt:
        c1, c2 = st.columns(2, gap="large")
        if hc:
            with c1: 
                st.markdown('<div class="cc"><div class="ct">Phân bố độ tin cậy</div>', unsafe_allow_html=True)
                st.pyplot(hc, clear_figure=True)
                st.markdown('</div>', unsafe_allow_html=True)
        if lt:
            with c2: 
                st.markdown('<div class="cc"><div class="ct">Xu hướng tích lũy</div>', unsafe_allow_html=True)
                st.pyplot(lt, clear_figure=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
    rows = "".join(f'<tr><td>{x["timestamp"]}</td><td style="max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{x["text"]}</td><td><span class="{"bp" if x["label"]=="Positive" else "bn"}">{get_icon("check-circle" if x["label"]=="Positive" else "x-circle", 12)} {"Tích cực" if x["label"]=="Positive" else "Tiêu cực"}</span></td><td>{x["confidence"]:.1%}</td><td>{x["latency"]:.4f}s</td></tr>' for x in reversed(h))
    st.markdown(f'<div class="hw"><table class="ht"><thead><tr><th>Thời gian</th><th>Nội dung chuỗi thô</th><th>Phân loại cảm xúc</th><th>Độ chính xác</th><th>Thời gian chạy</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

def render_batch(model, vectorizer, batch_func):
    st.markdown('<div class="sh">Phân tích hàng loạt</div>', unsafe_allow_html=True)
    up = st.file_uploader("Tải lên file CSV", type=["csv"], label_visibility="collapsed", key="batch_up")
    if up is None:
        st.markdown('<div class="ib">Vui lòng nạp file dữ liệu lớn để stress-test hệ thống.</div>', unsafe_allow_html=True)
        return
        
    df_up = pd.read_csv(up)
    tc = detect_text_col(df_up)
    if tc is None: 
        st.markdown('<div class="wb">Không phát hiện được cột chuỗi văn bản hợp lệ.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="ib">Hệ thống chấp nhận cột: <strong>{tc}</strong> — Quét thấy {len(df_up):,} dòng dữ liệu.</div>', unsafe_allow_html=True)
    
    if st.button("Chạy phân tích hàng loạt", key="btn_batch"):
        with st.spinner(f"Mạng Neural đang xử lý {len(df_up):,} mẫu dữ liệu..."):
            res = batch_func(df_up[tc].tolist(), model, vectorizer)
            df_out = df_up.copy()
            df_out["Dự đoán"] = [r[0] for r in res]
            df_out["Độ tin cậy"] = [round(r[1], 4) for r in res]
            df_out["Độ trễ (s)"] = [r[2] for r in res]
            st.session_state.batch_results = df_out

    if st.session_state.batch_results is not None:
        df_out = st.session_state.batch_results
        pos = (df_out["Dự đoán"] == "Positive").sum()
        st.markdown(f'<div class="sb"><div class="si"><div class="sl">Tổng số dòng</div><div class="sv">{len(df_out):,}</div></div><div class="si"><div class="sl">Nhãn Tích cực</div><div class="sv" style="color:#22C55E;">{pos:,}</div></div><div class="si"><div class="sl">Nhãn Tiêu cực</div><div class="sv" style="color:#EF4444;">{len(df_out)-pos:,}</div></div><div class="si"><div class="sl">Mức tin cậy trung bình</div><div class="sv">{df_out["Độ tin cậy"].mean():.1%}</div></div></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([0.4, 0.6], gap="large")
        with c1: 
            st.markdown('<div class="cc"><div class="ct">Tỷ lệ phân bổ nhãn</div>', unsafe_allow_html=True)
            st.pyplot(chart_batch_pie(df_out), clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="cc"><div class="ct">Mật độ phân phối xác suất</div>', unsafe_allow_html=True)
            st.pyplot(chart_batch_hist(df_out), clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.dataframe(df_out.head(50), use_container_width=True, height=270)
        st.download_button("Xuất kết quả CSV", data=df_out.to_csv(index=False).encode("utf-8"), file_name="batch_output.csv", mime="text/csv")

def render_visualization(raw_df, clean_df):
    st.markdown('<div class="sh">Trực quan hóa dữ liệu</div>', unsafe_allow_html=True)
    df = clean_df if clean_df is not None else raw_df
    t1, t2, t3 = st.tabs(["Cân bằng dữ liệu", "Tần suất phân bổ từ", "Mật độ đám mây từ"])
    with t1: 
        st.pyplot(chart_distribution(df, detect_label_col(df)), clear_figure=True)
    with t2: 
        st.pyplot(chart_word_freq(df, detect_text_col(df)), clear_figure=True)
    with t3: 
        wc = chart_wordcloud(df, detect_text_col(df))
        if wc: 
            st.pyplot(wc, clear_figure=True)
        else:
            st.markdown('<div class="ib">Hệ thống Đám mây từ (Word Cloud) hiện chưa có dữ liệu hoặc thiếu thư viện liên kết. Vui lòng kiểm tra lại cột văn bản.</div>', unsafe_allow_html=True)

def render_performance(metrics):
    st.markdown('<div class="sh">Hiệu suất mô hình</div>', unsafe_allow_html=True)
    if metrics is None: 
        return
    
    cols = st.columns(4, gap="medium")
    for col, (k, name) in zip(cols, [("accuracy","Độ chính xác"),("precision","Precision"),("recall","Recall"),("f1","F1 Score")]):
        v = metrics.get(k)
        with col: 
            st.markdown(f'<div class="mc"><div class="mc-name">{name}</div><div class="mc-val">{v:.4f}</div><div class="mc-pct">{v*100:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.52, 0.48], gap="large")
    with c1:
        st.markdown('<div class="cc"><div class="ct">Ma trận nhầm lẫn (Confusion Matrix)</div>', unsafe_allow_html=True)
        st.pyplot(chart_confusion(metrics.get("confusion_matrix")), clear_figure=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cc"><div class="ct">Tiến trình hội tụ mạng Neural (Learning Curve)</div>', unsafe_allow_html=True)
        lc_fig = chart_learning_curve()
        if lc_fig: 
            st.pyplot(lc_fig, clear_figure=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_data_info(raw_df, clean_df):
    st.markdown('<div class="sh">Thông tin tập dữ liệu</div>', unsafe_allow_html=True)
    df = clean_df if clean_df is not None else raw_df
    if df is None:
        st.markdown('<div class="wb">Không tìm thấy tập dữ liệu. Đặt file raw.csv / clean.csv vào thư mục data/.</div>', unsafe_allow_html=True)
        return
        
    lc, tc = detect_label_col(df), detect_text_col(df)
    pos = neg = vocab = "N/A"
    if lc:
        v_counts = df[lc].value_counts()
        if lc.lower() == "rating":
            pos_count = v_counts.get(4, 0) + v_counts.get(5, 0)
        else:
            pos_count = v_counts.get(1, v_counts.get('Positive', 0))
            
        pos = f"{pos_count:,}"
        neg = f"{len(df) - pos_count:,}" 
    if tc:
        wds = set()
        for row in df[tc].dropna(): 
            wds.update(re.findall(r"\b[a-z]{2,}\b", str(row).lower()))
        vocab = f"{len(wds):,}"
        
    st.markdown(f'<style>.di-card {{ background: {T()["surface"]}; border-radius: 12px; padding: 1.5rem; box-shadow: {T()["shadow"]}; border: 1px solid {T()["border"]}; display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }} .di-icon {{ width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }} .di-lbl {{ font-size: .75rem; font-weight: 600; color: {T()["muted"]}; text-transform: uppercase; margin-bottom: .2rem; }} .di-val {{ font-size: 1.4rem; font-weight: 800; color: {T()["text"]}; }}</style>', unsafe_allow_html=True)
    
    data_points = [
        ("Tổng số dòng", f"{len(df):,}", "layers", "#EFF6FF", "#2563EB"), 
        ("Tổng số cột", str(len(df.columns)), "layout-dashboard", "#F5F3FF", "#D97706"), 
        ("Mẫu tích cực", pos, "check-circle", "#DCFCE7", "#16A34A"), 
        ("Mẫu tiêu cực", neg, "x-circle", "#FEE2E2", "#DC2626"), 
        ("Giá trị trông", f"{int(df.isnull().sum().sum()):,}", "activity", "#F3F4F6", "#4B5563"), 
        ("Kích thước bộ từ vựng", vocab, "file-text", "#FCE7F3", "#7C3AED")
    ]
    
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3, c1, c2, c3]
    for i, (lbl, val, icon, bg, color) in enumerate(data_points):
        with cols[i]: 
            st.markdown(
                f'''
                <div class="di-card">
                    <div class="di-icon" style="background:{bg}; color:{color};">
                        {get_icon(icon, 20, color)}
                    </div>
                    <div>
                        <div class="di-lbl">{lbl}</div>
                        <div class="di-val">{val}</div>
                    </div>
                </div>
                ''', 
                unsafe_allow_html=True
            )