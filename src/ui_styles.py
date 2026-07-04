import streamlit as st

STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","is","it","this","that","was","are","be","with","as","by","from","they","we","i","my","your","me","he","she","his","her","its","our","have","had","has","do","did","not","so","if","up","all","also","will","just","can","more","been","than","then","there","their","out","would","could","what","which","who","how","no","one","about","when","into","very","too","am","were","being"}

NAV_PAGES = {
    "Tổng quan": ("overview", "layout-dashboard"),
    "Phân tích đơn lẻ": ("single", "mouse-pointer-click"),
    "Phân tích hàng loạt": ("batch", "layers"),
    "Trực quan dữ liệu": ("viz", "bar-chart-2"),
    "Hiệu suất mô hình": ("performance", "activity"),
    "Thông tin dữ liệu": ("info", "file-text"),
}

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
    svg = raw_svg.replace("COLORPLACEHOLDER", color).replace("#", "%23").replace('"', "'")
    return f"data:image/svg+xml,{svg}"

def inject_css():
    t = T()
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; background-color: {t['bg']} !important; color: {t['text']} !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 2rem 2.5rem 4rem !important; max-width: 1280px; }}
    section[data-testid="stSidebar"] {{ background: {t['nav_bg']} !important; border-right: 1px solid {t['border']}; min-width: 260px !important; }}
    section[data-testid="stSidebar"] > div {{ padding: 1.2rem 1rem; }}
    .hdr {{ background: linear-gradient(135deg, #1E40AF 0%, #2563EB 55%, #4F46E5 100%); border-radius: 20px; padding: 2rem 2.6rem; margin-bottom: 2rem; box-shadow: 0 16px 50px rgba(37,99,235,.22); position: relative; overflow: hidden; }}
    .hdr-badge {{ display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25); color: #fff; font-size: .66rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; padding: 4px 12px; border-radius: 99px; margin-bottom: .85rem; }}
    .hdr-title {{ font-size: 2rem; font-weight: 800; color: #fff; margin: 0 0 .25rem; line-height: 1.2; }}
    .hdr-sub   {{ font-size: .95rem; color: rgba(255,255,255,.85); margin: 0; }}
    .sidebar-logo {{ display: flex; align-items: center; gap: .75rem; padding: .5rem .5rem 1.2rem; border-bottom: 1px solid {t['border']}; margin-bottom: 1.2rem; }}
    .sidebar-logo-icon {{ width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #2563EB, #4F46E5); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .sidebar-logo-text {{ font-size: .95rem; font-weight: 700; color: {t['text']}; line-height: 1.2; }}
    .sidebar-logo-sub  {{ font-size: .7rem; color: {t['muted']}; }}
    .nav-section-label {{ font-size: .65rem; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; color: {t['muted']}; padding: 0 .5rem; margin: 0 0 .6rem; }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] {{ margin-bottom: .2rem !important; }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{ display: flex !important; align-items: center !important; justify-content: flex-start !important; padding: .5rem 1rem .5rem 2.6rem !important; border-radius: 10px !important; cursor: pointer !important; transition: all .2s ease !important; font-size: .85rem !important; font-weight: 500 !important; color: {t['text']} !important; background-color: transparent !important; background-repeat: no-repeat !important; background-position: .8rem center !important; background-size: 16px 16px !important; border: 1px solid transparent !important; box-shadow: none !important; width: 100% !important; text-align: left !important; height: auto !important; }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{ background-color: {t['nav_hover']} !important; color: {t['primary']} !important; }}
    .ib {{ background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; border-radius: 10px; padding: .8rem 1rem; font-size: .85rem; color: #1E40AF; margin-bottom: 1rem; }}
    .wb {{ background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B; border-radius: 10px; padding: .8rem 1rem; font-size: .85rem; color: #92400E; margin-bottom: 1rem; }}
    .sh {{ font-size: .85rem; font-weight: 700; color: {t['text']}; text-transform: uppercase; letter-spacing: 1.1px; margin: 0 0 1rem; padding-bottom: .5rem; border-bottom: 2px solid {t['border']}; }}
    .card {{ background: {t['surface']}; border-radius: 14px; padding: 1.25rem 1.3rem; box-shadow: {t['shadow']}; border: 1px solid {t['border']}; transition: transform .25s, box-shadow .25s; height: 100%; }}
    .card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 28px rgba(37,99,235,.11); }}
    .card-lbl {{ font-size: .65rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: {t['muted']}; margin-bottom: .4rem; display: flex; align-items: center; gap: 6px; }}
    .card-val {{ font-size: 1.1rem; font-weight: 700; color: {t['text']}; line-height: 1.3; word-break: break-all; }}
    .mc {{ background: {t['surface']}; border-radius: 14px; padding: 1.4rem 1rem; box-shadow: {t['shadow']}; border: 1px solid {t['border']}; text-align: center; transition: transform .25s; }}
    .mc:hover {{ transform: translateY(-3px); box-shadow: 0 10px 26px rgba(37,99,235,.1); }}
    .mc-name {{ font-size: .65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: {t['muted']}; margin-bottom: .5rem; }}
    .mc-val  {{ font-size: 1.9rem; font-weight: 800; background: linear-gradient(135deg,#2563EB,#4F46E5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }}
    .res {{ border-radius: 14px; padding: 1.5rem 1.6rem; border: 1.5px solid transparent; }}
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
    .hw {{ background: {t['surface']}; border-radius: 14px; padding: 1.4rem; box-shadow: {t['shadow']}; border: 1px solid {t['border']}; }}
    table.ht {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
    table.ht thead th {{ border-bottom: 2px solid {t['border']}; padding: .6rem .4rem; text-align: left; font-size: .65rem; text-transform: uppercase; letter-spacing: .8px; color: {t['muted']}; font-weight: 600; }}
    table.ht tbody tr {{ border-bottom: 1px solid {t['border']}; transition: background .12s; }}
    table.ht tbody tr:hover {{ background: {t['surface2']}; }}
    table.ht tbody td {{ padding: .7rem .4rem; color: {t['text']}; vertical-align: middle; }}
    .bp {{ background:#DCFCE7; color:#15803D; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; display: inline-flex; align-items: center; gap: 4px; }}
    .bn {{ background:#FEE2E2; color:#B91C1C; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; display: inline-flex; align-items: center; gap: 4px; }}
    .sb {{ display: flex; gap: 1rem; flex-wrap: wrap; background: {t['surface2']}; border-radius: 12px; padding: 1rem 1.4rem; border: 1px solid {t['border']}; margin-bottom: 1.2rem; }}
    .si {{ flex: 1; min-width: 90px; }}
    .sl {{ font-size: .65rem; text-transform: uppercase; letter-spacing: .9px; color: {t['muted']}; font-weight: 600; margin-bottom: .25rem; display: flex; align-items: center; gap: 4px; }}
    .sv {{ font-size: 1.25rem; font-weight: 700; color: {t['text']}; }}
    .cc {{ background: {t['surface']}; border-radius: 14px; padding: 1.4rem; box-shadow: {t['shadow']}; border: 1px solid {t['border']}; margin-bottom: 1.2rem; }}
    .ct {{ font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: {t['text']}; margin-bottom: 1rem; border-bottom: 1px solid {t['border']}; padding-bottom: .5rem; }}
    .dv {{ height: 1px; background: {t['border']}; margin: 2rem 0; }}
    .ft {{ margin-top: 3rem; padding: 1.2rem 0; border-top: 1px solid {t['border']}; text-align: center; font-size: .8rem; color: {t['muted']}; }}
    .main div[data-testid="stButton"] > button {{ background: linear-gradient(135deg,#2563EB 0%,#4F46E5 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; padding: .6rem 1.6rem !important; font-size: .86rem !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(37,99,235,.28) !important; transition: opacity .2s, transform .15s, box-shadow .2s !important; width: 100%; }}
    .main div[data-testid="stButton"] > button:hover {{ opacity: .9 !important; transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(37,99,235,.36) !important; }}
    div[data-testid="stDownloadButton"] > button {{ background: linear-gradient(135deg,#2563EB 0%,#4F46E5 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: .86rem !important; box-shadow: 0 4px 12px rgba(37,99,235,.28) !important; }}
    div[data-testid="stTextArea"] textarea {{ border-radius: 10px !important; border: 1.5px solid {t['border']} !important; font-family: 'Inter', sans-serif !important; font-size: .9rem !important; color: {t['text']} !important; background: {t['surface2']} !important; padding: 1rem !important; }}
    div[data-testid="stTabs"] button {{ font-family: 'Inter', sans-serif !important; font-size: .88rem !important; font-weight: 500 !important; color: {t['muted']} !important; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {t['primary']} !important; font-weight: 700 !important; border-bottom: 2px solid {t['primary']} !important; }}
    .session-box {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1rem; margin-top: .5rem; box-shadow: 0 2px 8px rgba(0,0,0,.04); }}
    .session-row {{ display: flex; justify-content: space-between; align-items: center; font-size: .82rem; color: {t['muted']}; font-weight: 500; }}
    .session-val {{ font-weight: 700; color: {t['text']}; font-size: .9rem; }}
    </style>
    """, unsafe_allow_html=True)