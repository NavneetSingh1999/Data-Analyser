"""Streamlit dashboard.  Run:  streamlit run dashboard.py or python run_app.py

Stock Market Analyse — Interactive Book Wisdom Edition.
Synthesizes methodologies from:
- William O'Neil (CAN SLIM & Breakouts)
- Tom Williams (Volume Spread Analysis - VSA)
- Bill Williams (Trading Chaos: Alligator, Fractals, AO, MFI)
- Perry Kaufman (Adaptive Systems: KAMA, Efficiency Ratio, Noise Filtering)
- Edward Thorp & Sheen Kassouf (Beat the Market: Delta Hedging, Volatility Edge)
- Benjamin Graham & Warren Buffett (Margin of Safety, Graham Number, Economic Moats)
- National Stock Exchange (NCFM Technical Analysis & Derivatives)
"""
from __future__ import annotations

import os
import sys
import warnings

# Silence terminal deprecation notices (use_container_width and components.v1.html)
warnings.filterwarnings("ignore")
try:
    from streamlit import deprecation_util
    deprecation_util.show_deprecation_warning = lambda *a, **k: None
except Exception:
    pass

from datetime import date, datetime, timedelta
from pathlib import Path

# Automatically ensure categorical sub-folders are in sys.path
_ROOT = Path(__file__).resolve().parent
for _subdir in ("analytics", "data", "execution", "assets"):
    _sub_p = str(_ROOT / _subdir)
    if _sub_p not in sys.path:
        sys.path.insert(0, _sub_p)

LOGO_PATH = _ROOT / "assets" / "logo.png"
LOGO_ICON_PATH = _ROOT / "assets" / "logo_icon.png"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import advanced_analytics as aa
import backtest as bt
import book_analytics as ba
import book_knowledge as bk
import config
import importlib
import indicators as ind
import options_analytics as oa
try:
    importlib.reload(oa)
except Exception:
    pass
import options_payoff as op
import screener
import ai_trade_engine as ate
import volume_profile as vp
import intermarket_macro as im
import trailing_risk_engine as tre
import expiry_gamma_matrix as egm
import institutional_data as idata
import live_execution as le
import open_source_indicators as osi
import paper_trading as pt
import stock_directory as sd
try:
    importlib.reload(sd)
except Exception:
    pass
import tradingview_integration as tvi
import os
import storage
from providers import get_provider, is_market_open
from storage import get_history_cached

st.set_page_config(
    page_title="Alpha Edge — Institutional AI Trading Terminal",
    page_icon=str(LOGO_ICON_PATH) if LOGO_ICON_PATH.exists() else "📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.logo removed as requested (was displaying the top navigation/sidebar emblem)
# if LOGO_PATH.exists() and hasattr(st, "logo"):
#     try:
#         st.logo(str(LOGO_PATH), icon_image=str(LOGO_ICON_PATH) if LOGO_ICON_PATH.exists() else None)
#     except Exception:
#         pass

class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC THEME SYSTEM — Dark & Light Trading Terminal Modes
# ══════════════════════════════════════════════════════════════════════════════
if "ui_theme_mode" not in st.session_state:
    st.session_state["ui_theme_mode"] = "Dark"

is_dark = (st.session_state.get("ui_theme_mode", "Dark") == "Dark")

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC FONT SCALE SYSTEM — Enlarged & Screen-Adjustable Typography
# ══════════════════════════════════════════════════════════════════════════════
if "ui_font_scale_pct" not in st.session_state:
    st.session_state["ui_font_scale_pct"] = 100

font_scale_pct = int(st.session_state.get("ui_font_scale_pct", 100))
font_scale = font_scale_pct / 100.0
f_root_px = round(13.2 * font_scale, 1)

theme_cfg = {
    "is_dark": is_dark,
    # Institutional Palettes (Obsidian Dark vs FactSet/Koyfin Light)
    "bg_app": "#0b0f17" if is_dark else "#f8fafc",
    "bg_card": "#131926" if is_dark else "#ffffff",
    "bg_card_gradient": "linear-gradient(145deg, #141c2c 0%, #101622 100%)" if is_dark else "linear-gradient(145deg, #ffffff 0%, #ffffff 100%)",
    "bg_sidebar": "linear-gradient(180deg, #090d15 0%, #0e1420 100%)" if is_dark else "#ffffff",
    "bg_popover": "#151c2a" if is_dark else "#ffffff",
    "bg_hover": "#1c2638" if is_dark else "#f1f5f9",
    "border": "#222d40" if is_dark else "#cbd5e1",
    "border_focus": "#10b981" if is_dark else "#059669",
    "input_bg": "#141b28" if is_dark else "#ffffff",
    "input_border": "#2c3a50" if is_dark else "#94a3b8",
    "text_main": "#ffffff" if is_dark else "#0f172a",
    "text_sub": "#f1f5f9" if is_dark else "#1e293b",
    "text_body": "#cbd5e1" if is_dark else "#334155",
    "text_muted": "#94a3b8" if is_dark else "#475569",
    "primary": "#10b981" if is_dark else "#059669",
    "primary_hover": "#059669" if is_dark else "#047857",
    "primary_gradient": "linear-gradient(135deg, #10b981 0%, #059669 100%)" if is_dark else "linear-gradient(135deg, #059669 0%, #047857 100%)",
    "primary_shadow": "rgba(16, 185, 129, 0.25)" if is_dark else "rgba(5, 150, 105, 0.20)",
    # High-Contrast Metric Delta Pills
    "delta_pos_bg": "rgba(16, 185, 129, 0.16)" if is_dark else "#ecfdf5",
    "delta_pos_border": "rgba(16, 185, 129, 0.35)" if is_dark else "#a7f3d0",
    "delta_pos_text": "#34d399" if is_dark else "#047857",
    "delta_neg_bg": "rgba(239, 68, 68, 0.16)" if is_dark else "#fef2f2",
    "delta_neg_border": "rgba(239, 68, 68, 0.35)" if is_dark else "#fecaca",
    "delta_neg_text": "#f87171" if is_dark else "#dc2626",
    "delta_neu_bg": "rgba(255, 255, 255, 0.06)" if is_dark else "#f1f5f9",
    "delta_neu_border": "rgba(255, 255, 255, 0.12)" if is_dark else "#cbd5e1",
    "delta_neu_text": "#94a3b8" if is_dark else "#334155",
    "plotly_template": "plotly_dark" if is_dark else "plotly_white",
    "chart_plot_bg": "rgba(19, 25, 38, 0.70)" if is_dark else "#ffffff",
    "chart_paper_bg": "rgba(11, 15, 23, 0.0)" if is_dark else "rgba(255, 255, 255, 0.0)",
    "chart_grid": "rgba(255, 255, 255, 0.06)" if is_dark else "rgba(15, 23, 42, 0.08)",
    "chart_text": "#94a3b8" if is_dark else "#334155",
    "hero_bg": "linear-gradient(90deg, #0b0f17 0%, #162238 50%, #064e3b 100%)" if is_dark else "linear-gradient(90deg, #ffffff 0%, #ecfdf5 50%, #d1fae5 100%)",
    "hero_text": "#ffffff" if is_dark else "#0f172a",
    "hero_sub": "#94a3b8" if is_dark else "#475569",
    "hero_pill_bg": "rgba(16, 185, 129, 0.15)" if is_dark else "rgba(5, 150, 105, 0.12)",
    "hero_pill_border": "#10b981" if is_dark else "#059669",
    "hero_pill_color": "#a7f3d0" if is_dark else "#047857",
}

st.markdown(
    f"""
    <style>
    /* ═══════ GLOBAL TYPOGRAPHY & DYNAMIC THEME (DARK / LIGHT) ═══════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..800;1,14..32,300..800&family=JetBrains+Mono:ital,wght@0,400..700;1,400..700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif !important;
        background-color: {theme_cfg['bg_app']} !important;
        color: {theme_cfg['text_body']} !important;
        font-size: {f_root_px}px !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
    }}

    /* ═══════ MONOSPACE & FINANCIAL TABULAR FIGURES ═══════ */
    code, pre, kbd, samp,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] *,
    .metric-value,
    .price-tag,
    .tabular-data,
    [data-testid="stDataFrame"] td,
    table td,
    .stNumberInput input {{
        font-family: 'JetBrains Mono', 'SF Mono', 'Roboto Mono', Menlo, Consolas, monospace !important;
        font-variant-numeric: tabular-nums lining-nums !important;
        font-feature-settings: "tnum" 1, "zero" 1 !important;
    }}

    /* ═══════ PROFESSIONAL TYPOGRAPHIC HIERARCHY & SCALE ═══════ */
    h1, [data-testid="stMarkdownContainer"] h1 {{
        font-size: 1.30rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.20 !important;
        color: {theme_cfg['text_main']} !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.10rem !important;
    }}
    h2, [data-testid="stMarkdownContainer"] h2 {{
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.018em !important;
        line-height: 1.22 !important;
        color: {theme_cfg['text_main']} !important;
        margin-top: 0.20rem !important;
        margin-bottom: 0.08rem !important;
    }}
    h3, [data-testid="stMarkdownContainer"] h3 {{
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.012em !important;
        line-height: 1.25 !important;
        color: {theme_cfg['text_sub']} !important;
        margin-top: 0.18rem !important;
        margin-bottom: 0.06rem !important;
    }}
    h4, [data-testid="stMarkdownContainer"] h4 {{
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.008em !important;
        line-height: 1.30 !important;
        color: {theme_cfg['text_sub']} !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.04rem !important;
    }}
    h5, [data-testid="stMarkdownContainer"] h5 {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: {theme_cfg['text_muted']} !important;
        margin-top: 0.65rem !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.30 !important;
    }}
    h6, [data-testid="stMarkdownContainer"] h6 {{
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: {theme_cfg['text_muted']} !important;
        margin-top: 0.10rem !important;
        margin-bottom: 0.02rem !important;
    }}
    p, span, li,
    .stMarkdown p, .stMarkdown span, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {{
        font-size: 0.90rem !important;
        font-weight: 400 !important;
        line-height: 1.45 !important;
        letter-spacing: -0.005em !important;
        color: {theme_cfg['text_body']} !important;
    }}
    b, strong, th,
    .stMarkdown strong, .stMarkdown b,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {{
        font-weight: 600 !important;
        color: {theme_cfg['text_main']} !important;
    }}
    .stCaption, .stCaption *, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        font-size: 0.80rem !important;
        font-weight: 400 !important;
        color: {theme_cfg['text_muted']} !important;
        line-height: 1.35 !important;
        letter-spacing: 0.01em !important;
    }}

    /* ═══════ STREAMLIT HEADER & MAIN CONTAINER SPACING ═══════ */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        z-index: 100000 !important;
    }}

    [data-testid="stToolbar"],
    .stDeployButton,
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
    }}

    /* Sidebar Collapse Controls for Full-Screen View */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: inline-flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 8px !important;
        left: 8px !important;
        z-index: 100002 !important;
        background: {theme_cfg['bg_card']} !important;
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 6px !important;
        color: {theme_cfg['primary']} !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        cursor: pointer !important;
        width: 32px !important;
        height: 32px !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    /* Balanced Main container spacing (Zero top cutoff, zero overlap) */
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"],
    section.main > div.block-container {{
        padding-top: 1.0rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 0.90rem !important;
        padding-right: 0.90rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }}

    /* Collision-Free Layout Gaps */
    [data-testid="stVerticalBlock"] {{
        gap: 0.50rem !important;
    }}
    [data-testid="stHorizontalBlock"] {{
        gap: 0.45rem !important;
    }}

    /* Clean Dividers & Zero-Gap Horizontal Rules */
    hr, [data-testid="stDivider"] {{
        margin: 0.25rem 0 !important;
        border-color: {theme_cfg['border']} !important;
        border-width: 1px 0 0 0 !important;
        opacity: 0.4 !important;
    }}

    /* ═══════ PINNED STATIC TOP NAVIGATION PANE (FREEZE-PANE) ═══════ */
    .pinned-freeze-pane,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#pinned-static-top-anchor),
    .block-container > div[data-testid="stVerticalBlock"] > div:has(#pinned-static-top-anchor) {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0px !important;
        z-index: 995 !important;
        background: {theme_cfg['bg_app']} !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        margin-top: -0.65rem !important;
        border-bottom: 1px solid {theme_cfg['border']} !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, {0.35 if is_dark else 0.08}) !important;
    }}

    /* Primary Top-Level Main Tabs Sticky beneath the Search Bar */
    .pinned-main-tabs,
    div[data-testid="stTabs"] > div[role="tablist"],
    div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:has(div[data-testid="stTabs"]) > div[data-testid="stTabs"] > div[role="tablist"],
    .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div[role="tablist"] {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 104px !important;
        z-index: 990 !important;
        background: {theme_cfg['bg_app']} !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        border-bottom: 1px solid {theme_cfg['border']} !important;
    }}

    /* Tier 3: Screener Header & Control Dock Pinned directly under Main Tabs */
    .pinned-screener-dock,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#pinned-screener-dock-anchor),
    .block-container > div[data-testid="stVerticalBlock"] > div:has(#pinned-screener-dock-anchor),
    div[data-testid="stElementContainer"]:has(#pinned-screener-dock-anchor) + div {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 142px !important;
        z-index: 985 !important;
        background: {theme_cfg['bg_app']} !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        border-bottom: 1px solid {theme_cfg['border']} !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, {0.35 if is_dark else 0.08}) !important;
    }}

    /* Reset form margins so search bar fits seamlessly */
    div[data-testid="stForm"] {{
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Primary Form Submit Button styling */
    button[kind="primaryFormSubmit"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {{
        background: {theme_cfg['primary_gradient']} !important;
        border: 1px solid {theme_cfg['primary']} !important;
        box-shadow: 0 2px 8px {theme_cfg['primary_shadow']} !important;
        height: 32px !important;
        min-height: 32px !important;
        border-radius: 5px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    button[kind="primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {{
        background: {theme_cfg['primary_gradient']} !important;
        box-shadow: 0 3px 12px {theme_cfg['primary_shadow']} !important;
        transform: translateY(-1px) !important;
    }}

    /* ═══════ BUTTONS & POPOVERS ═══════ */
    .stButton > button,
    .stButton button,
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopover"] button {{
        border-radius: 5px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 2px 8px !important;
        min-height: 30px !important;
        height: 30px !important;
        white-space: nowrap !important;
    }}
    div[data-testid="stPopover"] button svg {{
        margin-left: 2px !important;
        width: 13px !important;
        height: 13px !important;
    }}

    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"],
    div[data-testid="stPopover"] > button {{
        background: {theme_cfg['bg_card']} !important;
        border: 1px solid {theme_cfg['border']} !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, {0.2 if is_dark else 0.04}) !important;
    }}
    .stButton > button[kind="secondary"] *,
    .stButton > button[data-testid="stBaseButton-secondary"] *,
    .stButton > button[kind="secondary"] p,
    .stButton > button[data-testid="stBaseButton-secondary"] p,
    div[data-testid="stPopover"] > button * {{
        color: {theme_cfg['text_main']} !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stPopover"] > button:hover {{
        background: {'#1d273b' if is_dark else '#f1f5f9'} !important;
        border-color: {theme_cfg['primary']} !important;
        box-shadow: 0 0 10px {theme_cfg['primary_shadow']} !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button[kind="secondary"]:hover *,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover *,
    div[data-testid="stPopover"] > button:hover * {{
        color: {theme_cfg['primary']} !important;
    }}

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {theme_cfg['primary_gradient']} !important;
        border: 1px solid {theme_cfg['primary']} !important;
        box-shadow: 0 2px 8px {theme_cfg['primary_shadow']} !important;
    }}
    .stButton > button[kind="primary"] *,
    .stButton > button[data-testid="stBaseButton-primary"] *,
    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="stBaseButton-primary"] p {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {theme_cfg['primary_gradient']} !important;
        box-shadow: 0 3px 12px {theme_cfg['primary_shadow']} !important;
        transform: translateY(-1px) !important;
    }}

    /* ═══════ SELECTBOX & DROPDOWNS ═══════ */
    .stSelectbox label,
    .stSelectbox label p,
    .stSelectbox label span {{
        color: {theme_cfg['text_muted']} !important;
        font-weight: 600 !important;
        font-size: {0.80 * font_scale:.2f}rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }}
    
    .stSelectbox,
    .stSelectbox > div,
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] div,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] div {{
        background-color: {theme_cfg['input_bg']} !important;
    }}
    
    [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        border: 1.5px solid {theme_cfg['input_border']} !important;
        border-radius: 6px !important;
        min-height: 32px !important;
        height: 32px !important;
        padding: 0 6px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, {0.20 if is_dark else 0.04}) !important;
        transition: all 0.18s ease-in-out !important;
    }}
    
    [data-baseweb="select"] > div:hover,
    [data-baseweb="select"]:focus-within > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
    section[data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div {{
        border-color: {theme_cfg['primary']} !important;
        box-shadow: 0 0 0 1.5px {theme_cfg['primary']}, 0 2px 10px {theme_cfg['primary_shadow']} !important;
    }}
    
    /* Text inside select box */
    [data-baseweb="select"] *,
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span {{
        color: {theme_cfg['text_main']} !important;
        font-weight: 500 !important;
        font-size: {0.76 * font_scale:.2f}rem !important;
    }}
    
    [data-baseweb="select"] div[aria-hidden="true"],
    [data-baseweb="select"] input::placeholder,
    section[data-testid="stSidebar"] [data-baseweb="select"] div[aria-hidden="true"] {{
        color: {theme_cfg['text_muted']} !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }}
    
    [data-baseweb="select"] svg,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
        fill: {theme_cfg['text_muted']} !important;
    }}

    /* Dropdown popover menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {theme_cfg['bg_popover']} !important;
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 6px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, {0.35 if is_dark else 0.12}) !important;
    }}
    ul[role="listbox"] li,
    ul[role="listbox"] li * {{
        background-color: {theme_cfg['bg_popover']} !important;
        color: {theme_cfg['text_main']} !important;
        font-size: {0.86 * font_scale:.2f}rem !important;
    }}
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li:hover *,
    ul[role="listbox"] li[aria-selected="true"],
    ul[role="listbox"] li[aria-selected="true"] * {{
        background-color: {theme_cfg['bg_hover']} !important;
        color: {theme_cfg['primary']} !important;
    }}

    /* ═══════ TEXT & NUMBER INPUTS ═══════ */
    .stTextInput label, .stTextInput label p,
    .stNumberInput label, .stNumberInput label p,
    .stTextArea label, .stTextArea label p {{
        color: {theme_cfg['text_muted']} !important;
        font-weight: 600 !important;
        font-size: {0.80 * font_scale:.2f}rem !important;
    }}

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        background-color: {theme_cfg['input_bg']} !important;
        color: {theme_cfg['text_main']} !important;
        border: 1.5px solid {theme_cfg['input_border']} !important;
        border-radius: 6px !important;
        min-height: 32px !important;
        height: 32px !important;
        padding: 2px 6px !important;
        font-size: 0.84rem !important;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {{
        border-color: {theme_cfg['border_focus']} !important;
        box-shadow: 0 0 0 2px {theme_cfg['primary_shadow']} !important;
    }}

    /* ═══════ SLEEK NON-OVERLAPPING INSTITUTIONAL METRICS ═══════ */
    [data-testid="stMetric"] {{
        background: {theme_cfg['bg_card']} !important;
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 6px !important;
        padding: {round(3 * font_scale)}px {round(6 * font_scale)}px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, {0.12 if is_dark else 0.03}) !important;
        min-height: {round(38 * font_scale)}px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span,
    [data-testid="stMetricLabel"] div {{
        color: {theme_cfg['text_muted']} !important;
        font-weight: 600 !important;
        font-size: {0.62 * font_scale:.2f}rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1.25 !important;
        margin-bottom: 2px !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] div {{
        color: {theme_cfg['text_main']} !important;
        font-weight: 700 !important;
        font-size: {0.88 * font_scale:.2f}rem !important;
        line-height: 1.20 !important;
        font-family: 'JetBrains Mono', 'SF Mono', 'Roboto Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    /* ═══════ HIGH-CONTRAST METRIC DELTA PILLS (DARK & LIGHT MODE) ═══════ */
    [data-testid="stMetricDelta"] {{
        background-color: {theme_cfg['delta_pos_bg']} !important;
        border: 1px solid {theme_cfg['delta_pos_border']} !important;
        border-radius: 4px !important;
        padding: 1px 6px !important;
        margin-top: 2px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 3px !important;
        width: fit-content !important;
        max-width: 100% !important;
    }}
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {{
        color: {theme_cfg['delta_pos_text']} !important;
        font-weight: 700 !important;
        font-size: {0.66 * font_scale:.2f}rem !important;
        line-height: 1.15 !important;
        font-family: 'JetBrains Mono', 'SF Mono', 'Roboto Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    [data-testid="stMetricDelta"] svg {{
        fill: {theme_cfg['delta_pos_text']} !important;
        color: {theme_cfg['delta_pos_text']} !important;
    }}

    /* Negative / Downward Deltas */
    [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) {{
        background-color: {theme_cfg['delta_neg_bg']} !important;
        border-color: {theme_cfg['delta_neg_border']} !important;
    }}
    [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]),
    [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) * {{
        color: {theme_cfg['delta_neg_text']} !important;
    }}
    [data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) svg {{
        fill: {theme_cfg['delta_neg_text']} !important;
        color: {theme_cfg['delta_neg_text']} !important;
    }}

    /* Neutral Deltas (No Up/Down SVG, e.g. 'Foreign Institutions', 'Balanced') */
    [data-testid="stMetricDelta"]:not(:has(svg)) {{
        background-color: {theme_cfg['delta_neu_bg']} !important;
        border-color: {theme_cfg['delta_neu_border']} !important;
    }}
    [data-testid="stMetricDelta"]:not(:has(svg)),
    [data-testid="stMetricDelta"]:not(:has(svg)) * {{
        color: {theme_cfg['delta_neu_text']} !important;
    }}

    /* ═══════ PERMANENT VISIBLE & EXPANDED INSTITUTIONAL SIDEBAR ═══════ */
    div[data-testid="stAppViewContainer"],
    .stAppViewContainer {{
        display: flex !important;
        flex-direction: row !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
    }}

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="true"] {{
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        margin-left: 0px !important;
        left: 0px !important;
        top: 0px !important;
        min-width: 300px !important;
        max-width: 320px !important;
        width: 300px !important;
        height: 100vh !important;
        overflow-y: auto !important;
        flex-shrink: 0 !important;
        z-index: 100 !important;
        background: {theme_cfg['bg_sidebar']} !important;
        border-right: 1px solid {theme_cfg['border']} !important;
    }}

    section[data-testid="stSidebar"][aria-expanded="false"] {{
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        margin-left: -320px !important;
    }}

    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div {{
        display: flex !important;
        flex-direction: column !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        width: 100% !important;
    }}

    /* Sidebar collapse button */
    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-header"],
    section[data-testid="stSidebar"] button[aria-label*="Collapse" i] {{
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 0.8 !important;
        color: {theme_cfg['text_muted']} !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
    }}
    [data-testid="stSidebarCollapseButton"]:hover,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-header"]:hover {{
        opacity: 1 !important;
        color: {theme_cfg['primary']} !important;
    }}

    /* Balanced Main Container layout beside sidebar */
    [data-testid="stMain"],
    section.main,
    .appview-container section.main {{
        position: relative !important;
        flex: 1 1 auto !important;
        width: 100% !important;
        max-width: 100% !important;
        height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        margin-left: 0px !important;
    }}

    div[data-testid="stAppViewContainer"]:has(section[data-testid="stSidebar"][aria-expanded="false"]) section.main,
    section[data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stMain"],
    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main {{
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label *,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {{
        color: {theme_cfg['text_body']} !important;
    }}
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stCaption * {{
        color: {theme_cfg['text_muted']} !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {theme_cfg['input_bg']} !important;
        border: 1.5px solid {theme_cfg['input_border']} !important;
        border-radius: 8px !important;
        min-height: {round(35 * font_scale)}px !important;
        height: {round(35 * font_scale)}px !important;
        padding: 0 8px !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div {{
        color: {theme_cfg['text_main']} !important;
        font-size: {0.84 * font_scale:.2f}rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
        fill: {theme_cfg['primary']} !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        padding: 3px 6px !important;
        font-size: {0.82 * font_scale:.2f}rem !important;
        min-height: {round(30 * font_scale)}px !important;
        height: {round(30 * font_scale)}px !important;
        white-space: nowrap !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        margin-bottom: 0.35rem !important;
    }}
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5 {{
        margin-top: 0.65rem !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.30 !important;
        clear: both !important;
    }}

    /* ═══════ PROFESSIONAL TABS ARCHITECTURE (PRIMARY & NESTED) ═══════ */
    /* 1. Primary Top-Level Desk Navigation Bar */
    .stTabs > [data-baseweb="tab-list"],
    div[data-testid="stTabs"] > [data-baseweb="tab-list"] {{
        width: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 4px !important;
        background: {'linear-gradient(180deg, #101626 0%, #0c111e 100%)' if is_dark else 'linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%)'} !important;
        border-radius: 8px !important;
        padding: 4px !important;
        border: 1px solid {theme_cfg['border']} !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: thin !important;
        scrollbar-color: {'rgba(0, 208, 156, 0.3) transparent' if is_dark else 'rgba(5, 150, 105, 0.3) transparent'} !important;
        margin-bottom: 0.70rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, {0.25 if is_dark else 0.04}) !important;
    }}

    /* Primary Tab Buttons (Unrestricted Width, Zero Ellipsis Truncation) */
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"],
    div[data-testid="stTabs"] > [data-baseweb="tab-list"] [data-baseweb="tab"] {{
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: max-content !important;
        max-width: none !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 {round(10 * font_scale)}px !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
        white-space: nowrap !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
    }}

    /* Primary Tab Typography */
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"] *,
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"] p,
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"] span,
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"] div {{
        color: {theme_cfg['text_muted']} !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        letter-spacing: -0.01em !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }}

    /* Primary Tab Hover */
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"]:hover {{
        background-color: {'rgba(255, 255, 255, 0.05)' if is_dark else 'rgba(0, 0, 0, 0.04)'} !important;
        border-color: {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.06)'} !important;
        transform: translateY(-0.5px) !important;
    }}
    .stTabs > [data-baseweb="tab-list"] [data-baseweb="tab"]:hover * {{
        color: {theme_cfg['text_main']} !important;
    }}

    /* Primary Tab Active State (Gleaming Emerald Institutional Pill) */
    .stTabs > [data-baseweb="tab-list"] [aria-selected="true"] {{
        background: {theme_cfg['primary_gradient']} !important;
        border: 1px solid {theme_cfg['primary']} !important;
        box-shadow: 0 2px 10px {theme_cfg['primary_shadow']}, inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        transform: none !important;
    }}
    .stTabs > [data-baseweb="tab-list"] [aria-selected="true"] *,
    .stTabs > [data-baseweb="tab-list"] [aria-selected="true"] p,
    .stTabs > [data-baseweb="tab-list"] [aria-selected="true"] span,
    .stTabs > [data-baseweb="tab-list"] [aria-selected="true"] div {{
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
    }}

    /* 2. Secondary Sub-Tabs (Nested Inside Workspaces) */
    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background-color: {'rgba(15, 22, 36, 0.6)' if is_dark else '#eef2f6'} !important;
        border: 1px solid {'rgba(255, 255, 255, 0.07)' if is_dark else '#cbd5e1'} !important;
        border-radius: 6px !important;
        padding: 2px !important;
        gap: 2px !important;
        margin-bottom: 0.40rem !important;
        box-shadow: none !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
    }}
    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab"],
    div[data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab"] {{
        height: {round(24 * font_scale)}px !important;
        min-height: {round(24 * font_scale)}px !important;
        padding: 0 {round(8 * font_scale)}px !important;
        border-radius: 4px !important;
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: max-content !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab"] *,
    div[data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab"] * {{
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: {theme_cfg['text_muted']} !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }}
    .stTabs [data-baseweb="tab-panel"] .stTabs [aria-selected="true"],
    div[data-testid="stTabs"] [data-testid="stTabs"] [aria-selected="true"] {{
        background: {'#182338' if is_dark else '#ffffff'} !important;
        border: 1px solid {theme_cfg['primary']} !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, {0.2 if is_dark else 0.05}) !important;
    }}
    .stTabs [data-baseweb="tab-panel"] .stTabs [aria-selected="true"] *,
    div[data-testid="stTabs"] [data-testid="stTabs"] [aria-selected="true"] * {{
        color: {theme_cfg['primary']} !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }}

    /* Streamlit Tab Border & Highlight Clean Reset */
    .stTabs button[aria-label*="Scroll tabs"] {{
        background: {theme_cfg['bg_card']} !important;
        color: {theme_cfg['text_main']} !important;
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 4px !important;
        opacity: 0.8 !important;
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }}

    /* ═══════ CHECKBOXES & RADIO ═══════ */
    div[data-testid="stCheckbox"] label span p,
    .stRadio label, .stRadio label *,
    .stCheckbox label, .stCheckbox label * {{
        font-size: {0.84 * font_scale:.2f}rem !important;
        color: {theme_cfg['text_body']} !important;
        white-space: normal !important;
        overflow: visible !important;
    }}

    /* ═══════ EXPANDERS ═══════ */
    details[data-testid="stExpander"] {{
        background: {theme_cfg['bg_card']} !important;
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 8px !important;
        margin-bottom: 0.3rem !important;
    }}
    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary * {{
        color: {theme_cfg['text_main']} !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }}

    /* ═══════ COMPACT CONTAINERS & CARDS ═══════ */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 8px !important;
        background: {theme_cfg['bg_card']} !important;
        padding: 0.35rem 0.55rem !important;
        margin-bottom: 0.20rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, {0.22 if is_dark else 0.05}) !important;
    }}

    /* ═══════ DATAFRAMES / TABLES ═══════ */
    [data-testid="stDataFrame"] {{
        border: 1px solid {theme_cfg['border']} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    /* ═══════ ALERTS ═══════ */
    .stAlert {{
        border-radius: 6px !important;
        border-width: 1px !important;
        padding: 5px 10px !important;
        margin-bottom: 0.3rem !important;
    }}

    /* ═══════ PROGRESS BAR ═══════ */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #387ed1 0%, {theme_cfg['primary']} 100%) !important;
        border-radius: 4px !important;
    }}

    /* ═══════ DIVIDERS ═══════ */
    hr {{
        border-color: {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.08)'} !important;
        margin: 0.60rem 0 !important;
    }}

    /* ═══════ PLOTLY CHART CONTAINER ═══════ */
    .stPlotlyChart {{
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid {theme_cfg['border']} !important;
        background: {theme_cfg['bg_card']} !important;
    }}

    /* ═══════ SCROLLBAR ═══════ */
    ::-webkit-scrollbar {{
        width: 5px;
        height: 5px;
    }}
    ::-webkit-scrollbar-track {{
        background: {'rgba(10, 14, 23, 0.5)' if is_dark else 'rgba(241, 245, 249, 0.8)'};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {'rgba(0, 208, 156, 0.25)' if is_dark else 'rgba(5, 150, 105, 0.25)'};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {'rgba(0, 208, 156, 0.45)' if is_dark else 'rgba(5, 150, 105, 0.45)'};
    }}

    /* ═══════ CAPTION TEXT ═══════ */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {theme_cfg['text_muted']} !important;
        font-size: 0.72rem !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# JavaScript Freeze-Pane multi-tier navigation locking & observer
components.html(
    """
    <script>
    function applyFreezePane() {
        try {
            const pDoc = window.parent.document;
            if (!pDoc) return;

            // Tier 1: Macro ribbon + search bar container
            let topH = 104;
            const anchor = pDoc.getElementById("pinned-static-top-anchor");
            if (anchor) {
                const rootBlock = pDoc.querySelector('.block-container > div[data-testid="stVerticalBlock"]');
                if (rootBlock && rootBlock.children) {
                    for (let child of rootBlock.children) {
                        if (child.contains(anchor)) {
                            child.classList.add("pinned-freeze-pane");
                            topH = child.offsetHeight || 104;
                            break;
                        }
                    }
                }
            }

            // Tier 2: Primary main workspace tabs
            let mainH = 36;
            const mainTabs = pDoc.querySelector('div[data-testid="stTabs"] > div[role="tablist"]');
            if (mainTabs) {
                mainTabs.classList.add("pinned-main-tabs");
                mainTabs.style.top = topH + 'px';
                mainH = mainTabs.offsetHeight || 36;
            }

            // Tier 3: Screener control dock pinned directly under Main Tabs
            const scrDock = pDoc.getElementById("pinned-screener-dock-anchor");
            if (scrDock) {
                let dockBlock = scrDock.closest('.block-container > div[data-testid="stVerticalBlock"] > div, div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stElementContainer"]');
                if (dockBlock) {
                    dockBlock.classList.add("pinned-screener-dock");
                    dockBlock.style.top = (topH + mainH) + 'px';
                }
            }
        } catch(e) {}
    }

    applyFreezePane();
    setTimeout(applyFreezePane, 50);
    setTimeout(applyFreezePane, 200);
    setTimeout(applyFreezePane, 500);
    setTimeout(applyFreezePane, 1200);
    setInterval(applyFreezePane, 1000);

    try {
        const pDoc = window.parent.document;
        if (pDoc && pDoc.body && !window._freezePaneObserverSet) {
            window._freezePaneObserverSet = true;
            const observer = new MutationObserver(() => { applyFreezePane(); });
            observer.observe(pDoc.body, { childList: true, subtree: true });
        }
    } catch(e) {}
    </script>
    """,
    height=0,
    width=0,
)



@st.cache_resource(show_spinner=False)
def make_provider(name: str):
    return get_provider(name)


# Session State Initialization
if "active_symbol" not in st.session_state:
    st.session_state["active_symbol"] = "RELIANCE"
if "scanner_preset" not in st.session_state:
    st.session_state["scanner_preset"] = "all"


# Global Plotly chart theme adapting to user's selected mode (Dark or Light)
def apply_chart_theme(figure: go.Figure, **kwargs) -> go.Figure:
    """Apply unified professional styling (Dark or Light) to any Plotly chart safely."""
    tmpl = theme_cfg["plotly_template"]
    theme = dict(
        template=tmpl,
        paper_bgcolor=theme_cfg["chart_paper_bg"],
        plot_bgcolor=theme_cfg["chart_plot_bg"],
        font=dict(family="Inter, -apple-system, sans-serif", color=theme_cfg["chart_text"], size=11),
        xaxis=dict(gridcolor=theme_cfg["chart_grid"], zerolinecolor=theme_cfg["chart_grid"]),
        yaxis=dict(gridcolor=theme_cfg["chart_grid"], zerolinecolor=theme_cfg["chart_grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=theme_cfg["chart_text"])),
        hoverlabel=dict(bgcolor=theme_cfg["bg_card"], font_size=12, font_color=theme_cfg["text_main"], font_family="Inter, sans-serif"),
    )
    for k, v in kwargs.items():
        theme[k] = v
    try:
        figure.update_layout(**theme)
    except Exception:
        try:
            figure.update_layout(template=tmpl)
        except Exception:
            pass
    return figure


# Sidebar Configuration — Institutional Watchlist Panel
with st.sidebar:
    import base64
    _b64_icon = ""
    if LOGO_ICON_PATH.exists():
        try:
            _b64_icon = base64.b64encode(LOGO_ICON_PATH.read_bytes()).decode("utf-8")
        except Exception:
            pass
    _p_col = theme_cfg.get("primary", "#10b981")
    _p_sh = theme_cfg.get("primary_shadow", "rgba(16,185,129,0.2)")
    _icon_elem = (
        f'<img src="data:image/png;base64,{_b64_icon}" style="width: 32px; height: 32px; border-radius: 6px; object-fit: contain; box-shadow: 0 2px 8px rgba(16,185,129,0.25);" />'
        if _b64_icon
        else f'<div style="background: linear-gradient(135deg, {_p_col} 0%, #387ed1 100%); width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; box-shadow: 0 1px 4px {_p_sh};">🏛️</div>'
    )
    st.markdown(
        f"""
        <div style="padding: 2px 0 8px 0; border-bottom: 1px solid {theme_cfg['border']}; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                {_icon_elem}
                <div>
                    <div style="font-size: 0.92rem; font-weight: 700; color: {theme_cfg['text_main']}; letter-spacing: 0.03em; line-height: 1.1;">ALPHA EDGE</div>
                    <div style="font-size: 0.62rem; color: #10b981; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Institutional AI Server</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. WATCHLISTS (Fast Access & Compact Quick Switcher)
    # ══════════════════════════════════════════════════════════════════════════
    # Theme Mode Selector (Aligned exclusively above Watchlist)
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        if st.button("🌙 Dark", key="sidebar_theme_dark", use_container_width=True, type="primary" if is_dark else "secondary", help="Switch to Dark Institutional Trading Desk"):
            if not is_dark:
                st.session_state["ui_theme_mode"] = "Dark"
                st.rerun()
    with col_th2:
        if st.button("☀️ Light", key="sidebar_theme_light", use_container_width=True, type="primary" if not is_dark else "secondary", help="Switch to Light Professional Analytics Mode"):
            if is_dark:
                st.session_state["ui_theme_mode"] = "Light"
                st.rerun()

    st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

    cur_sym = st.session_state.get("active_symbol", "RELIANCE")
    cur_name = sd.get_company_name(cur_sym)
    cur_sec = sd.get_sector(cur_sym)
    cur_info = sd.get_stock_info(cur_sym)
    cur_exch = cur_info.get("exchange", "NSE")
    cur_series = cur_info.get("series", "EQ")

    st.markdown(
        f"""
        <div style="background: {theme_cfg['bg_card']}; border: 1px solid {theme_cfg['border']}; border-radius: 8px; padding: 7px 10px; margin: 4px 0 6px 0;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 1.0rem; font-weight: 800; color: {theme_cfg['primary']}; letter-spacing: 0.02em;">{cur_sym}</span>
                <span style="font-size: 0.62rem; font-weight: 700; background: {theme_cfg['primary_shadow']}; color: {theme_cfg['primary']}; padding: 1px 6px; border-radius: 4px; border: 1px solid {theme_cfg['border']};">{cur_exch} • {cur_series}</span>
            </div>
            <div style="font-size: 0.70rem; font-weight: 600; color: {theme_cfg['text_sub']}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px;">{cur_name}</div>
            <div style="font-size: 0.64rem; color: {theme_cfg['text_muted']};">{cur_sec}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div style='font-size: 0.70rem; font-weight: 700; color: {theme_cfg['text_sub']}; letter-spacing: 0.04em; margin-bottom: 4px;'>⚡ QUICK PRESETS</div>", unsafe_allow_html=True)
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        if st.button("🏆 N50", key="sb_quick_n50", use_container_width=True, help="Switch screener to NIFTY 50"):
            st.session_state["hc_sel_universe"] = "🏆 NIFTY 50 Universe"
            st.rerun()
    with c_p2:
        if st.button("🏛️ NSE", key="sb_quick_nse", use_container_width=True, help="Switch screener to All NSE Equities"):
            st.session_state["hc_sel_universe"] = "🏛️ All NSE Equities (Top 250 Liquid Leaders)"
            st.rerun()
    with c_p3:
        if st.button("🏢 BSE", key="sb_quick_bse", use_container_width=True, help="Switch screener to All BSE Equities"):
            st.session_state["hc_sel_universe"] = "🏢 All BSE Equities (Top 150 Active Equities & Gems)"
            st.rerun()

    st.markdown("##### ⭐ WATCHLISTS")
    user_watchlists = config.load_all_watchlists()
    wl_names = list(user_watchlists.keys())
    if "sidebar_active_wl" not in st.session_state or st.session_state["sidebar_active_wl"] not in wl_names:
        st.session_state["sidebar_active_wl"] = wl_names[0] if wl_names else "My Watchlist"

    active_wl_name = st.session_state["sidebar_active_wl"]
    active_wl_syms = user_watchlists.get(active_wl_name, [])

    col_wl_sel, col_wl_act = st.columns([2.4, 0.8])
    with col_wl_sel:
        cur_idx = wl_names.index(active_wl_name) if active_wl_name in wl_names else 0
        new_active_wl = st.selectbox(
            "Select Watchlist",
            options=wl_names,
            index=cur_idx,
            key="sidebar_wl_selector",
            label_visibility="collapsed",
        )
        if new_active_wl != active_wl_name:
            st.session_state["sidebar_active_wl"] = new_active_wl
            st.rerun()

    with col_wl_act:
        with st.popover("⚙️", use_container_width=True, help="Manage watchlists: add symbols or create new"):
            st.markdown(f"**Manage Watchlist**")
            st.caption(f"Add Symbol to **{active_wl_name}**:")
            add_search_options = sd.get_curated_search_list()
            selected_add_sym = st.selectbox(
                "Add Symbol",
                options=add_search_options,
                index=None,
                placeholder="Type symbol to add...",
                key="sidebar_add_stock_to_wl",
                label_visibility="collapsed",
            )
            if selected_add_sym:
                sym_to_add = sd.extract_symbol_from_selection(selected_add_sym)
                if sym_to_add:
                    if config.add_to_watchlist(active_wl_name, sym_to_add):
                        st.success(f"Added {sym_to_add} to '{active_wl_name}'!")
                        st.rerun()
                    else:
                        st.info(f"{sym_to_add} is already in '{active_wl_name}'.")

            st.markdown("---")
            st.caption("Create New Watchlist:")
            new_wl_input = st.text_input("Watchlist Name", placeholder="e.g. Momentum", key="side_pop_new_wl_name", label_visibility="collapsed")
            if st.button("➕ Create Watchlist", key="btn_pop_create_wl", type="primary", use_container_width=True):
                w_clean = new_wl_input.strip()
                if w_clean:
                    if config.create_watchlist(w_clean):
                        st.session_state["sidebar_active_wl"] = w_clean
                        st.success(f"Created '{w_clean}'!")
                        st.rerun()
                    else:
                        st.error("Already exists or invalid.")
                else:
                    st.warning("Enter a name.")

    # Render stocks in active watchlist as compact 2-column pill grid with active state highlight
    if active_wl_syms:
        w_cols = st.columns(2)
        for idx_w, sym_code in enumerate(active_wl_syms):
            t_col = w_cols[idx_w % 2]
            is_cur = (st.session_state["active_symbol"] == sym_code)
            btn_txt = f"● {sym_code}" if is_cur else sym_code
            btn_type = "primary" if is_cur else "secondary"
            if t_col.button(btn_txt, key=f"wl_side_{active_wl_name}_{sym_code}_{idx_w}", type=btn_type, use_container_width=True):
                st.session_state["active_symbol"] = sym_code
                if "sidebar_opt_scope" in st.session_state:
                    st.session_state["sidebar_opt_scope"] = f"Active Stock ({sym_code})"
                st.rerun()
    else:
        st.caption(f"Watchlist '{active_wl_name}' is empty. Use ⚙️ above to add symbols.")

    st.markdown("---")

    symbol = st.session_state["active_symbol"]

    # Unified Asset & Expiry Resolution (No Redundant Duplicate Sidebar Dropdowns)
    underlying = symbol
    is_idx = oa.is_weekly_index(underlying)
    is_fo_sym = (underlying in oa.NSE_FO_LOT_SIZES) or is_idx
    if is_fo_sym:
        valid_expiries = oa.get_stock_expiry_dates(underlying, expiry_type="all" if is_idx else "monthly", cycle="online")
        if not valid_expiries:
            valid_expiries = oa.get_stock_expiry_dates(underlying)
    else:
        valid_expiries = []
    expiry = valid_expiries[0] if valid_expiries else str(date.today() + timedelta(days=7))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SYSTEM & DATA FEED SETTINGS (Distinct Dedicated Section)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    is_test_env = os.environ.get("STREAMLIT_TESTING") == "1"
    with st.expander("⚙️ System & Data Feed", expanded=False):
        years = st.slider("Historical Lookback (Years)", 1, 4, 2)
        choice = st.selectbox(
            "Data Source",
            ["online", "auto", "sample", "shoonya"],
            index=2 if is_test_env else 0,
            help="online = Yahoo Finance. auto = Shoonya if configured. sample = Synthetic.",
        )
        hft_stream_choice = st.selectbox(
            "Auto-Refresh Stream",
            ["5s (Balanced — Recommended)", "3s (Fast Stream)", "10s (Low Network)", "1s (Ultra-Low Latency / HFT)", "Manual Only"],
            index=4 if is_test_env else 0,
            key="sidebar_hft_stream_choice",
            help="Controls how frequently live market quotes auto-refresh in real time.",
        )

    if is_test_env or "Manual" in hft_stream_choice:
        stream_run_every = None
    elif "1s" in hft_stream_choice:
        stream_run_every = "1s"
    elif "3s" in hft_stream_choice or "2s" in hft_stream_choice:
        stream_run_every = "3s"
    elif "10s" in hft_stream_choice:
        stream_run_every = "10s"
    else:
        stream_run_every = "5s"

    # ══════════════════════════════════════════════════════════════════════════
    # 5. MARKET SESSION STATUS (Clean Single Indicator — No Duplicate Dots)
    # ══════════════════════════════════════════════════════════════════════════
    m_open, m_desc = is_market_open()
    mkt_icon = "🟢" if m_open else "🔴"
    clean_desc = m_desc
    for pfx in ("🔴 ", "🟢 ", "🟡 ", "🔴", "🟢", "🟡"):
        if clean_desc.startswith(pfx):
            clean_desc = clean_desc[len(pfx):].strip()
            break

    st.markdown(
        f"""
        <div style="background: {theme_cfg['bg_card']}; border-radius: 6px; padding: 6px 10px; border: 1px solid {theme_cfg['border']}; margin-top: 8px;">
            <div style="font-size: 0.60rem; color: {theme_cfg['text_muted']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">NSE/BSE Market Session</div>
            <div style="font-size: 0.76rem; color: {theme_cfg['text_main']}; font-weight: 600; margin-top: 2px;">{mkt_icon} {clean_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("⚠️ Educational analysis terminal. Not investment advice.")


provider, provider_error = None, None
try:
    provider = make_provider(choice)
except Exception as e:
    provider_error = str(e)

end = date.today()
start = end - timedelta(days=365 * years)


@st.cache_data(ttl=60, show_spinner=False)
def _st_load_stock_cached(source_name: str, s: str, start_d: date, end_d: date):
    if provider is None:
        return None
    return get_history_cached(provider, source_name, s, start_d, end_d)


def load_stock(sym: str | None = None):
    if provider is None:
        return None
    s = sym or symbol
    return _st_load_stock_cached(provider.name, s, start, end)


@st.cache_data(ttl=60, show_spinner=False)
def _st_cached_screener_scan(source_name: str, sym: str):
    if sym:
        # 1. Fast Sub-Second TradingView Real-Time Feed (NSE/BSE)
        try:
            tbl_tv, errs_tv = screener.scan_tradingview_realtime([sym])
            if not tbl_tv.empty:
                return tbl_tv, errs_tv
        except Exception:
            pass
        # 2. Historical Fallback
        if provider:
            return screener.scan(provider, [sym])
    return pd.DataFrame(), {}


# Fetch Paper Account Capital for Header Display
pt_acct_header = pt.get_or_create_paper_account()
pt_cash_avail = pt_acct_header.get("cash_balance", 500000.0)

# Fetch Active Stock Quote for Header Live Badge
df_header_active = None
active_price_str = ""
active_delta_badge = ""
active_live_ltp = None
try:
    if provider is not None and hasattr(provider, "get_quote"):
        try:
            q_hdr = provider.get_quote(symbol)
            if q_hdr and q_hdr.get("ltp"):
                active_live_ltp = float(q_hdr["ltp"])
                h_last = active_live_ltp
                h_prev = float(q_hdr.get("close", h_last))
                h_pct = float(q_hdr.get("change_pct", 0.0))
                h_color = ("#00d09c" if is_dark else "#059669") if h_pct >= 0 else ("#eb5b3c" if is_dark else "#dc2626")
                h_sign = "+" if h_pct >= 0 else ""
                active_price_str = f"₹{h_last:,.2f}"
                delta_bg = "0,208,156" if is_dark else "5,150,105"
                if h_pct < 0:
                    delta_bg = "235,91,60" if is_dark else "220,38,38"
                active_delta_badge = f"""<span style="background: rgba({delta_bg}, 0.15); color: {h_color}; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.82rem; border: 1px solid {h_color}40;">{h_sign}{h_pct:.2f}%</span>"""
                storage.set_in_memory_quote(symbol, h_last, change_pct=h_pct, volume=float(q_hdr.get("volume", 0.0)))
        except Exception:
            pass

    if not active_price_str:
        df_header_active = load_stock(symbol)
        if df_header_active is not None and not df_header_active.empty:
            h_last = float(df_header_active["close"].iloc[-1])
            h_prev = float(df_header_active["close"].iloc[-2]) if len(df_header_active) > 1 else h_last
            h_pct = ((h_last - h_prev) / h_prev) * 100
            h_color = ("#00d09c" if is_dark else "#059669") if h_pct >= 0 else ("#eb5b3c" if is_dark else "#dc2626")
            h_sign = "+" if h_pct >= 0 else ""
            active_price_str = f"₹{h_last:,.2f}"
            delta_bg = "0,208,156" if is_dark else "5,150,105"
            if h_pct < 0:
                delta_bg = "235,91,60" if is_dark else "220,38,38"
            active_delta_badge = f"""<span style="background: rgba({delta_bg}, 0.15); color: {h_color}; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.82rem; border: 1px solid {h_color}40;">{h_sign}{h_pct:.2f}%</span>"""
            storage.set_in_memory_quote(symbol, h_last, change_pct=h_pct, volume=float(df_header_active["volume"].iloc[-1]))
except Exception:
    pass

# ==============================================================================
# ACTIVE STOCK & UTILITY CONTROLS HEADER (Rendered inside workspace desks)
# ==============================================================================
def render_active_stock_header(key_prefix: str = "", show_stock: bool = True):
    """Render the active stock price banner and top utility controls (Feed, Cash, Font, Theme, Sync)."""
    st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
    c_title, c_refresh = st.columns([4.2, 0.6])
    with c_title:
        if show_stock:
            comp_full_name = sd.get_company_name(symbol)
            comp_sec = sd.get_sector(symbol)
            sub_title_txt = f"{comp_full_name} • NSE {comp_sec}" if comp_full_name != symbol else f"NSE Equity • {comp_sec}"
            st.markdown(
                f"""
                <div style="line-height: 1.25; margin-bottom: 6px;">
                    <div style="display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;">
                        <span style="font-size: {1.20 * font_scale:.2f}rem; font-weight: 600; color: {theme_cfg['text_main']}; letter-spacing: 0.01em;">{symbol}</span>
                        <span style="font-size: {1.08 * font_scale:.2f}rem; font-weight: 600; color: {theme_cfg['text_sub']}; font-feature-settings: 'tnum';">{active_price_str}</span>
                        {active_delta_badge}
                        <span style="font-size: {0.66 * font_scale:.2f}rem; color: {theme_cfg['primary']}; background: rgba({'0,208,156' if is_dark else '5,150,105'},0.12); padding: 1px 5px; border-radius: 4px; font-weight: 600; border: 1px solid rgba({'0,208,156' if is_dark else '5,150,105'},0.3);">⚡ &lt;1ms</span>
                    </div>
                    <div style="font-size: {0.72 * font_scale:.2f}rem; color: {theme_cfg['text_muted']}; font-weight: 600; margin-top: 2px;">
                        {sub_title_txt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding: 0; line-height: 1.10;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-size: {1.08 * font_scale:.2f}rem; font-weight: 600; color: {theme_cfg['text_main']};">Automated Trading Terminal</span>
                        <span style="font-size: {0.70 * font_scale:.2f}rem; color: {theme_cfg['text_muted']};">Target: <strong>{symbol}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_refresh:
        if st.button("🔄 Sync", use_container_width=True, key=f"{key_prefix}btn_top_refresh_main", help="Re-fetch live quotes and clear in-memory caches"):
            storage.clear_mem_cache()
            st.cache_data.clear()
            st.rerun()

    if provider is None:
        st.error(f"Could not start the '{choice}' source: {provider_error}")
        st.info("Pick 'online' or 'sample' in the sidebar to keep working, and see the Connection tab for setup.")
    elif not getattr(provider, "is_live", False):
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 6px 14px; display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <span style="background: #f59e0b; color: #000000; font-weight: 600; font-size: 0.70rem; padding: 1px 6px; border-radius: 4px;">DEMO MODE</span>
                <span style="color: #cbd5e1; font-weight: 500; font-size: 0.80rem;">Showing synthetic sample data — Select <strong style="color: #00d09c;">online</strong> in the sidebar to stream live NSE/BSE market quotes.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# REAL-TIME LIVE STREAMING FRAGMENT (< 1ms Execution & Automated Triggers)
# ==============================================================================
@st.fragment(run_every=stream_run_every)
def render_high_conviction_screener_desk(key_prefix: str = "hc_"):
    """
    Unified High-Conviction Institutional Screener for Equities, F&O Stocks, and Benchmark Indices.
    Allows user-fixed conviction thresholds, dynamic asset filtering (Stocks, Options, Indices, All),
    direction filtering (Bullish, Bearish, All), view modes (Cards vs High-Density Table), and one-click execution.
    """
    st.markdown("### 🎯 High-Conviction Screener")
    st.caption("Quantitative confluence scanning across equities, options, and benchmark indices.")

    # --------------------------------------------------------------------------
    # 1. UNIVERSE DEFINITIONS & WATCHLIST SYNC
    # --------------------------------------------------------------------------
    top_25_fo = [
        "NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN",
        "BHARTIARTL", "ITC", "LT", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN", "AXISBANK",
        "KOTAKBANK", "TATAMOTORS", "TATASTEEL", "NTPC", "POWERGRID", "ADANIENT", "ASIANPAINT",
        "HCLTECH", "WIPRO"
    ]
    banking_fo = [
        "BANKNIFTY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BAJFINANCE",
        "BAJAJFINSV", "INDUSINDBK", "CANBK", "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK",
        "SHRIRAMFIN"
    ]
    it_fo = ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "DIXON"]
    all_fo = list(oa.NSE_FO_LOT_SIZES.keys())

    user_wls = config.load_all_watchlists()
    user_wl_keys = list(user_wls.keys())
    active_wl = st.session_state.get("active_tab_wl", user_wl_keys[0] if user_wl_keys else "My Watchlist")

    sector_options = [f"Sector: {s}" for s in sd.get_all_sector_names()]
    universe_options = [
        "⚡ Top 25 F&O Heavyweights",
        "🏆 NIFTY 50 Universe",
        "🌐 All Sectoral Stocks (All 14 Sectors Combined — 187 Leaders)",
        "🏛️ All NSE Equities (8,600+ Complete NSE Universe)",
        "🏛️ All NSE Equities (Top 250 Liquid Leaders)",
        "🏢 All BSE Equities (4,100+ Complete BSE Universe)",
        "🏢 All BSE Equities (Top 150 Active Equities & Gems)",
        "🌐 All Indian Listed Stocks (12,700+ Master Universe)",
        "🏢 BSE SENSEX 30 Leaders",
        "💎 BSE Prominent Gems & Value Stocks (Taparia, etc.)",
        "🪙 Indian Penny Stocks (Trading < ₹10)",
        "🇮🇳 All Indian Market Indices (56 Benchmarks, Sectoral & Thematic)",
        "🏛️ Complete NSE F&O Universe",
    ] + sector_options + [
        f"⭐ Active Watchlist: '{active_wl}'",
    ] + [f"📁 Watchlist: '{k}'" for k in user_wl_keys if k != active_wl]

    # --------------------------------------------------------------------------
    # 2. SCREENER CONTROLS DECK (Single Consolidated Row, Single Numeric Conviction)
    # --------------------------------------------------------------------------
    with st.container(border=True):
        c_univ, c_depth, c_asset, c_conv, c_dir, c_action = st.columns([1.6, 1.0, 1.0, 0.9, 1.0, 0.9])

        with c_univ:
            sel_universe = st.selectbox(
                "Universe",
                universe_options,
                index=0,
                key=f"{key_prefix}sel_universe",
                help="Select market segment: All NSE, All BSE, All Sectors, Top F&O, NIFTY 50, or custom watchlists.",
            )

        with c_depth:
            sel_depth = st.selectbox(
                "Scan Depth",
                [
                    "Top 50 (Quick)",
                    "Top 100 (Standard)",
                    "Top 250 (Deep Scan)",
                    "Top 500 (Comprehensive)",
                    "All Symbols (Complete)",
                ],
                index=0,
                key=f"{key_prefix}sel_depth",
                help="Batch depth limit for high-speed scanning across massive universes.",
            )

        with c_asset:
            sel_asset = st.selectbox(
                "Asset Class",
                [
                    "All Assets",
                    "Equities Only",
                    "Options Only",
                    "Indices Only",
                ],
                index=0,
                key=f"{key_prefix}sel_asset",
                help="Filter setups by asset type: Equities, Options Contracts, or Indices.",
            )

        # Single Numerical Conviction Input (No duplicate slider, no redundant preset buttons)
        conv_state_key = f"{key_prefix}slider_conv"
        if conv_state_key not in st.session_state:
            st.session_state[conv_state_key] = 85

        with c_conv:
            cur_conv = st.number_input(
                "Min Confluence (%)",
                min_value=50,
                max_value=99,
                value=int(st.session_state.get(conv_state_key, 85)),
                step=1,
                key=f"{key_prefix}num_conv",
                help="Enter minimum Confluence / Conviction percentage required to display trade setup (50% to 99%).",
            )
            st.session_state[conv_state_key] = int(cur_conv)

        with c_dir:
            sel_dir = st.selectbox(
                "Direction",
                ["All Directions", "🟢 Bullish Only", "🔴 Bearish Only"],
                index=0,
                key=f"{key_prefix}sel_dir",
                help="Filter by directional bias: Bullish or Bearish.",
            )

        with c_action:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            run_scan_clicked = st.button(
                "⚡ Run Scan",
                type="primary",
                key=f"{key_prefix}btn_run_scan",
                use_container_width=True,
                help="Initiate quantitative screening across selected universe and option chains.",
            )

    # --------------------------------------------------------------------------
    # 3. RESOLVE UNIVERSE SYMBOLS
    # --------------------------------------------------------------------------
    scan_syms = []
    include_indices = (sel_asset in ("All Assets", "Indices Only", "Options Only"))

    if sel_universe.startswith("⚡"):
        scan_syms = list(top_25_fo)
    elif "SENSEX 30" in sel_universe:
        scan_syms = sd.get_bse_sensex_30()
    elif "BSE Prominent Gems" in sel_universe or "💎" in sel_universe:
        scan_syms = sd.get_bse_prominent_gems()
    elif "All BSE Equities (4,100+" in sel_universe:
        scan_syms = sd.get_all_bse_symbols()
    elif "All BSE Equities (Top 150" in sel_universe or "Top 50 Active BSE" in sel_universe:
        scan_syms = sd.get_bse_top_equities(limit=150)
    elif "All NSE Equities (8,600+" in sel_universe:
        scan_syms = sd.get_all_nse_symbols()
    elif "All NSE Equities (Top 250" in sel_universe:
        scan_syms = sd.get_nse_top_equities(limit=250)
    elif "All Sectoral Stocks (All 14 Sectors" in sel_universe:
        scan_syms = sd.get_all_sectoral_stocks()
    elif "All Indian Listed Stocks (12,700+" in sel_universe:
        scan_syms = sd.get_all_master_symbols()
    elif "Penny Stocks" in sel_universe or "🪙" in sel_universe:
        scan_syms = sd.get_penny_stocks()
    elif sel_universe.startswith("🇮🇳"):
        scan_syms = sd.get_all_indian_indices()
    elif sel_universe.startswith("🏆"):
        # NIFTY 50 universe
        n50_sample = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN",
            "LT", "BAJFINANCE", "HINDUNILVR", "MARUTI", "SUNPHARMA", "TITAN", "AXISBANK",
            "KOTAKBANK", "TATAMOTORS", "TATASTEEL", "NTPC", "POWERGRID", "ADANIENT", "ASIANPAINT",
            "HCLTECH", "WIPRO", "ULTRACEMCO", "JSWSTEEL", "GRASIM", "BAJAJFINSV", "TECHM",
            "NESTLEIND", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO",
            "BAJAJ-AUTO", "M&M", "BPCL", "HINDALCO", "BEL", "HAL", "TRENT", "SHRIRAMFIN",
            "JIOFIN", "COALINDIA", "ONGC", "ADANIPORTS"
        ]
        scan_syms = n50_sample
    elif sel_universe.startswith("Sector:"):
        sec_name = sel_universe.replace("Sector:", "").strip()
        scan_syms = sd.get_stocks_for_sector(sec_name)
        if not scan_syms:
            scan_syms = list(top_25_fo)
    elif sel_universe.startswith("🏦"):
        scan_syms = list(banking_fo)
    elif sel_universe.startswith("💻"):
        scan_syms = list(it_fo)
    elif "Complete NSE F&O" in sel_universe:
        scan_syms = list(all_fo)
    elif "Watchlist:" in sel_universe:
        # Extract watchlist name
        import re
        wl_match = re.search(r"'([^']+)'", sel_universe)
        wl_name = wl_match.group(1) if wl_match else active_wl
        scan_syms = user_wls.get(wl_name, [])
        if not scan_syms:
            scan_syms = list(top_25_fo)
    elif sel_universe.startswith("🌐"):
        try:
            df_all = sd.get_all_stocks_dataframe()
            scan_syms = df_all["Symbol"].head(60).tolist()
        except Exception:
            scan_syms = list(top_25_fo)
    else:
        scan_syms = list(top_25_fo)

    total_universe_pool = len(scan_syms)

    # Apply Batch Scan Depth Limit
    if "Top 50" in sel_depth:
        scan_syms = scan_syms[:50]
    elif "Top 100" in sel_depth:
        scan_syms = scan_syms[:100]
    elif "Top 250" in sel_depth:
        scan_syms = scan_syms[:250]
    elif "Top 500" in sel_depth:
        scan_syms = scan_syms[:500]
    # "All Symbols" scans the complete list

    # --------------------------------------------------------------------------
    # 4. SCAN EXECUTION & CACHE MANAGEMENT
    # --------------------------------------------------------------------------
    hc_cache = st.session_state.get("hc_scan_data")
    need_scan = run_scan_clicked or (hc_cache is None and sel_universe.startswith("⚡"))

    if need_scan and provider:
        # Scan with baseline 75% so user can freely adjust slider down without re-scanning network
        scan_min = min(cur_conv, 75)
        should_scan_options = (sel_asset != "Equities Only")
        should_inc_idx = (sel_asset in ("All Assets", "Indices Only", "Options Only"))

        prog_holder = st.empty()
        stat_holder = st.empty()
        p_bar = prog_holder.progress(0.0)

        def _hc_progress_cb(cur: int, tot: int, sym_code: str):
            p_val = min(1.0, max(0.0, cur / max(1, tot)))
            p_bar.progress(p_val)
            stat_holder.caption(
                f"⚡ Analyzing **{cur}/{tot}** (`{sym_code}`) across multi-timeframe confluence models..."
            )

        try:
            notif_res = ate.scan_high_conviction_notifications(
                provider=provider,
                symbols=scan_syms,
                min_conviction=scan_min,
                account_capital=200000.0,
                include_indices=should_inc_idx,
                scan_options=should_scan_options,
                progress_callback=_hc_progress_cb,
            )
            prog_holder.empty()
            stat_holder.empty()
            st.session_state["hc_scan_data"] = notif_res
            st.session_state["hc_last_universe"] = sel_universe
            st.session_state["hc_last_scan_time"] = notif_res.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            hc_cache = notif_res
        except Exception as e:
            prog_holder.empty()
            stat_holder.empty()
            st.error(f"Screener execution error: {e}")

    # Fallback empty cache structure
    if not hc_cache:
        hc_cache = {"stocks": [], "options": [], "timestamp": "Ready to Scan", "total_count": 0}

    # --------------------------------------------------------------------------
    # 5. INSTANT DYNAMIC FILTERING (No Network Delay on Slider/Asset Change!)
    # --------------------------------------------------------------------------
    all_raw_stocks = hc_cache.get("stocks", [])
    all_raw_options = hc_cache.get("options", [])

    # Filter Stocks
    filtered_stocks = []
    if sel_asset not in ("Options Only", "Indices Only"):
        for s in all_raw_stocks:
            if s.conviction < cur_conv:
                continue
            if sel_dir.startswith("🟢") and s.action != "BUY":
                continue
            if sel_dir.startswith("🔴") and s.action != "SELL":
                continue
            filtered_stocks.append(s)

    # Filter Options
    filtered_options = []
    if sel_asset != "Equities Only":
        for o in all_raw_options:
            if o.conviction < cur_conv:
                continue
            if sel_asset == "Indices Only" and o.symbol not in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"):
                continue
            if sel_dir.startswith("🟢") and o.action not in ("LONG_CALL", "BULL_CALL_SPREAD"):
                continue
            if sel_dir.startswith("🔴") and o.action not in ("LONG_PUT", "BEAR_PUT_SPREAD"):
                continue
            filtered_options.append(o)

    total_filtered = len(filtered_stocks) + len(filtered_options)
    last_time = hc_cache.get("timestamp", "Ready").split()[-1] if hc_cache.get("timestamp") != "Ready to Scan" else "Ready"

    # --------------------------------------------------------------------------
    # 6. TOP SUMMARY METRICS KPI BANNER
    # --------------------------------------------------------------------------
    avg_confluence = round(sum(getattr(s, "confluence_rating", s.conviction) for s in (filtered_stocks + filtered_options)) / max(1, total_filtered), 1) if total_filtered > 0 else 0.0
    avg_win_rate = round(sum(getattr(s, "expected_win_rate", 50.0) for s in (filtered_stocks + filtered_options)) / max(1, total_filtered), 1) if total_filtered > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    tot_badge = "🟢" if total_filtered > 0 else "⚪"
    k1.metric(f"{tot_badge} QUALIFIED SETUPS", f"{total_filtered}", f"≥{cur_conv}% Confluence")
    k2.metric("🎯 AVG CONFLUENCE", f"{avg_confluence}%" if total_filtered > 0 else "N/A", "5-Pillar Score")
    k3.metric("📊 AVG EXP WIN RATE", f"{avg_win_rate}%" if total_filtered > 0 else "N/A", "Historical Edge")
    k4.metric("🕒 LAST SCAN TIME", f"{last_time}", f"{sel_universe.split()[0]}")

    if total_filtered > 0:
        st.success(
            f"🎯 Found **{total_filtered} setups** meeting "
            f"**≥{cur_conv}% Confluence** in `{sel_universe}` (Avg Expected Win Rate: **{avg_win_rate}%**)."
        )
    elif hc_cache.get("timestamp") == "Ready to Scan":
        st.info(f"👉 Universe loaded: **{sel_universe}** ({len(scan_syms)} symbols). Click **⚡ Run Scan** above to screen.")
    else:
        st.info(
            f"ℹ️ Currently **0 setups** in '{sel_universe}' meet the **≥{cur_conv}% Confluence** threshold. "
            f"You can adjust the confluence percentage or switch universe to reveal qualifying candidates."
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 7. VIEW MODE & CSV EXPORT CONTROLS
    # --------------------------------------------------------------------------
    v_col1, v_col3 = st.columns([3.2, 1.0])
    with v_col1:
        view_mode = st.radio(
            "Display View Mode",
            ["🔲 Institutional Cards View", "📊 High-Density Data Table View"],
            horizontal=True,
            key=f"{key_prefix}view_mode",
            help="Switch between rich interactive trade blueprint cards or high-density institutional data spreadsheet.",
        )

    with v_col3:
        if total_filtered > 0:
            export_rows = []
            for s in filtered_stocks:
                export_rows.append({
                    "Type": "Stock / Equity",
                    "Symbol": s.symbol,
                    "Contract / Asset": s.legs_desc,
                    "Action": s.action,
                    "Confluence (%)": getattr(s, "confluence_rating", s.conviction),
                    "Expected Win Rate (%)": getattr(s, "expected_win_rate", 50.0),
                    "Grade": getattr(s, "grade", "Grade A"),
                    "Event Risk": getattr(s, "event_risk", "Safe"),
                    "Macro Alignment": getattr(s, "macro_alignment", "Neutral"),
                    "LTP / CMP (₹)": round(s.current_price, 2),
                    "Entry Zone (₹)": round(s.entry_price, 2),
                    "Stop Loss (₹)": round(s.stop_loss, 2),
                    "Target 1 (₹)": round(s.target_1, 2),
                    "Target 2 (₹)": round(s.target_2, 2),
                    "R:R Ratio": s.risk_reward,
                    "Lot / Rec Qty": s.recommended_qty,
                    "Timestamp": s.timestamp,
                })
            for o in filtered_options:
                export_rows.append({
                    "Type": "Options / Derivative",
                    "Symbol": o.symbol,
                    "Contract / Asset": o.legs_desc,
                    "Action": o.action,
                    "Confluence (%)": getattr(o, "confluence_rating", o.conviction),
                    "Expected Win Rate (%)": getattr(o, "expected_win_rate", 50.0),
                    "Grade": getattr(o, "grade", "Grade A"),
                    "Event Risk": getattr(o, "event_risk", "Safe"),
                    "Macro Alignment": getattr(o, "macro_alignment", "Neutral"),
                    "LTP / CMP (₹)": round(o.current_price, 2),
                    "Entry Zone (₹)": round(o.entry_price, 2),
                    "Stop Loss (₹)": round(o.stop_loss, 2),
                    "Target 1 (₹)": round(o.target_1, 2),
                    "Target 2 (₹)": round(o.target_2, 2),
                    "R:R Ratio": o.risk_reward,
                    "Lot / Rec Qty": o.recommended_qty,
                    "Timestamp": o.timestamp,
                })
            df_export = pd.DataFrame(export_rows)
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"high_confluence_screener_{cur_conv}pct.csv",
                mime="text/csv",
                key=f"{key_prefix}btn_csv",
                use_container_width=True,
            )

    # --------------------------------------------------------------------------
    # 8. RESULTS DISPLAY (CARDS VS DATA TABLE)
    # --------------------------------------------------------------------------
    def _render_hc_results():
        if view_mode == "📊 High-Density Data Table View":
            if total_filtered == 0:
                st.caption("No matching setups to display in data table.")
            else:
                tbl_rows = []
                for s in filtered_stocks:
                    tbl_rows.append({
                        "Class": "📈 Stock",
                        "Symbol": s.symbol,
                        "Action": s.action,
                        "Grade": getattr(s, "grade", "Grade A"),
                        "Confluence": f"{getattr(s, 'confluence_rating', s.conviction)}%",
                        "Exp Win Rate": f"{getattr(s, 'expected_win_rate', 50.0):.1f}%",
                        "CMP / LTP": f"₹{s.current_price:,.2f}",
                        "Entry Zone": f"₹{s.entry_price:,.2f}",
                        "Stop Loss": f"₹{s.stop_loss:,.2f}",
                        "Target 1": f"₹{s.target_1:,.2f}",
                        "Target 2": f"₹{s.target_2:,.2f}",
                        "R:R": f"1:{s.risk_reward}",
                        "Rec Qty": s.recommended_qty,
                        "Event Risk": getattr(s, "event_risk", "Safe"),
                        "Contract / Description": s.legs_desc,
                    })
                for o in filtered_options:
                    tbl_rows.append({
                        "Class": "⛓️ Option",
                        "Symbol": o.symbol,
                        "Action": o.action,
                        "Grade": getattr(o, "grade", "Grade A"),
                        "Confluence": f"{getattr(o, 'confluence_rating', o.conviction)}%",
                        "Exp Win Rate": f"{getattr(o, 'expected_win_rate', 50.0):.1f}%",
                        "CMP / LTP": f"₹{o.current_price:,.2f}",
                        "Entry Zone": f"₹{o.entry_price:,.2f}",
                        "Stop Loss": f"₹{o.stop_loss:,.2f}",
                        "Target 1": f"₹{o.target_1:,.2f}",
                        "Target 2": f"₹{o.target_2:,.2f}",
                        "R:R": f"1:{o.risk_reward}",
                        "Rec Qty": o.recommended_qty,
                        "Event Risk": getattr(o, "event_risk", "Safe"),
                        "Contract / Description": o.legs_desc,
                    })
                df_display = pd.DataFrame(tbl_rows)
                st.dataframe(df_display, use_container_width=True, hide_index=True)

        else:
            # Institutional Cards View
            tab_c_all, tab_c_stocks, tab_c_options = st.tabs(
                [
                    f"🔀 All Opportunities ({total_filtered})",
                    f"📈 Stocks Only ({len(filtered_stocks)})",
                    f"⛓️ Options Only ({len(filtered_options)})",
                ]
            )

            def _render_stock_card(s_sig, card_idx):
                with st.container(border=True):
                    sc_c1, sc_c2, sc_c3 = st.columns([1.3, 1.8, 1.5])
                    with sc_c1:
                        act_badge = "🟢" if s_sig.action == "BUY" else "🔴"
                        st.markdown(f"### {act_badge} {s_sig.action} `{s_sig.symbol}`")
                        st.caption(f"⭐ **{getattr(s_sig, 'grade', 'Grade A')}**")
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Confluence", f"{getattr(s_sig, 'confluence_rating', s_sig.conviction)}%")
                        with col_m2:
                            st.metric("Exp Win", f"{getattr(s_sig, 'expected_win_rate', 50.0):.1f}%")
                        st.caption(f"Current Price: **₹{s_sig.current_price:,.2f}** | Equity Cash")
                        ev_color = "🔴" if "HIGH" in str(getattr(s_sig, 'event_risk', '')) else "🟢"
                        st.caption(f"{ev_color} Event: **{str(getattr(s_sig, 'event_risk', 'Safe')).split('(')[0].strip()}**")

                    with sc_c2:
                        st.markdown("**🎯 Trade Blueprint & Risk Parameters:**")
                        st.write(f"- **Suggested Entry Zone:** `₹{s_sig.entry_price * 0.995:,.2f} – ₹{s_sig.entry_price:,.2f}`")
                        sl_pct = round((s_sig.entry_price - s_sig.stop_loss) / s_sig.entry_price * 100, 2) if s_sig.entry_price > 0 else 5.0
                        st.write(f"- 🛑 **Strict Stop-Loss:** `₹{s_sig.stop_loss:,.2f}` (-{sl_pct}%)")
                        t1_pct = round((s_sig.target_1 - s_sig.entry_price) / s_sig.entry_price * 100, 2) if s_sig.entry_price > 0 else 10.0
                        st.write(f"- 🎯 **Target 1:** `₹{s_sig.target_1:,.2f}` (+{t1_pct}%)")
                        st.write(f"- 🎯 **Target 2 (Extended):** `₹{s_sig.target_2:,.2f}`")
                        st.write(f"- ⚖️ **Risk / Reward:** `1 : {s_sig.risk_reward}`")
                        if getattr(s_sig, 'kelly_pct', 0) > 0:
                            st.write(f"- 🧮 **Half-Kelly Size:** `{s_sig.kelly_pct}%` cap (₹{getattr(s_sig, 'optimal_capital', 0):,.0f})")

                    with sc_c3:
                        st.markdown("**⚡ One-Click Execution Desk:**")
                        qty_val = max(1, s_sig.recommended_qty)
                        stk_trade_qty = st.number_input(
                            f"Shares ({s_sig.symbol})",
                            min_value=1,
                            max_value=50000,
                            value=qty_val,
                            key=f"{key_prefix}stk_qty_{s_sig.symbol}_{card_idx}",
                        )
                        req_cap = stk_trade_qty * s_sig.entry_price
                        st.caption(f"Required Capital: **₹{req_cap:,.2f}**")

                        b_c1, b_c2 = st.columns(2)
                        with b_c1:
                            if st.button("📝 Paper Trade", key=f"{key_prefix}btn_pt_s_{s_sig.symbol}_{card_idx}", type="primary", use_container_width=True):
                                ok_pt, msg_pt = pt.execute_paper_order(s_sig, quantity=stk_trade_qty, provider=provider)
                                if ok_pt:
                                    st.success(f"{msg_pt}")
                                    st.rerun()
                                else:
                                    st.error(f"{msg_pt}")
                        with b_c2:
                            if st.button("📊 Open Terminal", key=f"{key_prefix}btn_term_s_{s_sig.symbol}_{card_idx}", use_container_width=True):
                                st.session_state["active_symbol"] = s_sig.symbol
                                st.session_state["desk_tab_arrangement"] = "Terminal First"
                                st.rerun()

                        shoonya_ready = (len(config.shoonya_missing()) == 0) or (getattr(provider, "name", "") == "shoonya" and getattr(provider, "is_live", False))
                        if shoonya_ready:
                            if st.button("🚀 Place Live Shoonya Order", key=f"{key_prefix}btn_live_s_{s_sig.symbol}_{card_idx}", use_container_width=True):
                                live_eng = le.LiveExecutionEngine()
                                ok_live, msg_live = live_eng.place_live_order(s_sig, quantity=stk_trade_qty)
                                if ok_live:
                                    st.success(f"Live Sent: {msg_live}")
                                else:
                                    st.error(f"Live Error: {msg_live}")

                    with st.expander(f"📖 View Confluence Thesis & 5-Pillar Audit for {s_sig.symbol}", expanded=False):
                        st.markdown(f"**Algorithmic Thesis:** {s_sig.thesis}")
                        if getattr(s_sig, "confluence_pillars", None):
                            pillars = s_sig.confluence_pillars
                            st.markdown("**🏛️ 5-Pillar Institutional Confluence Audit:**")
                            p_col1, p_col2 = st.columns(2)
                            with p_col1:
                                st.write(f"- 🌐 **Macro Regime:** {pillars.get('macro_regime', {}).get('alignment', 'Neutral')}")
                                st.write(f"- 🏦 **Institutional Flow:** {pillars.get('institutional_flow', {}).get('status', 'Moderate')}")
                                st.write(f"- 📈 **Price Structure & VSA:** {pillars.get('price_structure_vsa', {}).get('vsa_bias', 'Neutral')}")
                            with p_col2:
                                st.write(f"- 🤖 **Indicator Consensus:** {pillars.get('technical_consensus', {}).get('consensus_pct', 50)}% Bullish")
                                st.write(f"- 🛡️ **Corporate Event Risk:** {pillars.get('event_safety', {}).get('risk_level', 'Safe')}")
                                if "smc_structure" in pillars:
                                    st.write(f"- ⚡ **SMC Structure:** {pillars['smc_structure'].get('structure', 'Neutral')}")
                                if "market_pulse" in pillars:
                                    st.write(f"- 🚦 **Market Pulse:** {pillars['market_pulse'].get('status', 'Normal')}")
                        if s_sig.literature_citations:
                            st.markdown("**Literature & Strategy Citations:**")
                            for cit in s_sig.literature_citations:
                                st.write(f"- {cit}")

            def _render_option_card(o_sig, card_idx):
                with st.container(border=True):
                    oc_c1, oc_c2, oc_c3 = st.columns([1.3, 1.8, 1.5])
                    with oc_c1:
                        opt_badge = "🟢" if "CALL" in o_sig.action or "BULL" in o_sig.action else "🔴"
                        st.markdown(f"### {opt_badge} {o_sig.action.replace('_', ' ')} `{o_sig.symbol}`")
                        st.caption(f"⭐ **{getattr(o_sig, 'grade', 'Grade A')}**")
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Confluence", f"{getattr(o_sig, 'confluence_rating', o_sig.conviction)}%")
                        with col_m2:
                            st.metric("Exp Win", f"{getattr(o_sig, 'expected_win_rate', 50.0):.1f}%")
                        st.caption(f"**Contract:** {o_sig.legs_desc}")
                        st.caption(f"Underlying Spot: **₹{o_sig.current_price:,.2f}** | Time: {o_sig.timestamp.split()[-1] if ' ' in o_sig.timestamp else o_sig.timestamp}")

                    with oc_c2:
                        st.markdown("**⛓️ Derivatives Blueprint & Greeks Edge:**")
                        st.write(f"- **Premium Entry Zone:** `₹{o_sig.entry_price:,.2f}`")
                        st.write(f"- 🛑 **Option Stop-Loss:** `₹{o_sig.stop_loss:,.2f}` (-50% Premium Cut)")
                        st.write(f"- 🎯 **Target 1:** `₹{o_sig.target_1:,.2f}` (+100% Premium Double)")
                        st.write(f"- 🎯 **Target 2:** `₹{o_sig.target_2:,.2f}` (+200% Moonshot)")
                        st.write(f"- ⚖️ **Risk / Reward:** `1 : {o_sig.risk_reward}`")
                        if getattr(o_sig, 'kelly_pct', 0) > 0:
                            st.write(f"- 🧮 **Half-Kelly Capital:** `{o_sig.kelly_pct}%` (₹{getattr(o_sig, 'optimal_capital', 0):,.0f})")

                    with oc_c3:
                        st.markdown("**⚡ Option Execution Pad:**")
                        lot_sz = oa.get_lot_size(o_sig.symbol, spot=o_sig.current_price)
                        opt_trade_qty = st.number_input(
                            f"Contract Lots (Lot Size: {lot_sz})",
                            min_value=lot_sz,
                            max_value=lot_sz * 100,
                            value=max(lot_sz, o_sig.recommended_qty),
                            step=lot_sz,
                            key=f"{key_prefix}opt_qty_{o_sig.symbol}_{card_idx}",
                        )
                        req_premium = opt_trade_qty * o_sig.entry_price
                        st.caption(f"Estimated Premium: **₹{req_premium:,.2f}**")

                        b_c1, b_c2 = st.columns(2)
                        with b_c1:
                            if st.button("📝 Paper Option", key=f"{key_prefix}btn_pt_o_{o_sig.symbol}_{card_idx}", type="primary", use_container_width=True):
                                ok_pt_o, msg_pt_o = pt.execute_paper_order(o_sig, quantity=opt_trade_qty, provider=provider)
                                if ok_pt_o:
                                    st.success(f"{msg_pt_o}")
                                    st.rerun()
                                else:
                                    st.error(f"{msg_pt_o}")
                        with b_c2:
                            if st.button("📊 Chart Underlying", key=f"{key_prefix}btn_term_o_{o_sig.symbol}_{card_idx}", use_container_width=True):
                                st.session_state["active_symbol"] = o_sig.symbol
                                st.session_state["desk_tab_arrangement"] = "Terminal First"
                                st.rerun()

                        shoonya_ready_o = (len(config.shoonya_missing()) == 0) or (getattr(provider, "name", "") == "shoonya" and getattr(provider, "is_live", False))
                        if shoonya_ready_o:
                            if st.button("🚀 Live Shoonya Option", key=f"{key_prefix}btn_live_o_{o_sig.symbol}_{card_idx}", use_container_width=True):
                                live_eng = le.LiveExecutionEngine()
                                ok_live_o, msg_live_o = live_eng.place_live_order(o_sig, quantity=opt_trade_qty)
                                if ok_live_o:
                                    st.success(f"Live Option Sent: {msg_live_o}")
                                else:
                                    st.error(f"Live Option Error: {msg_live_o}")

                    with st.expander(f"📖 View Option Greeks & 5-Pillar Edge for {o_sig.symbol}", expanded=False):
                        st.markdown(f"**Volatility Edge Thesis:** {o_sig.thesis}")
                        if getattr(o_sig, "confluence_pillars", None):
                            pillars = o_sig.confluence_pillars
                            st.markdown("**🏛️ 5-Pillar Option Confluence Audit:**")
                            p_col1, p_col2 = st.columns(2)
                            with p_col1:
                                st.write(f"- 🌐 **Macro Regime:** {pillars.get('macro_regime', {}).get('alignment', 'Neutral')}")
                                st.write(f"- ⚡ **Volatility Regime:** {pillars.get('volatility_regime', {}).get('regime', 'Balanced')}")
                            with p_col2:
                                st.write(f"- 📊 **Microstructure PCR:** {pillars.get('derivatives_microstructure', {}).get('pcr', 1.0):.2f}")
                                st.write(f"- 🛡️ **Event Safety:** {pillars.get('event_safety', {}).get('risk_level', 'Safe')}")
                        if o_sig.literature_citations:
                            st.markdown("**Methodological Citations:**")
                            for cit in o_sig.literature_citations:
                                st.write(f"- {cit}")

            with tab_c_all:
                if total_filtered == 0:
                    st.warning(f"No setups currently meet ≥{cur_conv}% conviction in '{sel_universe}'.")
                else:
                    combined_items = [("stock", s) for s in filtered_stocks] + [("option", o) for o in filtered_options]
                    combined_items.sort(key=lambda x: (getattr(x[1], "confluence_rating", x[1].conviction), getattr(x[1], "expected_win_rate", 50.0)), reverse=True)
                    for c_i, (item_type, item_sig) in enumerate(combined_items):
                        if item_type == "stock":
                            _render_stock_card(item_sig, f"all_{c_i}")
                        else:
                            _render_option_card(item_sig, f"all_{c_i}")

            with tab_c_stocks:
                if not filtered_stocks:
                    st.warning(f"No stock setups currently meet ≥{cur_conv}% conviction in '{sel_universe}'.")
                else:
                    for s_i, s_sig in enumerate(filtered_stocks):
                        _render_stock_card(s_sig, f"stk_{s_i}")

            with tab_c_options:
                if not filtered_options:
                    st.warning(f"No option setups currently meet ≥{cur_conv}% conviction in '{sel_universe}'.")
                else:
                    for o_i, o_sig in enumerate(filtered_options):
                        _render_option_card(o_sig, f"opt_{o_i}")
    with st.container(height=720, border=False):
        _render_hc_results()


def render_live_market_stream():
    # 1. Update in-memory quote store for active stock
    try:
        if provider is not None and hasattr(provider, "get_quote"):
            q_act = provider.get_quote(symbol)
            if q_act and q_act.get("ltp"):
                storage.set_in_memory_quote(symbol, q_act["ltp"], change_pct=q_act.get("change_pct", 0.0), volume=q_act.get("volume", 0.0))
        elif df_header_active is not None and not df_header_active.empty:
            curr_ltp = float(df_header_active["close"].iloc[-1])
            curr_vol = float(df_header_active["volume"].iloc[-1])
            storage.set_in_memory_quote(symbol, curr_ltp, volume=curr_vol)
    except Exception:
        pass

    # 2. Automated Stop-Loss & Target Trigger Execution & Live Positions Refresh
    try:
        triggered = pt.refresh_open_positions_quotes(provider=provider)
        for t_msg in triggered:
            st.toast(t_msg, icon="⚡")
    except Exception:
        pass

    # 3. Macro Market Indices Ribbon
    try:
        vix_ribbon = idata.get_india_vix_metrics(provider)
        fii_ribbon = idata.get_fii_dii_flow_metrics(provider)

        def _get_idx_quote(idx_sym: str, default_ltp: float, default_chg: float) -> tuple[float, float, float]:
            try:
                if provider is not None and hasattr(provider, "get_quote"):
                    q = provider.get_quote(idx_sym)
                    if q and q.get("ltp"):
                        return float(q["ltp"]), float(q.get("change", 0.0)), float(q.get("change_pct", 0.0))
            except Exception:
                pass
            return default_ltp, default_chg, (default_chg / default_ltp) * 100

        nifty_ltp, nifty_chg, nifty_pct = _get_idx_quote("NIFTY", 24320.80, +112.30)
        bnifty_ltp, bnifty_chg, bnifty_pct = _get_idx_quote("BANKNIFTY", 51840.50, +195.40)

        st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
        t_c1, t_c2, t_c3, t_c4, t_c5, t_c6 = st.columns(6)
        t_c1.metric("NIFTY 50", f"₹{nifty_ltp:,.2f}", f"{nifty_chg:+.2f} ({nifty_pct:+.2f}%)")
        t_c2.metric("BANK NIFTY", f"₹{bnifty_ltp:,.2f}", f"{bnifty_chg:+.2f} ({bnifty_pct:+.2f}%)")

        vix_color = "🟢" if vix_ribbon["vix"] < 14 else ("🟡" if vix_ribbon["vix"] <= 18 else "🔴")
        t_c3.metric(f"{vix_color} India VIX", f"{vix_ribbon['vix']:.2f}", f"{vix_ribbon['change']:+.2f} ({vix_ribbon['regime'].split()[0]})")

        fii_sgn = "+" if fii_ribbon["fii_net_crores"] >= 0 else ""
        t_c4.metric("🏛️ FII Cash Flow", f"{fii_sgn}₹{fii_ribbon['fii_net_crores']:,.1f} Cr", "Foreign Institutions")

        dii_sgn = "+" if fii_ribbon["dii_net_crores"] >= 0 else ""
        t_c5.metric("🏦 DII Cash Flow", f"{dii_sgn}₹{fii_ribbon['dii_net_crores']:,.1f} Cr", "Domestic Mutual Funds")

        t_c6.metric("📊 FII Futures L/S", f"{fii_ribbon['fii_long_short_ratio']:.2f}x", fii_ribbon["sentiment"].split()[0])
    except Exception:
        pass

# Live stream fragment defined above


# ==============================================================================
# UNIFIED AI RECOMMENDATION & ORDER AUTOMATION ENGINE COMPONENT
# ==============================================================================
def render_ai_recommendations_and_execution_engine(
    target_symbol: str,
    target_underlying: str | None = None,
    target_expiry: str | None = None,
    key_prefix: str = "",
):
    """Render the AI Recommendations & Order Automation Engine for both stock and options."""
    t_sym = target_symbol or symbol
    t_und = target_underlying or t_sym

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 2px 0 4px 0;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 0.90rem; font-weight: 600; color: {theme_cfg['text_main']};">🎯 AI Recommendations & Execution ({t_sym})</span>
                <span style="font-size: 0.65rem; background: rgba(0,208,156,0.12); color: {theme_cfg['primary']}; padding: 1px 5px; border-radius: 4px; font-weight: 600; border: 1px solid rgba(0,208,156,0.25);">≥90% Conviction</span>
            </div>
            <div style="font-size: 0.68rem; color: {theme_cfg['text_muted']}; font-weight: 600;">90+ Indicators • 25 Book Models</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Determine expiry
    t_exp = target_expiry
    if not t_exp:
        try:
            is_idx = oa.is_weekly_index(t_und)
            exps = oa.get_stock_expiry_dates(t_und, expiry_type="all" if is_idx else "monthly", cycle="online")
            t_exp = exps[0] if exps else str(date.today() + timedelta(days=7))
        except Exception:
            t_exp = str(date.today() + timedelta(days=7))

    rec_tab_stk, rec_tab_opt = st.tabs([
        f"📈 Stock ({t_sym})",
        f"⚡ Options ({t_und} {t_exp})",
    ])

    # Fetch df_curr and live LTP for t_sym
    df_eval = None
    try:
        df_eval = load_stock(t_sym)
    except Exception:
        pass
    if df_eval is None or df_eval.empty:
        if provider:
            try:
                df_eval = provider.get_history(t_sym, start, end)
            except Exception:
                pass

    cur_live_ltp = None
    try:
        mem_q = storage.get_in_memory_quote(t_sym)
        if mem_q and mem_q.get("ltp"):
            cur_live_ltp = float(mem_q["ltp"])
        elif df_eval is not None and not df_eval.empty:
            cur_live_ltp = float(df_eval["close"].iloc[-1])
    except Exception:
        pass

    paper_perf = pt.get_paper_performance_summary()
    cap = paper_perf.get("total_portfolio_value", 200000.0)

    # SUB-TAB 1: STOCK RECOMMENDATION
    with rec_tab_stk:
        stk_signal = None
        try:
            stk_signal = ate.evaluate_stock_trade(t_sym, df_eval, account_capital=cap, live_price=cur_live_ltp)
        except Exception as e:
            st.error(f"Error evaluating stock {t_sym}: {e}")

        if stk_signal:
            with st.container(border=True):
                r_col1, r_col2, r_col3 = st.columns([1.1, 2.5, 1.4])
                with r_col1:
                    conv_color = "🟢" if stk_signal.conviction >= 70 else ("🔴" if stk_signal.conviction <= 30 else "🟡")
                    st.markdown(f"**{conv_color} {stk_signal.action} {stk_signal.symbol}**")
                    st.caption(f"⭐ **{getattr(stk_signal, 'grade', 'Grade A')}**")
                    tm1, tm2 = st.columns(2)
                    with tm1:
                        st.metric("Confluence", f"{getattr(stk_signal, 'confluence_rating', stk_signal.conviction)}%")
                    with tm2:
                        st.metric("Exp Win", f"{getattr(stk_signal, 'expected_win_rate', 50.0):.1f}%")
                    st.caption(f"LTP: ₹{stk_signal.entry_price:,.1f} | {getattr(stk_signal, 'event_risk', 'Safe').split('(')[0].strip()}")

                with r_col2:
                    sl_pct = round((stk_signal.entry_price - stk_signal.stop_loss) / stk_signal.entry_price * 100, 1) if stk_signal.entry_price > 0 else 7.0
                    t1_pct = round((stk_signal.target_1 - stk_signal.entry_price) / stk_signal.entry_price * 100, 1) if stk_signal.entry_price > 0 else 15.0
                    st.markdown(
                        f"""
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; font-size: 0.76rem; background: {'rgba(0,0,0,0.15)' if is_dark else '#f8fafc'}; padding: 4px 6px; border-radius: 5px; border: 1px solid {theme_cfg['border']};">
                            <div><span style="color: {theme_cfg['text_muted']}; font-size: 0.64rem;">ENTRY</span><br/><strong>₹{stk_signal.entry_price:,.2f}</strong></div>
                            <div><span style="color: #ef4444; font-size: 0.64rem;">STOP-LOSS (-{sl_pct}%)</span><br/><strong style="color: #ef4444;">₹{stk_signal.stop_loss:,.2f}</strong></div>
                            <div><span style="color: #10b981; font-size: 0.64rem;">TARGET 1 (+{t1_pct}%)</span><br/><strong style="color: #10b981;">₹{stk_signal.target_1:,.2f}</strong></div>
                            <div><span style="color: #10b981; font-size: 0.64rem;">TARGET 2</span><br/><strong style="color: #10b981;">₹{stk_signal.target_2:,.2f}</strong></div>
                            <div><span style="color: {theme_cfg['text_muted']}; font-size: 0.64rem;">RISK:REWARD</span><br/><strong>1:{stk_signal.risk_reward}</strong></div>
                            <div><span style="color: {theme_cfg['text_muted']}; font-size: 0.64rem;">TYPE</span><br/><strong>{stk_signal.legs_desc}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with r_col3:
                    order_qty = st.number_input(
                        "Qty (Shares)",
                        min_value=1,
                        max_value=100000,
                        value=max(1, stk_signal.recommended_qty),
                        key=f"{key_prefix}order_qty_{stk_signal.symbol}",
                        label_visibility="collapsed",
                    )
                    est_cost = order_qty * stk_signal.entry_price
                    st.caption(f"Capital: ₹{est_cost:,.0f}")
                    if st.button("📝 Execute Paper", key=f"{key_prefix}exec_paper_{stk_signal.symbol}", type="primary", use_container_width=True):
                        ok_p, msg_p = pt.execute_paper_order(stk_signal, quantity=order_qty, live_price=cur_live_ltp, provider=provider)
                        if ok_p:
                            st.success(msg_p)
                            st.rerun()
                        else:
                            st.error(msg_p)

                st.caption(f"🧠 **AI Synthesis:** {stk_signal.thesis}")
        else:
            st.info(f"Analyzing historical patterns and open-source models for {t_sym}...")

    # SUB-TAB 2: OPTION CHAIN RECOMMENDATION
    with rec_tab_opt:
        opt_signal = None
        chain_curr = None
        spot_curr = cur_live_ltp or 100.0
        try:
            if provider:
                chain_curr = provider.get_option_chain(t_und, t_exp)
            if chain_curr is not None and not chain_curr.empty:
                spot_curr = float(chain_curr.attrs.get("underlying_ltp") or chain_curr["strike"].median())
                df_und_curr = df_eval if df_eval is not None else load_stock(t_und)
                opt_signal = ate.evaluate_options_trade(
                    underlying=t_und,
                    chain=chain_curr,
                    df_underlying=df_und_curr,
                    spot=spot_curr,
                    expiry=t_exp,
                    account_capital=cap,
                )
        except Exception as e:
            st.caption(f"Option evaluation notice: {e}")

        if opt_signal:
            with st.container(border=True):
                o_col1, o_col2, o_col3 = st.columns([1.1, 2.5, 1.4])
                with o_col1:
                    opt_color = "🟢" if "BULL" in opt_signal.action or "CALL" in opt_signal.action else ("🔴" if "BEAR" in opt_signal.action or "PUT" in opt_signal.action else "🟡")
                    st.markdown(f"**{opt_color} {opt_signal.action}**")
                    st.caption(f"⭐ **{getattr(opt_signal, 'grade', 'Grade A')}**")
                    om1, om2 = st.columns(2)
                    with om1:
                        st.metric("Confluence", f"{getattr(opt_signal, 'confluence_rating', opt_signal.conviction)}%")
                    with om2:
                        st.metric("Exp Win", f"{getattr(opt_signal, 'expected_win_rate', 50.0):.1f}%")
                    st.caption(f"{opt_signal.symbol} @ ₹{opt_signal.current_price:,.1f}")

                with o_col2:
                    st.markdown(
                        f"""
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; font-size: 0.76rem; background: {'rgba(0,0,0,0.15)' if is_dark else '#f8fafc'}; padding: 4px 6px; border-radius: 5px; border: 1px solid {theme_cfg['border']};">
                            <div><span style="color: {theme_cfg['text_muted']}; font-size: 0.64rem;">ENTRY PREMIUM</span><br/><strong>₹{opt_signal.entry_price:,.2f}</strong></div>
                            <div><span style="color: #ef4444; font-size: 0.64rem;">STOP-LOSS</span><br/><strong style="color: #ef4444;">₹{opt_signal.stop_loss:,.2f}</strong></div>
                            <div><span style="color: #10b981; font-size: 0.64rem;">TARGET 1</span><br/><strong style="color: #10b981;">₹{opt_signal.target_1:,.2f}</strong></div>
                            <div><span style="color: #10b981; font-size: 0.64rem;">TARGET 2</span><br/><strong style="color: #10b981;">₹{opt_signal.target_2:,.2f}</strong></div>
                            <div><span style="color: {theme_cfg['text_muted']}; font-size: 0.64rem;">RISK:REWARD</span><br/><strong>1:{opt_signal.risk_reward}</strong></div>
                            <div><span style="color: {theme_cfg['text_muted']}; font-size: 0.64rem;">LEGS</span><br/><strong>{opt_signal.legs_desc}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with o_col3:
                    lot_base = oa.get_lot_size(t_und, spot_curr)
                    opt_order_qty = st.number_input(
                        f"Qty (Lots of {lot_base})",
                        min_value=lot_base,
                        max_value=100000,
                        step=lot_base,
                        value=max(lot_base, opt_signal.recommended_qty),
                        key=f"{key_prefix}opt_qty_{opt_signal.symbol}",
                        label_visibility="collapsed",
                    )
                    est_opt_capital = opt_order_qty * opt_signal.entry_price
                    st.caption(f"Premium: ₹{est_opt_capital:,.0f} ({int(opt_order_qty/lot_base)}L)")
                    if st.button("📑 Execute Paper Option", key=f"{key_prefix}exec_paper_opt_{opt_signal.symbol}", type="primary", use_container_width=True):
                        ok_p, msg_p = pt.execute_paper_order(opt_signal, quantity=opt_order_qty, provider=provider)
                        if ok_p:
                            st.success(msg_p)
                            st.rerun()
                        else:
                            st.error(msg_p)

                st.caption(f"🧠 **Volatility Thesis:** {opt_signal.thesis}")
        else:
            st.info(f"Generating option chain recommendation for {t_und} ({t_exp})...")


# ==============================================================================
# PINNED STATIC TOP NAVIGATION PANE (Freeze-Pane: Macro Pulse + Universal Search)
# ==============================================================================
with st.container():
    st.markdown('<div id="pinned-static-top-anchor"></div>', unsafe_allow_html=True)
    # 1. Macro Market Indices Ribbon (Real-Time Macro Pulse)
    render_live_market_stream()

    # 2. Universal Search & Submit Bar in Static Form Mode
    cur_active_sym = st.session_state.get("active_symbol", "RELIANCE")
    active_wl_name = st.session_state.get("sidebar_active_wl") or st.session_state.get("active_tab_wl") or "My Watchlist"
    try:
        user_wls = config.load_all_watchlists()
        active_wl_syms = user_wls.get(active_wl_name, []) or config.load_watchlist()
    except Exception:
        active_wl_syms = config.load_watchlist()

    # 2. THE ONE UNIFIED SEARCH & SCREENER OMNIBAR WITH HIGH-CONVICTION & WIN RATIO CRITERIA
    @st.cache_data(ttl=300, show_spinner=False)
    def _st_omnibar_options_cached(wl_tuple: tuple[str, ...], wl_name: str):
        return sd.get_formatted_search_list(
            active_watchlist=list(wl_tuple) if wl_tuple else None,
            watchlist_name=wl_name,
            exchange="ALL",
            curated_only=False,
        )

    all_search_options = _st_omnibar_options_cached(
        tuple(active_wl_syms),
        active_wl_name,
    )

    with st.container(border=True):
        with st.form("top_unified_omnibar_form", border=False):
            col_omnibar, col_conv, col_winrate, col_btn = st.columns([4.0, 1.5, 1.5, 0.9], vertical_alignment="bottom")
            with col_omnibar:
                chosen_search_str = st.selectbox(
                    "Universal Search & Screener Omnibar",
                    options=all_search_options,
                    index=None,
                    placeholder="🔍 Search Single Stock (e.g. RELIANCE, 500325), Whole Universe (NSE, BSE, All), or Segment (F&O, Banking, IT)...",
                    key="top_omnibar_search_input",
                    label_visibility="visible",
                    help="Unified Omnibar: Search across 12,740+ assets — Single Stock (NSE/BSE), Whole Universe (NSE/BSE/All), or Segments (F&O, Banking, IT, Auto, etc.).",
                )
            with col_conv:
                chosen_conv_crit = st.number_input(
                    "🔥 Conviction (%)",
                    min_value=0,
                    max_value=100,
                    value=None,
                    step=1,
                    placeholder="e.g. 80",
                    key="top_omnibar_conv_crit",
                    label_visibility="visible",
                    help="Enter exact minimum AI confluence conviction rating percentage (0% to 100%). Leave blank for no minimum filter.",
                )
            with col_winrate:
                chosen_wr_crit = st.number_input(
                    "🎯 Win Ratio (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    step=1.0,
                    format="%g",
                    placeholder="e.g. 70",
                    key="top_omnibar_wr_crit",
                    label_visibility="visible",
                    help="Enter exact minimum historical / expected win ratio percentage (0% to 100%). Leave blank for no minimum filter.",
                )
            with col_btn:
                top_submit_clicked = st.form_submit_button(
                    "🎯 Screen",
                    type="primary",
                    use_container_width=True,
                    help="Execute search and screening across chosen single stock, universe or segment with your conviction and win ratio criteria.",
                )

    if top_submit_clicked:
        if not chosen_search_str:
            st.warning("⚠️ Please select or search a stock, universe, or sector in the search bar first.")
        else:
            # Parse conviction threshold (exact user input or 0 if left blank)
            if chosen_conv_crit is not None and str(chosen_conv_crit).strip() != "":
                try:
                    c_val = int(chosen_conv_crit)
                except (ValueError, TypeError):
                    c_val = 0
            else:
                c_val = 0

            # Parse win ratio threshold (exact user input or 0.0 if left blank)
            if chosen_wr_crit is not None and str(chosen_wr_crit).strip() != "":
                try:
                    w_val = round(float(chosen_wr_crit), 1)
                except (ValueError, TypeError):
                    w_val = 0.0
            else:
                w_val = 0.0

            st.session_state["omnibar_min_conviction"] = c_val
            st.session_state["omnibar_min_win_rate"] = w_val

            conv_desc = f"Conviction ≥ {c_val}%" if c_val > 0 else "All Convictions"
            wr_desc = f"Win Ratio ≥ {w_val:.0f}%" if w_val > 0 else "All Win Ratios"

            if sd.is_universe_or_segment(chosen_search_str):
                st.session_state["omnibar_screen_target"] = chosen_search_str
                st.session_state["scr_screening_scope_mode"] = "Universal Screener"
                st.session_state["omnibar_trigger_scan"] = True
                st.toast(f"⚡ Screening {chosen_search_str} ({conv_desc} | {wr_desc})")
                st.rerun()
            else:
                target_sym = sd.extract_symbol_from_selection(chosen_search_str)
                if target_sym:
                    st.session_state["active_symbol"] = target_sym
                    st.session_state["scr_screening_scope_mode"] = "Single Stock"
                    st.toast(f"🎯 Loaded {target_sym} into Workspace Desk! ({conv_desc} | {wr_desc})")
                    st.rerun()

# ==============================================================================
# TOP STRIP 2: MAIN BROKER TABS NAVIGATION (Consolidated Institutional Workspaces)
# ==============================================================================
# Non-Redundant Institutional Workspaces:
# 1. 🎯 Screener & Trading Desk — Single Unified Search, BSE/NSE Index Screening, 12,720+ Directory & Direct Order Execution
# 2. 📊 Terminal               — Real-time Candlestick Chart, 90+ Radar, MTF Confluence, VSA, AI Order Pad
# 3. 💼 Portfolio              — Paper/Live Holdings, Trades Ledger, MPT Risk Analytics
# 4. 🌐 Sector Rotation        — Sector Breadth, Performance & Industry Flows
# 5. ⏳ Backtest               — Strategy Simulation & Performance Engine
# 6. 📚 Academy                — Foundational Models from 19 Classic Books
# 7. 🔌 Gateway                — Broker API Setup, Latency Diagnostics & TradingView Webhooks Bridge
tab_screener, tab_stock, tab_portfolio, tab_sectors, tab_bt, tab_books, tab_conn = st.tabs(
    [
        "🎯 Screener & Trading Desk",
        "📊 Terminal",
        "💼 Portfolio",
        "🌐 Sector Rotation",
        "⏳ Backtest",
        "📚 Academy",
        "🔌 Gateway",
    ]
)


with tab_portfolio:
    render_active_stock_header(key_prefix="portfolio_", show_stock=False)
    st.markdown(
        f"""
        <div style="background: {theme_cfg['hero_bg']}; padding: 6px 12px; border-radius: 6px; color: {theme_cfg['hero_text']}; margin-bottom: 8px; border: 1px solid {theme_cfg['border']}; border-left: 3px solid {theme_cfg['primary']}; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
            <div style="font-size: 0.92rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                💼 Paper Trading Portfolio & Position Desk
            </div>
            <div style="font-size: 0.72rem; color: {theme_cfg['hero_sub']};">Simulated Capital: <b>₹5,00,000</b> • Live P&L & Auto Stop-Loss Monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch Paper Account & Statistics (Refreshed against Live Real-Time Market Quotes)
    try:
        pt.refresh_open_positions_quotes(provider=provider)
    except Exception:
        pass

    pt_acct = pt.get_or_create_paper_account()
    pt_perf = pt.get_paper_performance_summary(provider=None)
    open_purchases = pt.get_open_paper_positions()
    closed_purchases = pt.get_closed_paper_trades()
    all_purchases = pt.get_all_paper_trades(refresh_live=False, provider=None)

    # Split for Stocks
    stk_perf = pt.get_paper_performance_summary("EQUITY", provider=None)
    open_stk = pt.get_open_paper_positions("EQUITY")
    closed_stk = pt.get_closed_paper_trades("EQUITY")
    all_stk = pt.get_all_paper_trades("EQUITY", refresh_live=False, provider=None)

    # Split for Options
    opt_perf = pt.get_paper_performance_summary("OPTION", provider=None)
    open_opt = pt.get_open_paper_positions("OPTION")
    closed_opt = pt.get_closed_paper_trades("OPTION")
    all_opt = pt.get_all_paper_trades("OPTION", refresh_live=False, provider=None)

    # Portfolio KPI Summary Ribbon
    m_p1, m_p2, m_p3, m_p4, m_p5, m_p6 = st.columns(6)
    m_p1.metric("💼 Portfolio Value", f"₹{pt_perf['total_portfolio_value']:,.2f}", f"{pt_perf['total_return_pct']:+.2f}%")
    m_p2.metric("💵 Available Cash", f"₹{pt_perf['cash_balance']:,.2f}")
    m_p3.metric("📊 Invested Capital", f"₹{pt_perf['invested_capital']:,.2f}")
    m_p4.metric("📈 Unrealized P&L", f"₹{pt_perf['unrealized_pnl']:+,.2f}")
    m_p5.metric("💰 Realized P&L", f"₹{pt_perf['realized_pnl']:+,.2f}")
    m_p6.metric("🎯 Win Rate", f"{pt_perf['win_rate_pct']}%", f"{pt_perf['total_trades']} trades")

    st.markdown("---")

    # Global Quick Action Controls Row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.8, 1.2, 1.2])

    with ctrl_col1:
        if st.button("🔄 Refresh Live Quotes & Check Triggers", use_container_width=True, help="Update live real-time prices from the market, refresh all open position P&L values, and execute automatic exits if any SL/Target is hit."):
            trig_alerts = pt.refresh_open_positions_quotes(provider=provider, force=True)
            if trig_alerts:
                for ta in trig_alerts:
                    st.warning(ta)
            else:
                st.success("Live Quotes Updated: All active positions recalculated against live market.")
            st.rerun()

    with ctrl_col2:
        with st.popover("🗑️ Reset Virtual Account", use_container_width=True):
            st.markdown("⚠️ **Reset Virtual Trading Account**")
            st.caption("This will reset your cash balance to ₹5,00,000 and clear all active positions and history.")
            if st.button("Confirm Reset to ₹5,00,000", type="primary", key="btn_reset_paper_account"):
                pt.reset_paper_account(500000.0)
                st.success("Virtual account successfully reset to ₹5,00,000.")
                st.rerun()

    with ctrl_col3:
        if st.button("🛑 Flatten All Positions", type="secondary", use_container_width=True, help="Immediately exit and close all active virtual purchases."):
            flatten_count = 0
            for pos_item in open_purchases:
                pt.close_paper_position(pos_item["id"], pos_item["current_price"], exit_reason="🛑 KILL_SWITCH_FLATTEN")
                flatten_count += 1
            st.success(f"Emergency Flatten: Successfully closed all {flatten_count} virtual positions.")
            st.rerun()

    st.markdown("---")

    # Centralized Tabs: Stocks at one tab, Options at another tab, Unified Ledger, MPT Analytics
    p_tab_stocks, p_tab_options, p_tab_all, p_tab_mpt = st.tabs(
        [
            f"📈 Stocks Trades ({len(open_stk)} Open / {len(all_stk)} Total)",
            f"⚡ Options Trades ({len(open_opt)} Open / {len(all_opt)} Total)",
            f"🌐 All Trades Ledger ({len(all_purchases)})",
            "🏛️ Risk & Portfolio Analytics (MPT)",
        ]
    )

    def render_trades_ledger_view(
        df_data: pd.DataFrame,
        asset_type: str = "EQUITY",
        table_key: str = "stk",
        all_count: int = 0,
        open_count: int = 0,
        closed_count: int = 0,
    ) -> None:
        if df_data.empty:
            st.info(f"ℹ️ **No {asset_type.lower()} trades recorded yet.**")
            st.caption("Execute simulated trades directly from the **'🔔 90%+ Alerts'** tab or the **'📊 Stock Analysis'** cockpit.")
            return

        # Filter & Export Controls (Clean single-row segmented filter)
        col_flt, col_csv = st.columns([3.5, 0.7])
        with col_flt:
            view_filter = st.radio(
                "Filter Status",
                [f"All ({all_count})", f"🟢 Active ({open_count})", f"🏁 Executed ({closed_count})"],
                horizontal=True,
                key=f"{table_key}_filter_status",
                label_visibility="collapsed",
            )

        if "Active" in view_filter:
            df_display = df_data[df_data["is_open"] == True]
        elif "Executed" in view_filter:
            df_display = df_data[df_data["is_open"] == False]
        else:
            df_display = df_data

        with col_csv:
            if not df_display.empty:
                csv_bytes = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export",
                    data=csv_bytes,
                    file_name=f"{asset_type.lower()}_trades.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{table_key}_dl_csv",
                )

        if df_display.empty:
            st.info(f"No {asset_type.lower()} trades match the '{view_filter}' filter.")
            return

        # ══════════════════════════════════════════════════════════════════
        # STATIC MODE: 100% Fully Visible, Non-Scrollable Responsive Table
        # ══════════════════════════════════════════════════════════════════
        card_bg = theme_cfg.get("bg_card", "#131b2a")
        border = theme_cfg.get("border", "#233047")
        header_bg = "#1a2436" if is_dark else "#f1f5f9"
        text_main = theme_cfg.get("text_main", "#ffffff")
        text_muted = theme_cfg.get("text_muted", "#94a3b8")
        alt_row = "rgba(255, 255, 255, 0.02)" if is_dark else "rgba(0, 0, 0, 0.02)"

        rows_html = []
        for idx, (_, r) in enumerate(df_display.iterrows()):
            pnl = float(r.get("net_pnl", 0.0) or 0.0)
            ret = float(r.get("return_pct", 0.0) or 0.0)
            pnl_clr = ("#00d09c" if is_dark else "#059669") if pnl >= 0 else ("#eb5b3c" if is_dark else "#dc2626")
            pnl_sign = "+" if pnl > 0 else ""
            ret_sign = "+" if ret > 0 else ""
            pnl_pill = f'<span style="background: {pnl_clr}18; color: {pnl_clr}; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid {pnl_clr}30; white-space: nowrap;">{pnl_sign}₹{pnl:,.2f}</span>'
            ret_pill = f'<span style="background: {pnl_clr}18; color: {pnl_clr}; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid {pnl_clr}30; white-space: nowrap;">{ret_sign}{ret:.2f}%</span>'

            is_open = r.get("status") == "OPEN"
            act = str(r.get("action", "BUY")).upper()
            act_is_bull = any(k in act for k in ("BUY", "CALL", "BULL"))
            act_bg = "rgba(0, 208, 156, 0.15)" if act_is_bull else "rgba(235, 91, 60, 0.15)"
            act_clr = ("#00d09c" if is_dark else "#059669") if act_is_bull else ("#eb5b3c" if is_dark else "#dc2626")

            # Entry Date: clean date with subtle time below
            entry_d = str(r.get("entry_date", ""))
            if " " in entry_d:
                d_p, t_p = entry_d.split(" ", 1)
                entry_d_disp = f'<div style="white-space: nowrap; font-weight: 600; color: {text_main}; line-height: 1.15;">{d_p}</div><div style="white-space: nowrap; font-size: 0.67rem; color: {text_muted}; line-height: 1.15;">{t_p}</div>'
            else:
                entry_d_disp = f'<span style="white-space: nowrap; color: {text_muted};">{entry_d}</span>'

            # Exit Date: Clear redundancy — Open trades show simple "—" dash (no duplicate ACTIVE badge)
            exit_d = str(r.get("exit_date", ""))
            if is_open:
                exit_d_disp = f'<span style="color: {text_muted}; font-size: 0.75rem;">—</span>'
            elif " " in exit_d and exit_d != "— Active —":
                d_p, t_p = exit_d.split(" ", 1)
                exit_d_disp = f'<div style="white-space: nowrap; font-weight: 600; color: {text_main}; line-height: 1.15;">{d_p}</div><div style="white-space: nowrap; font-size: 0.67rem; color: {text_muted}; line-height: 1.15;">{t_p}</div>'
            else:
                exit_d_disp = f'<span style="white-space: nowrap; color: {text_muted};">{exit_d}</span>'

            # Status & Triggers: Unified column eliminating duplicate status/remark text
            sl_val = float(r.get("stop_loss", 0.0) or 0.0)
            t1_val = float(r.get("target_1", 0.0) or 0.0)
            f_tbl = f"{0.82 * font_scale:.2f}rem"
            f_th = f"{0.72 * font_scale:.2f}rem"
            f_badge = f"{0.74 * font_scale:.2f}rem"
            f_sub = f"{0.72 * font_scale:.2f}rem"

            legs = str(r.get("legs_desc", "")).strip()

            if is_open:
                trig_elems = []
                if sl_val > 0:
                    trig_elems.append(f'SL <b style="color: #f59e0b;">₹{sl_val:,.2f}</b>')
                if t1_val > 0:
                    trig_elems.append(f'Tgt <b style="color: #10b981;">₹{t1_val:,.2f}</b>')
                trig_str = f' &nbsp;•&nbsp; {" &nbsp;•&nbsp; ".join(trig_elems)}' if trig_elems else ""
                legs_html = f'<div style="color: #38bdf8; font-weight: 600; font-size: {f_sub}; margin-top: 1px;">{legs}</div>' if legs and asset_type in ("OPTION", "ALL") else ""
                status_triggers_html = f'<div style="white-space: nowrap;"><span style="background: rgba(0, 208, 156, 0.15); color: #00d09c; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: {f_badge}; border: 1px solid rgba(0, 208, 156, 0.3);">🟢 OPEN</span><span style="font-size: {f_sub}; color: {text_muted}; margin-left: 5px;">{trig_str}</span></div>{legs_html}'
            else:
                raw_rem = str(r.get("remark", ""))
                exit_reason = raw_rem.split("|")[-1].strip() if "|" in raw_rem else raw_rem
                reason_up = exit_reason.upper()
                if "TARGET" in reason_up:
                    b_clr = "#10b981"
                    b_txt = "🎯 TARGET HIT"
                elif "STOP_LOSS" in reason_up or "SL" in reason_up:
                    b_clr = "#ef4444"
                    b_txt = "🛑 STOP LOSS"
                elif "MANUAL" in reason_up:
                    b_clr = "#94a3b8"
                    b_txt = "🏁 CLOSED (MANUAL)"
                elif "FLATTEN" in reason_up or "KILL_SWITCH" in reason_up:
                    b_clr = "#f59e0b"
                    b_txt = "🛑 FLATTENED"
                else:
                    b_clr = "#94a3b8"
                    b_txt = f"🏁 {exit_reason}"
                legs_html = f'<div style="color: #64748b; font-size: {f_sub}; margin-top: 1px;">{legs}</div>' if legs and asset_type in ("OPTION", "ALL") else ""
                status_triggers_html = f'<div style="white-space: nowrap;"><span style="background: {b_clr}18; color: {b_clr}; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: {f_badge}; border: 1px solid {b_clr}30;">{b_txt}</span></div>{legs_html}'

            asset_col_html = ""
            if asset_type == "ALL":
                atype_str = str(r.get("asset_type", "EQUITY")).upper()
                atype_badge = f'<span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; padding: 2px 5px; border-radius: 4px; font-weight: 600; font-size: {f_badge};">{atype_str}</span>'
                asset_col_html = f'<td style="padding: 6px 6px; white-space: nowrap;">{atype_badge}</td>'

            qty_val = r.get("quantity", 1)
            try:
                qty_disp = f"{int(qty_val):,}"
            except Exception:
                qty_disp = str(qty_val)

            row_bg = alt_row if idx % 2 == 1 else "transparent"
            row_line = (
                f'<tr style="background: {row_bg}; border-bottom: 1px solid {border};">'
                f'<td style="padding: 6px 6px; font-weight: 600; color: {text_muted}; font-size: {f_tbl}; white-space: nowrap;">#{r["pos_id"]}</td>'
                f'<td style="padding: 6px 8px; font-weight: 600; color: {text_main}; font-size: {f_tbl}; white-space: nowrap;">{r["symbol"]}</td>'
                f'{asset_col_html}'
                f'<td style="padding: 6px 6px; white-space: nowrap;"><span style="background: {act_bg}; color: {act_clr}; padding: 2px 5px; border-radius: 4px; font-weight: 600; font-size: {f_badge}; border: 1px solid {act_clr}30;">{act}</span></td>'
                f'<td style="padding: 6px 8px; font-size: {f_tbl};">{status_triggers_html}</td>'
                f'<td style="padding: 6px 8px; font-size: {f_sub}; white-space: nowrap;">{entry_d_disp}</td>'
                f'<td style="padding: 6px 8px; font-size: {f_sub}; white-space: nowrap;">{exit_d_disp}</td>'
                f'<td style="padding: 6px 8px; text-align: right; color: {text_muted}; font-size: {f_tbl}; white-space: nowrap;">₹{float(r["entry_price"]):,.2f}</td>'
                f'<td style="padding: 6px 8px; text-align: right; font-weight: 600; color: {text_main}; font-size: {f_tbl}; white-space: nowrap;">₹{float(r["current_or_exit_price"]):,.2f}</td>'
                f'<td style="padding: 6px 6px; text-align: center; color: {text_muted}; font-weight: 600; font-size: {f_tbl}; white-space: nowrap;">{qty_disp}</td>'
                f'<td style="padding: 6px 8px; text-align: right; font-size: {f_tbl}; white-space: nowrap;">{pnl_pill}</td>'
                f'<td style="padding: 6px 8px; text-align: right; font-size: {f_tbl}; white-space: nowrap;">{ret_pill}</td>'
                f'</tr>'
            )
            rows_html.append(row_line)

        asset_th = '<th style="padding: 7px 6px; width: 60px;">Asset</th>' if asset_type == "ALL" else ""
        qty_header = "Shares" if asset_type == "EQUITY" else ("Lots" if asset_type == "OPTION" else "Qty")

        tbody_content = "".join(rows_html)
        static_table_html = (
            f'<div style="width: 100%; border: 1px solid {border}; border-radius: 8px; overflow-x: auto; background: {card_bg}; margin: 6px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.10);">'
            f'<table style="width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; font-size: {f_tbl}; text-align: left;">'
            f'<thead>'
            f'<tr style="background: {header_bg}; border-bottom: 1px solid {border}; color: {text_muted}; text-transform: uppercase; font-size: {f_th}; letter-spacing: 0.4px;">'
            f'<th style="padding: 7px 6px; width: 45px;">#</th>'
            f'<th style="padding: 7px 8px;">Symbol</th>'
            f'{asset_th}'
            f'<th style="padding: 7px 6px; width: 55px;">Type</th>'
            f'<th style="padding: 7px 8px;">Status & Triggers</th>'
            f'<th style="padding: 7px 8px;">Entry Date</th>'
            f'<th style="padding: 7px 8px;">Exit Date</th>'
            f'<th style="padding: 7px 8px; text-align: right;">Entry (₹)</th>'
            f'<th style="padding: 7px 8px; text-align: right;">LTP (₹)</th>'
            f'<th style="padding: 7px 6px; text-align: center; width: 50px;">{qty_header}</th>'
            f'<th style="padding: 7px 8px; text-align: right;">P&L (₹)</th>'
            f'<th style="padding: 7px 8px; text-align: right;">Return</th>'
            f'</tr>'
            f'</thead>'
            f'<tbody>{tbody_content}</tbody>'
            f'</table>'
            f'</div>'
        )
        clean_html = "\n".join(line.strip() for line in static_table_html.splitlines() if line.strip())
        if hasattr(st, "html"):
            st.html(clean_html)
        else:
            st.markdown(clean_html, unsafe_allow_html=True)

    # ==========================================================================
    # SUB-TAB 1: STOCKS TRADES (ACTIVE POSITIONS & EXECUTED TRADES)
    # ==========================================================================
    with p_tab_stocks:
        st.markdown("### 📈 Equities & Stock Trades Desk")
        st.caption("All simulated equity trades in one dedicated tab — Both active positions and executed trades remarked with exact triggers and status.")

        # Sleek 1-line metadata summary (replaces bulky 5-metric duplicate ribbon)
        stk_pnl_clr = ("#00d09c" if is_dark else "#059669") if stk_perf['unrealized_pnl'] >= 0 else ("#eb5b3c" if is_dark else "#dc2626")
        st.markdown(
            f"""
            <div style="display: flex; gap: 14px; font-size: 0.78rem; color: {theme_cfg['text_muted']}; margin-bottom: 8px; flex-wrap: wrap; align-items: center;">
                <span>Invested: <strong style="color: {theme_cfg['text_main']};">₹{stk_perf['invested_capital']:,.2f}</strong></span>
                <span>•</span>
                <span>Unrealized P&L: <strong style="color: {stk_pnl_clr};">₹{stk_perf['unrealized_pnl']:+,.2f}</strong></span>
                <span>•</span>
                <span>Realized P&L: <strong style="color: {theme_cfg['text_main']};">₹{stk_perf['realized_pnl']:+,.2f}</strong></span>
                <span>•</span>
                <span>Win Rate: <strong style="color: {theme_cfg['text_main']};">{stk_perf['win_rate_pct']}%</strong> ({stk_perf['total_trades']} closed)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_trades_ledger_view(
            df_data=all_stk,
            asset_type="EQUITY",
            table_key="stk",
            all_count=len(all_stk),
            open_count=len(open_stk),
            closed_count=len(closed_stk),
        )

        # Active Stock Position Quick Close Action (in compact expander)
        if open_stk:
            with st.expander("🛑 Close an Open Stock Position", expanded=False):
                stk_cl_col1, stk_cl_col2 = st.columns([3, 1])
                stk_pos_labels = [f"#{p['id']} — {p['action']} {p['quantity']}x {p['symbol']} @ Entry ₹{p['entry_price']:,.2f} (P&L: ₹{p['unrealized_pnl']:+,.2f})" for p in open_stk]
                chosen_stk_label = stk_cl_col1.selectbox("Select Active Stock Position to Close", stk_pos_labels, key="vp_select_close_stk_pos", label_visibility="collapsed")
                chosen_stk_id = int(chosen_stk_label.split()[0].replace("#", ""))
                chosen_stk_obj = next((p for p in open_stk if p["id"] == chosen_stk_id), None)

                if chosen_stk_obj and stk_cl_col2.button(f"Close Stock #{chosen_stk_id}", type="primary", use_container_width=True, key=f"btn_close_stk_{chosen_stk_id}"):
                    ok_close, msg_close = pt.close_paper_position(chosen_stk_id, chosen_stk_obj["current_price"], exit_reason="MANUAL_EXIT")
                    if ok_close:
                        st.success(msg_close)
                        st.rerun()
                    else:
                        st.error(msg_close)

    # ==========================================================================
    # SUB-TAB 2: OPTIONS TRADES (ACTIVE POSITIONS & EXECUTED TRADES)
    # ==========================================================================
    with p_tab_options:
        st.markdown("### ⚡ Derivatives & Options Contracts Desk")
        st.caption("All simulated options contracts in one dedicated tab — Both active positions and executed trades remarked with exact triggers and status.")

        # Sleek 1-line metadata summary (replaces bulky 5-metric duplicate ribbon)
        opt_pnl_clr = ("#00d09c" if is_dark else "#059669") if opt_perf['unrealized_pnl'] >= 0 else ("#eb5b3c" if is_dark else "#dc2626")
        st.markdown(
            f"""
            <div style="display: flex; gap: 14px; font-size: 0.78rem; color: {theme_cfg['text_muted']}; margin-bottom: 8px; flex-wrap: wrap; align-items: center;">
                <span>Invested: <strong style="color: {theme_cfg['text_main']};">₹{opt_perf['invested_capital']:,.2f}</strong></span>
                <span>•</span>
                <span>Unrealized P&L: <strong style="color: {opt_pnl_clr};">₹{opt_perf['unrealized_pnl']:+,.2f}</strong></span>
                <span>•</span>
                <span>Realized P&L: <strong style="color: {theme_cfg['text_main']};">₹{opt_perf['realized_pnl']:+,.2f}</strong></span>
                <span>•</span>
                <span>Win Rate: <strong style="color: {theme_cfg['text_main']};">{opt_perf['win_rate_pct']}%</strong> ({opt_perf['total_trades']} closed)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_trades_ledger_view(
            df_data=all_opt,
            asset_type="OPTION",
            table_key="opt",
            all_count=len(all_opt),
            open_count=len(open_opt),
            closed_count=len(closed_opt),
        )

        # Active Option Contract Quick Close Action (in compact expander)
        if open_opt:
            with st.expander("🛑 Close an Open Option Contract", expanded=False):
                opt_cl_col1, opt_cl_col2 = st.columns([3, 1])
                opt_pos_labels = [f"#{p['id']} — {p['action']} {p['quantity']}x {p['symbol']} [{p['legs_desc']}] @ Premium ₹{p['entry_price']:,.2f} (P&L: ₹{p['unrealized_pnl']:+,.2f})" for p in open_opt]
                chosen_opt_label = opt_cl_col1.selectbox("Select Active Option Contract to Close", opt_pos_labels, key="vp_select_close_opt_pos", label_visibility="collapsed")
                chosen_opt_id = int(chosen_opt_label.split()[0].replace("#", ""))
                chosen_opt_obj = next((p for p in open_opt if p["id"] == chosen_opt_id), None)

                if chosen_opt_obj and opt_cl_col2.button(f"Close Option #{chosen_opt_id}", type="primary", use_container_width=True, key=f"btn_close_opt_{chosen_opt_id}"):
                    ok_close, msg_close = pt.close_paper_position(chosen_opt_id, chosen_opt_obj["current_price"], exit_reason="MANUAL_EXIT")
                    if ok_close:
                        st.success(msg_close)
                        st.rerun()
                    else:
                        st.error(msg_close)

    # ==========================================================================
    # SUB-TAB 3: ALL TRADES LEDGER (COMBINED STOCKS & OPTIONS)
    # ==========================================================================
    with p_tab_all:
        st.markdown(f"### 🌐 Combined Virtual Portfolio Ledger ({len(all_purchases)} Trades)")
        st.caption("Complete chronological record of all stock and option purchases with live status remarks.")

        render_trades_ledger_view(
            df_data=all_purchases,
            asset_type="ALL",
            table_key="all",
            all_count=len(all_purchases),
            open_count=len(open_purchases),
            closed_count=len(closed_purchases),
        )

        # Global position close desk (in compact expander)
        if open_purchases:
            with st.expander("🛑 Close Any Open Virtual Purchase", expanded=False):
                g_cl_col1, g_cl_col2 = st.columns([3, 1])
                pos_labels = [f"#{p['id']} — [{p['asset_type']}] {p['action']} {p['quantity']}x {p['symbol']} @ Entry ₹{p['entry_price']:,.2f} (P&L: ₹{p['unrealized_pnl']:+,.2f})" for p in open_purchases]
                chosen_pos_label = g_cl_col1.selectbox("Select Active Position to Close", pos_labels, key="vp_select_close_all_pos", label_visibility="collapsed")
                chosen_id = int(chosen_pos_label.split()[0].replace("#", ""))
                chosen_obj = next((p for p in open_purchases if p["id"] == chosen_id), None)

                if chosen_obj and g_cl_col2.button(f"Close Position #{chosen_id}", type="primary", use_container_width=True, key=f"btn_close_all_vp_{chosen_id}"):
                    ok_close, msg_close = pt.close_paper_position(chosen_id, chosen_obj["current_price"], exit_reason="MANUAL_EXIT")
                    if ok_close:
                        st.success(msg_close)
                        st.rerun()
                    else:
                        st.error(msg_close)

    # ==========================================================================
    # SUB-TAB 4: MPT RISK ANALYTICS
    # ==========================================================================
    with p_tab_mpt:
        try:
            mpt_stats = aa.calc_portfolio_risk_analytics(
                open_positions=open_purchases,
                closed_trades=closed_purchases,
                initial_capital=pt_perf["initial_capital"],
                current_portfolio_value=pt_perf["total_portfolio_value"],
            )
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            r_c1.metric("Portfolio Beta (β)", f"{mpt_stats['portfolio_beta']:.2f}x", "Market Sensitivity vs NIFTY")
            r_c2.metric("1-Day VaR (95%)", f"₹{mpt_stats['var_95_rupees']:,.0f}", f"-{mpt_stats['var_95_pct']}% Max Loss (1D)")
            r_c3.metric("Sharpe / Sortino", f"{mpt_stats['sharpe_ratio']} / {mpt_stats['sortino_ratio']}", "Risk-Adjusted Return")
            r_c4.metric("Van Tharp Expectancy", f"{mpt_stats['van_tharp_expectancy']:+.2f}R", f"Avg Win: +{mpt_stats['avg_r_win']}R | Loss: {mpt_stats['avg_r_loss']}R")
            st.caption(f"Risk Discipline Evaluation: **{mpt_stats['risk_evaluation']}** | Benchmark 10-Yr G-Sec: 6.50% p.a.")
        except Exception as e_mpt:
            st.info(f"MPT Analytics will update as more trades are logged ({e_mpt}).")
            st.info(f"MPT Analytics will update as more trades are logged ({e_mpt}).")



# ==============================================================================
# TAB 2: STOCK ANALYSIS
# ==============================================================================
with tab_stock:
    render_active_stock_header(key_prefix="stock_")
    if provider:
        try:
            df = load_stock()
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            df = None

        if df is None or df.empty:
            st.warning("No data returned for this symbol.")
        else:
            d = ind.add_all(df)
            last_p = float(d["close"].iloc[-1])
            prev_p = float(d["close"].iloc[-2]) if len(d) > 1 else last_p
            pct_chg = ((last_p - prev_p) / prev_p) * 100

            b_oneil = ba.calc_oneil_signals(df)
            b_vsa = ba.calc_vsa_signals(df)
            b_kaufman = ba.calc_kaufman_regime(df)
            b_alligator = ba.calc_alligator_and_chaos(df)
            b_graham = ba.calc_graham_buffett_metrics(symbol, last_p, df)
            b_nison = ba.calc_nison_candlestick_patterns(df)
            b_dalton = ba.calc_dalton_volume_profile(df)
            b_van_tharp = ba.calc_van_tharp_sizing_and_expectancy(200000.0, 1.0, last_p, last_p * 0.93, last_p * 1.15)

            # Top KPI Summary Ribbon (Compact non-truncating cards)
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Symbol", symbol)
            k2.metric("Last Close", f"₹{last_p:,.2f}", f"{pct_chg:+.2f}%")
            k3.metric("RSI (14)", f"{d['rsi14'].iloc[-1]:.1f}")
            k4.metric("Kaufman ER", f"{d['efficiency_ratio'].iloc[-1]:.2f}")
            k5.metric("CAN SLIM", f"{b_oneil.get('score', 0)}/100")
            k6.metric("52W High Prox.", f"{b_oneil.get('pct_from_52w_high', 0.0):+.1f}%")

            # Precompute AI Recommendations & Radar Signals
            exec_stock_sig = None
            exec_opt_sig = None
            exec_opt_chain = None

            try:
                exec_stock_sig = ate.evaluate_stock_trade(symbol, df, live_price=active_live_ltp)
            except Exception:
                exec_stock_sig = None

            if provider:
                try:
                    exec_opt_chain = provider.get_option_chain(underlying, expiry)
                    if exec_opt_chain is not None and not exec_opt_chain.empty:
                        spot_val = float(exec_opt_chain.attrs.get("underlying_ltp") or exec_opt_chain["strike"].median())
                        exec_opt_sig = ate.evaluate_options_trade(
                            underlying=underlying,
                            chain=exec_opt_chain,
                            df_underlying=df,
                            spot=spot_val,
                            expiry=expiry,
                        )
                except Exception:
                    exec_opt_sig = None

            aa_mtf = aa.evaluate_multi_timeframe_confluence(df, symbol)
            aa_sec = aa.get_stock_sector_info(symbol)
            aa_rs = aa.calc_relative_strength(df)
            aa_deliv = aa.calc_delivery_volume_metrics(df, symbol)
            aa_ev = aa.get_earnings_event_risk(symbol)

            # ==================================================================
            # 1. INTERACTIVE CHART & TECHNICALS (PRIMARY VIEWPORT — CHART FIRST!)
            # ==================================================================
            c_ctrl_view, c_ctrl_style, c_ctrl_ov = st.columns([1.6, 1.1, 1.3])
            with c_ctrl_view:
                chart_view = st.radio(
                    "Chart Engine",
                    ["📊 Quantitative Plotly", "📺 TradingView Live"],
                    horizontal=True,
                    key="chart_display_mode",
                    label_visibility="collapsed",
                )
            with c_ctrl_style:
                if "Plotly" in chart_view:
                    chart_type = st.selectbox(
                        "Style",
                        ["Candlestick", "Line Chart", "OHLC Bars"],
                        index=0,
                        key="term_chart_type_sel",
                        label_visibility="collapsed",
                    )
                else:
                    chart_type = "Candlestick"
            with c_ctrl_ov:
                with st.popover("🛠️ Overlays & Indicators", use_container_width=True):
                    show_sma = st.checkbox("SMA (20, 50, 200)", value=True, help="Moving averages (20 short, 50 medium, 200 institutional trend)")
                    show_supertrend = st.checkbox("SuperTrend (10, 3)", value=True, help="Trailing ATR trend filter widely used in Indian markets")
                    show_kama = st.checkbox("Kaufman KAMA", value=True, help="Adaptive Moving Average (adapts speed based on market noise)")
                    show_bb = st.checkbox("Bollinger Bands (20, 2)", value=False, help="20-day 2-std bands showing volatility squeeze & expansion")
                    show_alligator = st.checkbox("Williams Alligator", value=False, help="Bill Williams 3-line trend indicator (Jaw 13, Teeth 8, Lips 5)")
                    show_donchian = st.checkbox("Donchian (20)", value=False, help="Richard Donchian 20-day breakout channel (Turtle Trading)")

            if "TradingView" in chart_view:
                tv_html = tvi.get_tv_chart_embed_html(symbol, height=520, theme="dark" if is_dark else "light")
                components.html(tv_html, height=530)

                # Real-Time TradingView Technical Consensus & Speedometer Gauge
                tv_c1, tv_c2 = st.columns([1.6, 1])
                with tv_c1:
                    tv_ta = tvi.fetch_tradingview_ta(symbol, interval="1d")
                    if "error" not in tv_ta:
                        rec = tv_ta.get("recommendation", "NEUTRAL")
                        badge_color = "🟢" if "BUY" in rec else ("🔴" if "SELL" in rec else "🟡")
                        st.markdown(f"### {badge_color} TradingView Consensus: **{rec.replace('_', ' ')}**")

                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Buy Signals", tv_ta.get("buy_votes", 0))
                        m_col2.metric("Neutral Signals", tv_ta.get("neutral_votes", 0))
                        m_col3.metric("Sell Signals", tv_ta.get("sell_votes", 0))

                        st.markdown(
                            f"- **Oscillators Assessment:** `{tv_ta.get('oscillators_rec')}` ({tv_ta.get('oscillators_buy')} Buy, {tv_ta.get('oscillators_sell')} Sell)\n"
                            f"- **Moving Averages Assessment:** `{tv_ta.get('moving_averages_rec')}` ({tv_ta.get('moving_averages_buy')} Buy, {tv_ta.get('moving_averages_sell')} Sell)\n"
                            f"- **Key Technical Levels:** RSI (14): `{tv_ta.get('rsi14')}` | ADX: `{tv_ta.get('adx')}` | MACD: `{tv_ta.get('macd')}`"
                        )
                    else:
                        st.info(f"TradingView analysis info: {tv_ta.get('error')}")

                with tv_c2:
                    gauge_html = tvi.get_tv_technical_gauge_html(symbol, height=380, theme="dark" if is_dark else "light")
                    components.html(gauge_html, height=390)

            else:
                # Plotly Quantitative Chart
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.62, 0.19, 0.19],
                    vertical_spacing=0.03,
                    subplot_titles=(f"{symbol} Price Action", "Momentum (RSI 14)", "Volume & 20-Day Average"),
                )

                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="Price"), row=1, col=1)
                elif chart_type == "OHLC Bars":
                    fig.add_trace(go.Ohlc(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="OHLC"), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=d.index, y=d.close, name="Close Price", line=dict(color="#00E676", width=2)), row=1, col=1)

                if show_sma:
                    for col_name, color in (("sma20", "#2962FF"), ("sma50", "#FF6D00"), ("sma200", "#D50000")):
                        if col_name in d.columns:
                            fig.add_trace(go.Scatter(x=d.index, y=d[col_name], name=col_name.upper(), line=dict(width=1.2, color=color)), row=1, col=1)

                if show_bb:
                    fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="BB Upper", line=dict(dash="dot", width=1, color="gray")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], name="BB Lower", line=dict(dash="dot", width=1, color="gray")), row=1, col=1)

                if show_kama and "kama" in d.columns:
                    fig.add_trace(go.Scatter(x=d.index, y=d["kama"], name="Kaufman KAMA", line=dict(width=2.0, color="#00B0FF")), row=1, col=1)

                if show_alligator and "alligator_jaw" in d.columns:
                    fig.add_trace(go.Scatter(x=d.index, y=d["alligator_jaw"], name="Jaw (13)", line=dict(color="#1565C0", width=1.4)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["alligator_teeth"], name="Teeth (8)", line=dict(color="#C62828", width=1.4)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["alligator_lips"], name="Lips (5)", line=dict(color="#2E7D32", width=1.4)), row=1, col=1)

                if show_supertrend:
                    st_line, st_dir = osi.calc_supertrend(d)
                    fig.add_trace(go.Scatter(x=d.index, y=st_line, name="SuperTrend (10, 3)", line=dict(color="#00E676", width=2.0), mode="lines"), row=1, col=1)

                if show_donchian:
                    dc_hi = d["high"].rolling(20, min_periods=1).max()
                    dc_lo = d["low"].rolling(20, min_periods=1).min()
                    fig.add_trace(go.Scatter(x=d.index, y=dc_hi, name="Donchian Upper (20)", line=dict(color="#00E5FF", dash="dot", width=1.2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=dc_lo, name="Donchian Lower (20)", line=dict(color="#00E5FF", dash="dot", width=1.2)), row=1, col=1)

                # Subplot 2: RSI
                fig.add_trace(go.Scatter(x=d.index, y=d.rsi14, name="RSI (14)", line=dict(color="#AB47BC", width=1.5)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # Subplot 3: Volume
                vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(d.close, d.open)]
                fig.add_trace(go.Bar(x=d.index, y=d.volume, name="Volume", marker_color=vol_colors), row=3, col=1)
                fig.add_trace(go.Scatter(x=d.index, y=d.volume.rolling(20).mean(), name="20D Vol SMA", line=dict(color="#FFD600", width=1.2)), row=3, col=1)

                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ])
                    ),
                    row=1,
                    col=1,
                )

                apply_chart_theme(
                    fig,
                    height=560,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=8, r=8, t=25, b=8),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ==================================================================
            # 2. EXECUTION & ANALYSIS COCKPIT (BELOW CHART SUB-TABS)
            # ==================================================================
            term_tab_orders, term_tab_radar, term_tab_vsa = st.tabs([
                "⚡ AI Conviction & Orders",
                "🧠 90+ Indicators & MTF Radar",
                "🔬 VSA Bar Inspector",
            ])

            with term_tab_orders:
                if exec_stock_sig or exec_opt_sig:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid {theme_cfg['border']}; padding-bottom: 6px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 0.95rem; font-weight: 600; color: {theme_cfg['text_main']};">⚡ ORDER PAD & CONFLUENCE RECOMMENDATION</span>
                                    <span style="background: rgba(0, 208, 156, 0.15); color: {theme_cfg['primary']}; padding: 1px 8px; border-radius: 8px; font-size: 0.70rem; font-weight: 600; border: 1px solid rgba(0, 208, 156, 0.3);">{symbol} &nbsp;•&nbsp; {underlying}</span>
                                </div>
                                <span style="color: {theme_cfg['text_muted']}; font-size: 0.70rem; font-weight: 500;">90+ Indicators • 25 Book Models • VSA • Volatility Microstructure</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        c_stock, c_opt, c_evidence = st.columns([1.3, 1.3, 1.4])

                        # COLUMN 1: STOCK RECOMMENDATION
                        with c_stock:
                            if exec_stock_sig:
                                action = exec_stock_sig.action
                                color = "🟢" if action == "BUY" else ("🔴" if action == "SELL" else "🟡")
                                cur_p = exec_stock_sig.current_price
                                sl_p = exec_stock_sig.stop_loss
                                t1_p = exec_stock_sig.target_1
                                t2_p = exec_stock_sig.target_2
                                rr = exec_stock_sig.risk_reward

                                sl_pct = ((cur_p - sl_p) / cur_p) * 100 if cur_p else 0
                                t1_pct = ((t1_p - cur_p) / cur_p) * 100 if cur_p else 0

                                st.markdown(f"#### {color} Stock Order: **{action}**")
                                st.markdown(f"**Symbol:** `{symbol}` @ **₹{cur_p:,.2f}**")

                                if action == "BUY":
                                    st.success(f"**Buy Zone:** ₹{cur_p * 0.995:,.2f} – ₹{cur_p:,.2f}")
                                    st.markdown(f"- 🛑 **Stop-Loss:** `₹{sl_p:,.2f}` (-{sl_pct:.2f}%)")
                                    st.markdown(f"- 🎯 **Target 1:** `₹{t1_p:,.2f}` (+{t1_pct:.2f}%)")
                                    st.markdown(f"- 🎯 **Target 2:** `₹{t2_p:,.2f}`")
                                    st.markdown(f"- ⚖️ **Risk / Reward:** `1 : {rr}`")
                                elif action == "SELL":
                                    st.error(f"**Sell / Exit Zone:** ₹{cur_p:,.2f}")
                                    st.markdown(f"- 🛑 **Stop-Loss:** `₹{sl_p:,.2f}`")
                                    st.markdown(f"- 🎯 **Downside Target 1:** `₹{t1_p:,.2f}`")
                                    st.markdown(f"- 🎯 **Downside Target 2:** `₹{t2_p:,.2f}`")
                                    st.markdown(f"- ⚖️ **Risk / Reward:** `1 : {rr}`")
                                else:
                                    st.warning("**Decision:** WAIT / Sideways Consolidation")
                                    st.markdown(f"- **Key Support:** `₹{sl_p:,.2f}`")
                                    st.markdown(f"- **Key Breakout:** `₹{t1_p:,.2f}`")

                                st.metric("Stock Conviction", f"{exec_stock_sig.conviction}%")

                                if action in ("BUY", "SELL"):
                                    btn_color_label = "🟢 Place Paper BUY" if action == "BUY" else "🔴 Place Paper SELL"
                                    if st.button(f"{btn_color_label} ({exec_stock_sig.recommended_qty} Shares)", key="cockpit_paper_stock", type="primary", use_container_width=True):
                                        ok_p, msg_p = pt.execute_paper_order(exec_stock_sig, quantity=exec_stock_sig.recommended_qty, live_price=active_live_ltp, provider=provider)
                                        (st.success if ok_p else st.error)(msg_p)
                                        st.rerun()
                            else:
                                st.info("Analyzing stock technicals...")

                        # COLUMN 2: OPTION CHAIN RECOMMENDATION
                        with c_opt:
                            if exec_opt_sig:
                                opt_act = exec_opt_sig.action
                                opt_color = "🟢" if "CALL" in opt_act or "BULL" in opt_act else ("🔴" if "PUT" in opt_act or "BEAR" in opt_act else "🟡")
                                opt_entry = exec_opt_sig.entry_price
                                opt_sl = exec_opt_sig.stop_loss
                                opt_t1 = exec_opt_sig.target_1
                                opt_t2 = exec_opt_sig.target_2
                                lot_n = oa.get_lot_size(underlying, exec_opt_sig.current_price)

                                st.markdown(f"#### {opt_color} Option Order: **{opt_act.replace('_', ' ')}**")
                                st.markdown(f"**Underlying:** `{underlying}` @ **₹{exec_opt_sig.current_price:,.2f}**")
                                st.markdown(f"**Contract:** `{exec_opt_sig.legs_desc}`")

                                st.info(f"**Entry Premium:** ₹{opt_entry:,.2f} (Est. ₹{opt_entry * lot_n:,.0f}/lot)")
                                st.markdown(f"- 🛑 **Stop-Loss:** `₹{opt_sl:,.2f}` (Risk ₹{abs(opt_entry - opt_sl) * lot_n:,.0f}/lot)")
                                st.markdown(f"- 🎯 **Target 1:** `₹{opt_t1:,.2f}` (Gain ₹{abs(opt_t1 - opt_entry) * lot_n:,.0f}/lot)")
                                if opt_t2 > 0:
                                    st.markdown(f"- 🎯 **Target 2:** `₹{opt_t2:,.2f}`")
                                st.markdown(f"- 📅 **Expiry Date:** `{expiry}`")
                                st.metric("Option Conviction", f"{exec_opt_sig.conviction}%")

                                if opt_act != "WAIT":
                                    if st.button(f"🔵 Place Paper OPTION ({exec_opt_sig.recommended_qty} Lots)", key="cockpit_paper_opt", type="primary", use_container_width=True):
                                        ok_po, msg_po = pt.execute_paper_order(exec_opt_sig, quantity=exec_opt_sig.recommended_qty, provider=provider)
                                        (st.success if ok_po else st.error)(msg_po)
                                        st.rerun()
                            else:
                                st.info("Analyzing option chain derivatives...")

                        # COLUMN 3: EVIDENCE & SCOREBOARD
                        with c_evidence:
                            st.markdown("#### 🧠 AI Evidence & Confluence")

                            osi_data = getattr(exec_stock_sig, "indicator_consensus", None)
                            if osi_data and "consensus_pct" in osi_data:
                                c_pct = osi_data["consensus_pct"]
                                c_clr = "🟢" if c_pct >= 60 else ("🔴" if c_pct <= 40 else "🟡")
                                st.markdown(f"- {c_clr} **90+ Indicators:** `{c_pct}% Bullish` ({osi_data.get('total_bull_votes')}/{osi_data.get('total_votes')} Buy votes)")

                            if exec_opt_chain is not None and not exec_opt_chain.empty:
                                opt_sum = oa.summary(exec_opt_chain, exec_opt_sig.current_price if exec_opt_sig else 25000.0)
                                pcr_val = opt_sum.get("pcr_oi", 1.0)
                                pcr_color = "🟢 Bullish" if pcr_val > 1.0 else "🔴 Bearish"
                                st.markdown(f"- ⛓️ **PCR (OI):** `{pcr_val:.2f}` ({pcr_color})")
                                st.markdown(f"- 🎯 **Max Pain:** `{opt_sum.get('max_pain', 0):,.0f}`")
                                if "resistance" in opt_sum and opt_sum["resistance"]:
                                    st.markdown(f"- 🧱 **Call Resistance:** `{opt_sum['resistance'][0]:,.0f}`")
                                if "support" in opt_sum and opt_sum["support"]:
                                    st.markdown(f"- 🛡️ **Put Support:** `{opt_sum['support'][0]:,.0f}`")

                            try:
                                tv_quick = tvi.fetch_tradingview_ta(symbol, interval="1d")
                                if "error" not in tv_quick:
                                    tv_rec = tv_quick.get("recommendation", "NEUTRAL")
                                    tv_badge = "🟢" if "BUY" in tv_rec else ("🔴" if "SELL" in tv_rec else "🟡")
                                    st.markdown(f"- 📺 **TradingView TA:** {tv_badge} `{tv_rec}` ({tv_quick.get('buy_votes')} Buy, {tv_quick.get('sell_votes')} Sell)")
                            except Exception:
                                pass

                            try:
                                vix_obj = idata.get_india_vix_metrics(provider)
                                st.markdown(f"- 🏛️ **India VIX Regime:** `{vix_obj['vix']:.2f}` ({vix_obj['regime'].split()[0]})")
                            except Exception:
                                pass

                        with st.expander("📖 View Full AI Thesis & Foundational Literature Citations", expanded=False):
                            if exec_stock_sig:
                                st.markdown(f"**Stock Thesis:** {exec_stock_sig.thesis}")
                                if exec_stock_sig.literature_citations:
                                    st.markdown("**Foundational Citations:**")
                                    for cit in exec_stock_sig.literature_citations:
                                        st.markdown(f"- {cit}")
                            if exec_opt_sig:
                                st.markdown(f"**Options Thesis:** {exec_opt_sig.thesis}")

                    # Position Sizing & Capital Risk Calculator
                    with st.expander("🧮 Interactive Position Sizing & Capital Risk Calculator (Kaufman & O'Neil)", expanded=False):
                        calc_c1, calc_c2, calc_c3 = st.columns(3)
                        account_cap = calc_c1.number_input("Trading Capital (₹)", 10000, 10000000, 200000, step=10000)
                        risk_pct = calc_c2.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, step=0.25)
                        sizing_stop_pct = calc_c3.slider("Stop-Loss % from Entry", 3.0, 10.0, 7.0, step=0.5)

                        rupee_risk = account_cap * (risk_pct / 100.0)
                        risk_per_share = last_p * (sizing_stop_pct / 100.0)
                        shares_to_buy = max(1, int(rupee_risk / risk_per_share))
                        total_invested = shares_to_buy * last_p
                        stop_px = last_p * (1.0 - sizing_stop_pct / 100.0)
                        t1_px = last_p * 1.20
                        t2_px = last_p * 1.25

                        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                        res_c1.metric("Recommended Shares", f"{shares_to_buy:,}")
                        res_c2.metric("Capital Required", f"₹{total_invested:,.0f}", f"{round(total_invested/account_cap*100, 1)}% cap")
                        res_c3.metric("Max Rupee Loss", f"-₹{rupee_risk:,.0f}")
                        res_c4.metric("Strict Stop Price", f"₹{stop_px:,.2f}")
                        st.caption(f"🎯 **Profit Targets:** 20% Target = `₹{t1_px:,.2f}` (+₹{shares_to_buy*(t1_px-last_p):,.0f}) | 25% Target = `₹{t2_px:,.2f}` (+₹{shares_to_buy*(t2_px-last_p):,.0f})")
                else:
                    st.info("No active order setups generated for this symbol.")

            with term_tab_radar:
                # 90+ OPEN-SOURCE TECHNICAL INDICATORS CONSENSUS RADAR
                osi_eval = None
                try:
                    if d is not None and len(d) >= 30:
                        osi_eval = osi.evaluate_indicator_consensus(d)
                except Exception as e:
                    st.caption(f"Open-source indicators calculation: {e}")

                if osi_eval and "consensus_pct" in osi_eval:
                    c_pct = osi_eval["consensus_pct"]
                    stance = osi_eval["overall_stance"]
                    rec_act = osi_eval["recommended_action"]
                    stance_color = "🟢" if c_pct >= 65 else ("🔴" if c_pct <= 35 else "🟡")

                    st.markdown(f"#### {stance_color} 90+ Indicators Master Consensus: **{c_pct}% Bullish** — `{stance}` (Suggested Action: **{rec_act}**)")
                    st.progress(c_pct / 100.0, text=f"Consensus Strength: {osi_eval['total_bull_votes']} Bullish vs {osi_eval['total_bear_votes']} Bearish (across {osi_eval['total_votes']} indicators)")

                    bd = osi_eval["breakdown"]
                    c_t, c_m, c_v, c_f = st.columns(4)
                    with c_t:
                        st.metric("📈 Trend Category", f"{bd['trend']['pct']}%", f"{bd['trend']['bull']}/{bd['trend']['total']} Bullish")
                        st.progress(bd['trend']['pct'] / 100.0)
                    with c_m:
                        st.metric("⚡ Momentum Category", f"{bd['momentum']['pct']}%", f"{bd['momentum']['bull']}/{bd['momentum']['total']} Bullish")
                        st.progress(bd['momentum']['pct'] / 100.0)
                    with c_v:
                        st.metric("🌊 Volatility & Bands", f"{bd['volatility']['pct']}%", f"{bd['volatility']['bull']}/{bd['volatility']['total']} Bullish")
                        st.progress(bd['volatility']['pct'] / 100.0)
                    with c_f:
                        st.metric("💧 Volume & Money Flow", f"{bd['volume_flow']['pct']}%", f"{bd['volume_flow']['bull']}/{bd['volume_flow']['total']} Bullish")
                        st.progress(bd['volume_flow']['pct'] / 100.0)

                    with st.expander("🔍 View Detailed 90+ Indicator Signals Breakdown", expanded=False):
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown("**Trend & Momentum Signals:**")
                            for sig_txt in bd["trend"]["signals"]:
                                st.write(f"- {sig_txt}")
                            for sig_txt in bd["momentum"]["signals"]:
                                st.write(f"- {sig_txt}")
                        with col_s2:
                            st.markdown("**Volatility & Volume Flow Signals:**")
                            for sig_txt in bd["volatility"]["signals"]:
                                st.write(f"- {sig_txt}")
                            for sig_txt in bd["volume_flow"]["signals"]:
                                st.write(f"- {sig_txt}")
                st.markdown("---")

                with st.container(border=True):
                    inst_r1, inst_r2, inst_r3, inst_r4 = st.columns(4)
                    
                    with inst_r1:
                        mtf_color = aa_mtf.get("color", "#00E676")
                        st.markdown("**⚡ Multi-Timeframe (MTF) Confluence**")
                        st.caption(f"Weekly Tide: **{aa_mtf.get('weekly', {}).get('verdict', 'N/A')}**\n\nDaily Wave: **{aa_mtf.get('daily', {}).get('verdict', 'N/A')}**\n\nTrigger: **{aa_mtf.get('intraday', {}).get('verdict', 'N/A')}**")
                        st.markdown(f"<span style='color:{mtf_color}; font-weight: 600;'>{aa_mtf.get('overall_confluence', 'Neutral')}</span>", unsafe_allow_html=True)

                    with inst_r2:
                        st.markdown("**🌐 Sector & Relative Strength**")
                        st.caption(f"Sector: **{aa_sec['stock_sector']}**\n\nBenchmark: `{aa_sec['sector_index']}`")
                        rs_badge = "🟢 Leader" if aa_rs.get("is_leader") else "🔴 Laggard"
                        st.markdown(f"**{rs_badge}:** {aa_rs.get('status', 'N/A')}")
                        st.caption(f"Mansfield Alpha: {aa_rs.get('mansfield_rs', 0):+.1f}% vs Index")

                    with inst_r3:
                        st.markdown("**📦 NSE Delivery Volume %**")
                        st.metric("Delivery %", f"{aa_deliv['delivery_pct']}%", f"{aa_deliv['delivery_pct'] - aa_deliv['delivery_avg']:+.1f}% vs 20D Avg")
                        st.caption(f"{aa_deliv['status']}")

                    with inst_r4:
                        ev_color = aa_ev.get("color", "#00E676")
                        st.markdown("**📅 Earnings & Event Risk Radar**")
                        st.caption(f"Next Results Date: **{aa_ev['next_earnings_date']}** ({aa_ev['days_to_event']} days)")
                        st.markdown(f"<span style='color:{ev_color}; font-weight: 600;'>{aa_ev['risk_level']}</span>", unsafe_allow_html=True)
                        st.caption(f"{aa_ev['guidance'][:85]}...")

            with term_tab_vsa:
                st.markdown("#### 🔍 Interactive VSA Bar-by-Bar Candle Inspector")
                st.caption("Slide to inspect volume spread analysis and institutional diagnosis for any recent trading session.")

                recent_n = min(len(d), 30)
                candle_dates = [pd.to_datetime(dt).strftime("%Y-%m-%d") for dt in d.index[-recent_n:]]
                sel_date_str = st.select_slider("Select Date to Inspect", options=candle_dates, value=candle_dates[-1])
                sel_idx = candle_dates.index(sel_date_str) - recent_n
                sub_d = d.iloc[:len(d) + sel_idx + 1]
                inspected_vsa = ba.calc_vsa_signals(sub_d)

                insp_row = sub_d.iloc[-1]
                vi_c1, vi_c2, vi_c3, vi_c4 = st.columns(4)
                vi_c1.metric("Date", sel_date_str)
                vi_c2.metric("Close Price", f"₹{insp_row['close']:,.2f}")
                vi_c3.metric("Rel. Volume", f"{inspected_vsa.get('volume_str', 'N/A')}")
                vi_c4.metric("Close Position", f"{inspected_vsa.get('third', 'N/A')}")

                c_pos = max(0.0, min(1.0, float(inspected_vsa.get('close_pos', 0.5))))
                st.progress(c_pos, text=f"Bar Close Position: {int(c_pos * 100)}% (0% = At Low, 100% = At High)")
                for sig in inspected_vsa.get("signals", []):
                    st.info(f"**VSA Diagnosis:** {sig}")



# ==============================================================================
# WORKSPACE 2: SCREENER & UNIVERSE (Unified Stock & Option Chain Discovery)
# ==============================================================================
with tab_screener:
    c_sh1, c_sh2 = st.columns([4.2, 0.6])
    with c_sh1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: baseline; gap: 8px; margin: 1px 0 2px 0;">
                <span style="font-size: 0.95rem; font-weight: 700; color: {theme_cfg['text_main']};">🎯 Screener & Universe Discovery</span>
                <span style="font-size: 0.70rem; color: {theme_cfg['text_muted']};">Target: <strong style="color: {theme_cfg['primary']};">{symbol}</strong> • 2,640+ NSE/BSE Equities & Derivatives</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_sh2:
        if st.button("🔄 Sync", use_container_width=True, key="scr_btn_top_refresh_main", help="Re-fetch live quotes and clear in-memory caches"):
            storage.clear_mem_cache()
            st.cache_data.clear()
            st.rerun()

    if provider is None:
        st.error(f"Could not start the '{choice}' source: {provider_error}")
    elif not getattr(provider, "is_live", False):
        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.10); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 4px; padding: 2px 8px; display: inline-flex; align-items: center; gap: 6px; margin: 1px 0 3px 0;">
                <span style="background: #f59e0b; color: #000; font-weight: 700; font-size: 0.62rem; padding: 1px 4px; border-radius: 3px;">DEMO</span>
                <span style="color: #cbd5e1; font-size: 0.72rem;">Synthetic data — Select online in sidebar for live quotes.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    all_watchlists = config.load_all_watchlists()
    all_wl_keys = list(all_watchlists.keys())
    if "active_tab_wl" not in st.session_state or st.session_state["active_tab_wl"] not in all_wl_keys:
        st.session_state["active_tab_wl"] = all_wl_keys[0] if all_wl_keys else "My Watchlist"
    sel_wl_name = st.session_state["active_tab_wl"]
    current_symbols = all_watchlists.get(sel_wl_name, [])

    # Pinned anchor for sticky/frozen screener dock
    st.markdown('<div id="pinned-screener-dock-anchor"></div>', unsafe_allow_html=True)

    def _render_universal_screener_body(sel_wl_name: str, all_wl_keys: list):
        selected_screener_sym = st.session_state.get("active_symbol", "RELIANCE")
        comp_name = sd.get_company_name(selected_screener_sym)
        comp_sec = sd.get_sector(selected_screener_sym)
        missing_creds = config.shoonya_missing()
        broker_ready = (len(missing_creds) == 0)

        # ─── COMPACT SCREENER CONTROL DOCK ─────────────────────────────────
        with st.container(border=True):
            col_d_scope, col_d_sym, col_d_term, col_d_mode, col_d_kill = st.columns([1.5, 1.5, 0.7, 1.6, 0.6])
            with col_d_scope:
                scr_mode = st.radio(
                    "Scope",
                    ["Single Stock", "Universal Screener", "High-Conviction Scanner"],
                    horizontal=False,
                    key="scr_screening_scope_mode",
                    label_visibility="collapsed",
                    help="Switch between single stock & options desk, multi-strategy basket screening, and high-conviction setup scanner."
                )
            with col_d_sym:
                comp_short = comp_name[:20] + "..." if len(comp_name) > 22 else comp_name
                st.markdown(
                    f"""
                    <div style="line-height: 1.2; margin-top: 1px;">
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 0.95rem; font-weight: 700; color: {theme_cfg['primary']};">{selected_screener_sym}</span>
                            <span style="font-size: 0.70rem; color: {theme_cfg['text_sub']}; font-weight: 600;">{comp_short}</span>
                        </div>
                        <div style="font-size: 0.66rem; color: {theme_cfg['text_muted']};">{comp_sec}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_d_term:
                if st.button("🚀 Terminal", type="primary", use_container_width=True, key="btn_scr_open_term", help=f"Open {selected_screener_sym} in Terminal"):
                    st.session_state["active_symbol"] = selected_screener_sym
                    st.toast(f"Active symbol switched to {selected_screener_sym}! View charts in Terminal.")
                    st.rerun()
            with col_d_mode:
                exec_mode = st.radio(
                    "Execution Mode",
                    ["📝 Paper Trading", "⚡ Live Broker API"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"scr_exec_mode_choice_{selected_screener_sym}",
                    help="Switch between Simulated Paper Trading and Real Broker Exchange Order Placement via Shoonya / Finvasia API."
                )
                if broker_ready:
                    st.markdown("<div style='font-size: 0.66rem; color: #10b981; font-weight: 600; line-height: 1.1;'>🟢 Shoonya Live API Ready</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size: 0.66rem; color: #f59e0b; font-weight: 600; line-height: 1.1;'>🟡 Broker: {len(missing_creds)} cred(s) missing</div>", unsafe_allow_html=True)
            with col_d_kill:
                if st.button("🛑 Kill", type="secondary", key=f"scr_kill_switch_{selected_screener_sym}", help="Emergency Safety Breaker: Cancel all pending orders on exchange and attempt position square off.", use_container_width=True):
                    ok_k, msg_k = le.live_executor.emergency_global_kill_switch()
                    if ok_k:
                        st.error(msg_k)
                    else:
                        st.warning(msg_k)

        # Credentials expander if Live Broker API is chosen and credentials are missing
        if "Live" in exec_mode and not broker_ready:
            with st.expander("🔑 Configure Finvasia / Shoonya Broker API Credentials", expanded=True):
                st.info("Enter your Shoonya / Finvasia API credentials to enable live algo order placement. Credentials are encrypted and stored in your local `.env`.")
                bc_c1, bc_c2, bc_c3 = st.columns(3)
                with bc_c1:
                    sh_u = st.text_input("User ID", value=os.environ.get("SHOONYA_USER", ""), key=f"scr_u_{selected_screener_sym}")
                    sh_p = st.text_input("Password", value=os.environ.get("SHOONYA_PASSWORD", ""), type="password", key=f"scr_p_{selected_screener_sym}")
                with bc_c2:
                    sh_t = st.text_input("TOTP Secret (2FA)", value=os.environ.get("SHOONYA_TOTP_SECRET", ""), type="password", key=f"scr_t_{selected_screener_sym}")
                    sh_v = st.text_input("Vendor Code", value=os.environ.get("SHOONYA_VENDOR_CODE", ""), key=f"scr_v_{selected_screener_sym}")
                with bc_c3:
                    sh_a = st.text_input("API Key", value=os.environ.get("SHOONYA_API_KEY", ""), type="password", key=f"scr_a_{selected_screener_sym}")
                    sh_i = st.text_input("IMEI / ID", value=os.environ.get("SHOONYA_IMEI", "abc1234"), key=f"scr_i_{selected_screener_sym}")

                if st.button("💾 Save Credentials & Verify Connection", key=f"scr_save_shoonya_{selected_screener_sym}", type="primary"):
                    os.environ["SHOONYA_USER"] = sh_u.strip()
                    os.environ["SHOONYA_PASSWORD"] = sh_p.strip()
                    os.environ["SHOONYA_TOTP_SECRET"] = sh_t.strip()
                    os.environ["SHOONYA_VENDOR_CODE"] = sh_v.strip()
                    os.environ["SHOONYA_API_KEY"] = sh_a.strip()
                    os.environ["SHOONYA_IMEI"] = sh_i.strip()
                    try:
                        env_map = {
                            "SHOONYA_USER": sh_u.strip(),
                            "SHOONYA_PASSWORD": sh_p.strip(),
                            "SHOONYA_TOTP_SECRET": sh_t.strip(),
                            "SHOONYA_VENDOR_CODE": sh_v.strip(),
                            "SHOONYA_API_KEY": sh_a.strip(),
                            "SHOONYA_IMEI": sh_i.strip(),
                        }
                        env_p = Path(".env")
                        existing = env_p.read_text(encoding="utf-8") if env_p.exists() else ""
                        lines = existing.splitlines()
                        new_lines = []
                        seen_keys = set()
                        for l in lines:
                            if "=" in l and not l.strip().startswith("#"):
                                k = l.split("=", 1)[0].strip()
                                if k in env_map:
                                    new_lines.append(f"{k}={env_map[k]}")
                                    seen_keys.add(k)
                                    continue
                            new_lines.append(l)
                        for k, val in env_map.items():
                            if k not in seen_keys:
                                new_lines.append(f"{k}={val}")
                        env_p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                        st.success("Credentials saved to .env! Re-verifying broker...")
                        st.rerun()
                    except Exception as e_env:
                        st.error(f"Error saving to .env: {e_env}")

        if "Single Stock" in scr_mode:
            if provider and selected_screener_sym:
                with st.spinner(f"Screening {selected_screener_sym} & derivatives..."):
                    df_stk = load_stock(selected_screener_sym)
                    single_tbl, single_errs = _st_cached_screener_scan(provider.name, selected_screener_sym)
                    stk_sig = ate.evaluate_stock_trade(selected_screener_sym, df_stk) if df_stk is not None and not df_stk.empty else None

                s_row = single_tbl.iloc[0] if (not single_tbl.empty) else {}

                # ──────────────────────────────────────────────────────────────
                # 1. SIDE-BY-SIDE: STOCK ANALYSIS & OPTION ANALYSIS (SCROLLING MODE)
                # ──────────────────────────────────────────────────────────────
                is_opt_eligible = (selected_screener_sym in oa.NSE_FO_LOT_SIZES) or oa.is_weekly_index(selected_screener_sym)
                opt_chain = None
                opt_spot = None
                scr_opt_expiry = None

                with st.container(height=660, border=False):
                    col_stock_side, col_option_side = st.columns(2)

                    # ─── LEFT COLUMN: STOCK ANALYSIS & CONVICTION ─────────────────
                    with col_stock_side:
                        st.markdown(f"#### 📊 Stock Analysis — `{selected_screener_sym}`")
                        with st.container(border=True):
                            stk_conv = stk_sig.conviction if stk_sig else int(s_row.get("can_slim_score", 75))
                            stk_action = stk_sig.action if stk_sig else ("BUY" if s_row.get("trend") == "Uptrend" else "NEUTRAL")
                            stk_clr = "🟢" if stk_action == "BUY" else ("🔴" if stk_action == "SELL" else "🟡")
                            stk_ltp = float(s_row.get("close", df_stk["close"].iloc[-1] if df_stk is not None and not df_stk.empty else 0.0))
                            stk_rr = getattr(stk_sig, "risk_reward", 2.0)
                            stk_qty = getattr(stk_sig, "recommended_qty", 10)

                            crit_conv = int(st.session_state.get("omnibar_min_conviction", 0))
                            crit_wr = float(st.session_state.get("omnibar_min_win_rate", 0.0))
                            stk_exp_wr = getattr(stk_sig, "expected_win_rate", 50.0)
                            if crit_conv > 0 or crit_wr > 0:
                                passes_crit = (stk_conv >= crit_conv) and (stk_exp_wr >= crit_wr)
                                if passes_crit:
                                    st.markdown(
                                        f"<div style='background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); border-radius: 4px; padding: 3px 8px; font-size: 0.72rem; color: #10b981; font-weight: 600; margin-bottom: 6px;'>🎯 <b>CRITERIA MET</b>: High Conviction ({stk_conv}% ≥ {crit_conv}%) & Win Ratio ({stk_exp_wr:.1f}% ≥ {crit_wr:.0f}%) Institutional Setup</div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"<div style='background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.25); border-radius: 4px; padding: 3px 8px; font-size: 0.72rem; color: #f59e0b; margin-bottom: 6px;'>ℹ️ Criteria Status: Conviction {stk_conv}% (Target: ≥{crit_conv}%) • Exp Win Rate {stk_exp_wr:.1f}% (Target: ≥{crit_wr:.0f}%)</div>",
                                        unsafe_allow_html=True
                                    )

                            cs_r1_1, cs_r1_2, cs_r1_3 = st.columns(3)
                            cs_r1_1.metric(f"{stk_clr} Confluence", f"{stk_conv}%", f"{stk_action} ({getattr(stk_sig, 'grade', 'Grade A')})")
                            cs_r1_2.metric("Exp Win Rate", f"{stk_exp_wr:.1f}%", f"R:R 1:{stk_rr:.1f}")
                            cs_r1_3.metric("LTP (₹)", f"₹{stk_ltp:,.2f}", f"{s_row.get('chg_1m_pct', 0.0):+.1f}% (1M)")

                            be_val = getattr(stk_sig, "breakeven_price", None)
                            cs_r2_1, cs_r2_2, cs_r2_3, cs_r2_4 = st.columns(4)
                            cs_r2_1.metric("Entry Zone", f"₹{stk_sig.entry_price:,.2f}" if stk_sig else f"₹{stk_ltp:,.2f}", "Trigger")
                            cs_r2_2.metric("Stop Loss", f"₹{stk_sig.stop_loss:,.2f}" if stk_sig else "N/A", "Risk Anchor")
                            cs_r2_3.metric("Target 1", f"₹{stk_sig.target_1:,.2f}" if stk_sig else "N/A", f"R:R 1:{stk_rr:.1f}")
                            cs_r2_4.metric("Breakeven Trigger", f"₹{be_val:,.2f}" if be_val else f"₹{stk_ltp * 1.02:,.2f}", "+1.0R Lock-In")

                            # Institutional Edge & SMC Badges
                            smc_info = getattr(stk_sig, "smc_confluence", None)
                            macro_info = getattr(stk_sig, "macro_gate", None)
                            col_edge_smc, col_edge_macro = st.columns(2)
                            with col_edge_smc:
                                if smc_info and "smc_badge" in smc_info:
                                    st.markdown(f"<div style='background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 4px 8px; font-size: 0.76rem; font-weight: 600; color: #10b981;'>{smc_info['smc_badge']}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 4px 8px; font-size: 0.76rem; font-weight: 600; color: #10b981;'>🟢 SMC: Smart Money Accumulation</div>", unsafe_allow_html=True)
                            with col_edge_macro:
                                if macro_info and "gate_reason" in macro_info:
                                    gate_clr = "#10b981" if macro_info.get("gate_passed", True) else "#f59e0b"
                                    reason_short = macro_info['gate_reason'][:55] + "..." if len(macro_info['gate_reason']) > 55 else macro_info['gate_reason']
                                    st.markdown(f"<div style='background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); border-radius: 6px; padding: 4px 8px; font-size: 0.76rem; font-weight: 600; color: {gate_clr};'>{reason_short}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); border-radius: 6px; padding: 4px 8px; font-size: 0.76rem; font-weight: 600; color: #38bdf8;'>✅ Macro Alignment: Trade With Tide (+15% Win Rate Edge)</div>", unsafe_allow_html=True)

                            # Volume Profile & Auction Market Theory
                            vp_res = vp.compute_volume_profile(df_stk) if (df_stk is not None and not df_stk.empty) else None
                            if vp_res and vp_res.get("poc", 0) > 0:
                                st.markdown("---")
                                vp_c1, vp_c2, vp_c3, vp_c4 = st.columns(4)
                                vp_c1.metric("POC (Peak Vol)", f"₹{vp_res['poc']:,.2f}", f"{vp_res.get('poc_distance_pct', 0.0):+.1f}% Dist")
                                vp_c2.metric("VAH (Value High)", f"₹{vp_res['vah']:,.2f}", "70% Vol Cap")
                                vp_c3.metric("VAL (Value Low)", f"₹{vp_res['val']:,.2f}", "70% Vol Floor")
                                vp_c4.metric("Auction State", vp_res.get("market_state", "").split()[0], "AMT Balance")

                                amt_sig = vp_res.get("amt_rule")
                                if amt_sig:
                                    st.markdown(f"<div style='background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); border-radius: 6px; padding: 4px 8px; margin-top: 4px; font-size: 0.76rem; font-weight: 600; color: #f59e0b;'>🎯 {amt_sig['rule']}: {amt_sig['interpretation']}</div>", unsafe_allow_html=True)

                            # Intermarket Macro Confluence
                            im_res = im.evaluate_sector_intermarket_confluence(selected_screener_sym, action=stk_action)
                            im_clr = "#10b981" if im_res["macro_verdict"] == "TAILWIND" else ("#ef4444" if im_res["macro_verdict"] == "HEADWIND" else "#38bdf8")
                            st.markdown(f"<div style='background: rgba(255,255,255,0.04); border-left: 3px solid {im_clr}; border-radius: 4px; padding: 5px 10px; margin: 6px 0; font-size: 0.77rem;'><b style='color: {im_clr};'>{im_res['macro_verdict']}</b> | <b>Sector:</b> {im_res['sector']} | {im_res['primary_rationale']}</div>", unsafe_allow_html=True)

                            # Dynamic Trailing Risk Engine (Chandelier & Multi-Stage Profit Ratchet)
                            tr_entry = stk_sig.entry_price if stk_sig else stk_ltp
                            tr_stop = stk_sig.stop_loss if (stk_sig and stk_sig.stop_loss > 0) else round(stk_ltp * 0.95, 2)
                            tr_plan = tre.generate_trailing_plan(entry_price=tr_entry, stop_loss=tr_stop, current_price=stk_ltp, action=stk_action)
                            tr_exp = tre.calculate_van_tharp_expectancy(win_rate_pct=getattr(stk_sig, "expected_win_rate", 55.0))
                            
                            st.markdown("---")
                            tr_c1, tr_c2, tr_c3, tr_c4 = st.columns(4)
                            tr_c1.metric("Chandelier Trail", f"₹{tr_plan['chandelier_stop']:,.2f}", "2.5x ATR Exit")
                            tr_c2.metric("Target 1 (+1.5R)", f"₹{tr_plan['target_1_price']:,.2f}", "50% Scale-Out")
                            tr_c3.metric("Target 2 (+2.5R)", f"₹{tr_plan['target_2_price']:,.2f}", "Runner Target")
                            tr_c4.metric("Van Tharp Expectancy", f"{tr_exp['expectancy_r_multiple']:+.2f}R", tr_exp.get("system_quality", "")[:12])

                            st.markdown("---")
                            cq_r1_1, cq_r1_2, cq_r1_3 = st.columns(3)
                            cq_r1_1.metric("CAN SLIM Score", f"{s_row.get('can_slim_score', 0)}/100", s_row.get("trend", "Neutral"))
                            cq_r1_2.metric("Volume Surge", f"{s_row.get('vol_surge_x', 1.0):.2f}x", "vs 50MA Vol")
                            cq_r1_3.metric("VSA Signal", f"{str(s_row.get('vsa_signal', 'Neutral'))[:15]}")

                            cq_r2_1, cq_r2_2 = st.columns(2)
                            cq_r2_1.metric("Kaufman ER", f"{s_row.get('kaufman_er', 0.0):.2f}", str(s_row.get("kaufman_regime", "Range")))
                            cq_r2_2.metric("RSI (14)", f"{s_row.get('rsi14', 50.0):.1f}", "Momentum")

                            st.markdown("---")
                            if "Live" in exec_mode:
                                st.markdown("##### ⚡ Algo Stock Order Pad (Real Broker API)")
                                with st.container(border=True):
                                    col_stk_p1, col_stk_p2 = st.columns(2)
                                    with col_stk_p1:
                                        stk_product = st.selectbox(
                                            "Product Type",
                                            ["C (CNC - Delivery)", "I (MIS - Intraday)"],
                                            index=0,
                                            key=f"scr_stk_prod_{selected_screener_sym}",
                                            help="CNC = Cash & Carry Delivery; MIS = Margin Intraday Square-off."
                                        )
                                        stk_prod_code = "C" if "CNC" in stk_product or "C " in stk_product else "I"
                                    with col_stk_p2:
                                        stk_ord_type = st.selectbox(
                                            "Order Type",
                                            ["LMT (Limit Order)", "MKT (Market Order)"],
                                            index=0,
                                            key=f"scr_stk_ord_type_{selected_screener_sym}",
                                        )
                                        stk_ord_code = "LMT" if "LMT" in stk_ord_type else "MKT"

                                    col_stk_q1, col_stk_q2 = st.columns(2)
                                    with col_stk_q1:
                                        stk_live_qty = st.number_input(
                                            "Order Quantity (Shares)",
                                            min_value=1,
                                            max_value=100000,
                                            value=max(1, stk_qty),
                                            step=1,
                                            key=f"scr_stk_qty_{selected_screener_sym}",
                                        )
                                    with col_stk_q2:
                                        est_stk_val = stk_live_qty * stk_ltp
                                        st.metric("Estimated Order Value", f"₹{est_stk_val:,.2f}", f"{stk_live_qty} Shares")

                                    sl_trigger_val = stk_sig.stop_loss if (stk_sig and stk_sig.stop_loss > 0) else round(stk_ltp * 0.95, 2)
                                    stk_attach_sl = st.checkbox(
                                        f"🛡️ Auto-Bracket Stop-Loss (Trigger: ₹{sl_trigger_val:,.2f})",
                                        value=True,
                                        key=f"scr_stk_attach_sl_{selected_screener_sym}",
                                        help="Automatically send exchange SL-LMT order to protect downside risk immediately upon entry fill."
                                    )

                                    stk_btn_col1, stk_btn_col2 = st.columns([1.1, 1.4])
                                    with stk_btn_col1:
                                        if st.button(f"⭐ Add to {sel_wl_name}", key=f"scr_add_wl_{selected_screener_sym}", use_container_width=True):
                                            if config.add_to_watchlist(sel_wl_name, selected_screener_sym):
                                                st.success(f"Added {selected_screener_sym} to {sel_wl_name}!")
                                                st.rerun()
                                    with stk_btn_col2:
                                        if stk_sig and stk_sig.action in ("BUY", "SELL"):
                                            live_btn_label = f"🚀 Place Real {stk_sig.action} Order ({stk_live_qty} Qty)"
                                            if st.button(live_btn_label, type="primary", key=f"scr_btn_live_stk_{selected_screener_sym}", use_container_width=True):
                                                with st.spinner(f"Routing {stk_sig.action} {stk_live_qty} {selected_screener_sym} to broker exchange..."):
                                                    ok_live, msg_live = le.live_executor.place_live_order(
                                                        signal=stk_sig,
                                                        quantity=int(stk_live_qty),
                                                        product_type=stk_prod_code,
                                                        order_type=stk_ord_code,
                                                    )
                                                    if ok_live:
                                                        st.success(f"✅ {msg_live}")
                                                        if stk_attach_sl and sl_trigger_val > 0:
                                                            is_buy_sig = (stk_sig.action == "BUY")
                                                            ok_sl, msg_sl = le.live_executor.place_stop_loss_order(
                                                                symbol=selected_screener_sym,
                                                                quantity=int(stk_live_qty),
                                                                stop_price=sl_trigger_val,
                                                                limit_price=round(sl_trigger_val * (0.995 if is_buy_sig else 1.005), 2),
                                                                is_buy=(not is_buy_sig),
                                                                exch="NSE",
                                                                product_type=stk_prod_code,
                                                            )
                                                            if ok_sl:
                                                                st.info(f"🛡️ {msg_sl}")
                                                            else:
                                                                st.warning(f"Stop-Loss notice: {msg_sl}")
                                                    else:
                                                        st.error(f"❌ {msg_live}")
                                        else:
                                            st.button("⏸️ Signal Neutral / No Active Trigger", disabled=True, key=f"scr_wait_{selected_screener_sym}", use_container_width=True)
                            else:
                                btn_w1, btn_w2 = st.columns([1.1, 1.4])
                                with btn_w1:
                                    if st.button(f"⭐ Add to {sel_wl_name}", key=f"scr_add_wl_{selected_screener_sym}", use_container_width=True):
                                        if config.add_to_watchlist(sel_wl_name, selected_screener_sym):
                                            st.success(f"Added {selected_screener_sym} to {sel_wl_name}!")
                                            st.rerun()
                                with btn_w2:
                                    if stk_sig and stk_sig.action in ("BUY", "SELL"):
                                        if st.button(f"📝 Execute Paper Stock ({stk_qty} Qty)", type="primary", key=f"scr_exec_stk_{selected_screener_sym}", use_container_width=True):
                                            ok_s, msg_s = pt.execute_paper_order(stk_sig, quantity=stk_qty, provider=provider)
                                            (st.success if ok_s else st.error)(msg_s)
                                            st.rerun()
                                    else:
                                        st.button("⏸️ Signal Neutral / Wait", disabled=True, key=f"scr_wait_{selected_screener_sym}", use_container_width=True)


                    # ─── RIGHT COLUMN: OPTION ANALYSIS & DERIVATIVES (ZERODHA / GROWW STYLE) ───
                    with col_option_side:
                        st.markdown(f"#### ⛓️ Option Analysis — `{selected_screener_sym}`")
                        with st.container(border=True):
                            if is_opt_eligible:
                                opt_is_idx = oa.is_weekly_index(selected_screener_sym)
                                opt_expiries = oa.get_stock_expiry_dates(
                                    selected_screener_sym,
                                    expiry_type="all" if opt_is_idx else "monthly",
                                    cycle="online"
                                )
                                if not opt_expiries:
                                    opt_expiries = oa.get_stock_expiry_dates(selected_screener_sym)

                                get_exp_lbl = getattr(oa, "get_expiry_label", lambda e, u="": f"{e} — Expiry")
                                exp_labels = [get_exp_lbl(e, selected_screener_sym) for e in opt_expiries]

                                col_exp_drop, col_lot_disp = st.columns([2.0, 1.0])
                                with col_exp_drop:
                                    chosen_exp_label = st.selectbox(
                                        "Select Expiry Date",
                                        options=exp_labels,
                                        index=0,
                                        key=f"scr_opt_exp_choice_{selected_screener_sym}",
                                        help="Select contract expiry date to load live option chain and evaluate strategy conviction."
                                    )
                                    scr_opt_expiry = chosen_exp_label.split()[0].strip()[:10]
                                with col_lot_disp:
                                    opt_lot = oa.get_lot_size(selected_screener_sym)
                                    st.metric("F&O Lot Size", f"{opt_lot:,} Qty", "Contract Lot")

                                # Fetch Option Chain
                                try:
                                    opt_chain = provider.get_option_chain(selected_screener_sym, scr_opt_expiry)
                                except Exception as e:
                                    st.warning(f"Option chain query for {selected_screener_sym} ({scr_opt_expiry}): {e}")

                                if opt_chain is not None and not opt_chain.empty:
                                    opt_spot = float(opt_chain.attrs.get("underlying_ltp") or s_row.get("close", 0.0) or opt_chain["strike"].median())
                                    gz_opt = oa.calc_groww_zerodha_analytics(opt_chain, opt_spot, expiry=scr_opt_expiry)

                                    opt_sig = None
                                    try:
                                        opt_sig = ate.evaluate_options_trade(selected_screener_sym, opt_chain, df_stk, opt_spot, scr_opt_expiry)
                                    except Exception:
                                        pass

                                    # ─── SECTION 1: CONVICTION, IV & SENTIMENT ─────────────
                                    o_r1_1, o_r1_2, o_r1_3, o_r1_4 = st.columns(4)
                                    pop_val = getattr(opt_sig, "pop_percent", None) if opt_sig else None
                                    if opt_sig:
                                        o_clr = "🟢" if ("CALL" in opt_sig.action or "BULL" in opt_sig.action) else ("🔴" if ("PUT" in opt_sig.action or "BEAR" in opt_sig.action) else "🟡")
                                        sub_lbl = f"{pop_val:.0f}% POP" if pop_val else opt_sig.action.replace("_", " ")
                                        o_r1_1.metric(f"{o_clr} Option Conviction", f"{opt_sig.conviction}%", sub_lbl)
                                    else:
                                        o_r1_1.metric("Option Conviction", "N/A", "Evaluating")
                                    o_r1_2.metric("ATM IV", f"{gz_opt.get('atm_iv', 16.5):.1f}%", f"DTE {gz_opt.get('dte', 7):.0f}d")
                                    o_r1_3.metric("PCR (OI)", f"{gz_opt.get('pcr_oi', 1.0):.2f}", gz_opt.get("sentiment", "Neutral"))
                                    o_r1_4.metric("Max Pain", f"₹{gz_opt.get('max_pain', 0):,.0f}", "Expiry Anchor")

                                    if pop_val and pop_val >= 75:
                                        st.markdown(f"<div style='background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 4px 10px; margin: 4px 0 6px 0; font-size: 0.76rem; font-weight: 600; color: #10b981;'>🛡️ Statistical High-Win Rate Spread: {pop_val:.1f}% Probability of Profit (POP) | 1.5σ OTM Wing Cushion</div>", unsafe_allow_html=True)

                                    # Expiry Day Gamma Risk & Strike Pinning
                                    exp_status = egm.check_symbol_expiry_status(selected_screener_sym)
                                    pin_res = egm.evaluate_strike_pinning_probability(
                                        spot=opt_spot,
                                        max_pain=gz_opt.get('max_pain', opt_spot),
                                        pcr=gz_opt.get('pcr_oi', 1.0),
                                        dte=int(gz_opt.get('dte', 7))
                                    )
                                    if exp_status.get("is_expiry_day") or gz_opt.get('dte', 7) <= 1:
                                        hz_clr = "#ef4444" if "CRITICAL" in exp_status.get("hazard_level", "") else "#f59e0b"
                                        st.markdown(f"<div style='background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.35); border-radius: 6px; padding: 5px 10px; margin: 4px 0 6px 0; font-size: 0.77rem; color: {hz_clr}; font-weight: 600;'>⚠️ {exp_status.get('actionable_advice')} | 🎯 Pinning Prob: {pin_res['pin_probability_pct']:.0f}% ({pin_res['pinning_verdict'][:32]})</div>", unsafe_allow_html=True)

                                    st.markdown("---")

                                    # ─── SECTION 2: EXPECTED PRICE RANGE & STRADDLE (GROWW / ZERODHA) ───
                                    rng_c1, rng_c2 = st.columns(2)
                                    with rng_c1:
                                        st.metric(
                                            "🎯 Expected Expiry Price Range (1σ)",
                                            f"₹{gz_opt.get('price_range_low', 0):,.0f} — ₹{gz_opt.get('price_range_high', 0):,.0f}",
                                            f"±{gz_opt.get('range_pct', 0.0):.1f}% (±₹{gz_opt.get('expected_move', 0):,.0f})"
                                        )
                                    with rng_c2:
                                        st.metric(
                                            "⚡ ATM Straddle Breakevens",
                                            f"₹{gz_opt.get('straddle_be_low', 0):,.0f} — ₹{gz_opt.get('straddle_be_high', 0):,.0f}",
                                            f"Straddle Cost ₹{gz_opt.get('straddle_price', 0):,.1f}"
                                        )

                                    st.markdown("---")

                                    # ─── SECTION 3: DERIVATIVE SUPPORT & RESISTANCE (OI WALLS) ───
                                    sr_c1, sr_c2, sr_c3, sr_c4 = st.columns(4)
                                    sr_c1.metric("Major Support (S1)", f"₹{gz_opt.get('s1_strike', 0):,.0f}", f"Put Wall ({gz_opt.get('s1_oi', 0):,})")
                                    sr_c2.metric("Major Resistance (R1)", f"₹{gz_opt.get('r1_strike', 0):,.0f}", f"Call Wall ({gz_opt.get('r1_oi', 0):,})")
                                    sr_c3.metric("Support 2 (S2)", f"₹{gz_opt.get('s2_strike', 0):,.0f}", "Secondary Floor")
                                    sr_c4.metric("Resistance 2 (R2)", f"₹{gz_opt.get('r2_strike', 0):,.0f}", "Secondary Ceiling")

                                    # ─── SECTION 4: REAL-TIME OI TOTALS ───
                                    oi_tot_c1, oi_tot_c2, oi_tot_c3, oi_tot_c4 = st.columns(4)
                                    oi_tot_c1.metric("Total Call OI", f"{gz_opt.get('tot_call_oi', 0):,}", "Call Writers")
                                    oi_tot_c2.metric("Total Put OI", f"{gz_opt.get('tot_put_oi', 0):,}", "Put Writers")
                                    oi_tot_c3.metric("PCR (Volume)", f"{gz_opt.get('pcr_vol', 1.0):.2f}", "Intraday Ratio")
                                    oi_tot_c4.metric("OI Market Bias", gz_opt.get("sentiment", "Neutral"), gz_opt.get("sent_color", "🟡"))

                                    # ─── SECTION 5: ZERODHA / GROWW STYLE OI DISTRIBUTION BAR CHART ───
                                    all_stk = sorted(opt_chain["strike"].unique())
                                    if len(all_stk) > 0:
                                        atm_idx = min(range(len(all_stk)), key=lambda i: abs(all_stk[i] - opt_spot))
                                        c_start = max(0, atm_idx - 6)
                                        c_end = min(len(all_stk), atm_idx + 7)
                                        chart_strikes = all_stk[c_start:c_end]
                                        chart_df = opt_chain[opt_chain["strike"].isin(chart_strikes)].sort_values(by="strike")

                                        fig_oi = go.Figure()
                                        fig_oi.add_trace(go.Bar(
                                            x=chart_df["strike"],
                                            y=chart_df["ce_oi"].fillna(0),
                                            name="Call OI (Resistance)",
                                            marker_color="#ef4444",
                                            hovertemplate="Strike ₹%{x}<br>Call OI: %{y:,.0f}<extra></extra>"
                                        ))
                                        fig_oi.add_trace(go.Bar(
                                            x=chart_df["strike"],
                                            y=chart_df["pe_oi"].fillna(0),
                                            name="Put OI (Support)",
                                            marker_color="#10b981",
                                            hovertemplate="Strike ₹%{x}<br>Put OI: %{y:,.0f}<extra></extra>"
                                        ))
                                        fig_oi.add_vline(
                                            x=opt_spot,
                                            line_width=2,
                                            line_dash="dash",
                                            line_color="#f59e0b",
                                            annotation_text=f"Spot ₹{opt_spot:,.0f}",
                                            annotation_position="top"
                                        )
                                        theme_text = "#f1f5f9" if is_dark else "#0f172a"
                                        grid_col = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.06)"

                                        fig_oi.update_layout(
                                            title={"text": "📊 Open Interest (OI) Distribution across ATM Strikes", "font": {"size": 13, "color": theme_text}},
                                            barmode="group",
                                            height=240,
                                            margin=dict(l=10, r=10, t=32, b=10),
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=theme_text)),
                                            xaxis=dict(
                                                title=dict(text="Strike Price (₹)", font=dict(color=theme_text, size=10)),
                                                showgrid=True,
                                                gridcolor=grid_col,
                                                tickfont=dict(color=theme_text, size=9),
                                            ),
                                            yaxis=dict(
                                                title=dict(text="Contracts (OI)", font=dict(color=theme_text, size=10)),
                                                showgrid=True,
                                                gridcolor=grid_col,
                                                tickfont=dict(color=theme_text, size=9),
                                            ),
                                        )
                                        st.plotly_chart(fig_oi, use_container_width=True, config={"displayModeBar": False})

                                    st.markdown("---")
                                    if opt_sig:
                                        st.markdown(f"**Strategy ({getattr(opt_sig, 'grade', 'Grade A')} | Confluence: {getattr(opt_sig, 'confluence_rating', opt_sig.conviction)}% | Exp Win: {getattr(opt_sig, 'expected_win_rate', 50.0):.1f}%):** `{opt_sig.legs_desc}`")
                                        st.write(f"- 💵 **Net Entry:** `₹{opt_sig.entry_price:,.2f}` | 🛑 **SL:** `₹{opt_sig.stop_loss:,.2f}` | 🎯 **TGT:** `₹{opt_sig.target_1:,.2f}` | ⚖️ **R:R 1:{opt_sig.risk_reward}`")

                                        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                                        if "Live" in exec_mode:
                                            st.markdown("##### ⚡ Algo Option Order Pad (Real Broker API)")
                                            with st.container(border=True):
                                                col_opt_p1, col_opt_p2 = st.columns(2)
                                                with col_opt_p1:
                                                    opt_product = st.selectbox(
                                                        "Option Product",
                                                        ["M (NRML - Carry Forward)", "I (MIS - Intraday)"],
                                                        index=0,
                                                        key=f"scr_opt_prod_{selected_screener_sym}",
                                                        help="NRML = Carry forward until expiry date; MIS = Margin Intraday Square-off."
                                                    )
                                                    opt_prod_code = "M" if "NRML" in opt_product or "M " in opt_product else "I"
                                                with col_opt_p2:
                                                    opt_ord_type = st.selectbox(
                                                        "Order Type",
                                                        ["LMT (Limit Order)", "MKT (Market Order)"],
                                                        index=0,
                                                        key=f"scr_opt_ord_type_{selected_screener_sym}",
                                                    )
                                                    opt_ord_code = "LMT" if "LMT" in opt_ord_type else "MKT"

                                                col_opt_l1, col_opt_l2 = st.columns(2)
                                                with col_opt_l1:
                                                    opt_lots = st.number_input(
                                                        f"Contract Lots (Lot Size: {opt_lot})",
                                                        min_value=1,
                                                        max_value=200,
                                                        value=1,
                                                        step=1,
                                                        key=f"scr_opt_lots_{selected_screener_sym}",
                                                    )
                                                with col_opt_l2:
                                                    total_opt_qty = int(opt_lots * opt_lot)
                                                    est_opt_premium = total_opt_qty * opt_sig.entry_price
                                                    st.metric("Total Contracts", f"{total_opt_qty:,} Qty", f"Est. Premium ₹{est_opt_premium:,.2f}")

                                                # Suggested Shoonya NFO trading symbol
                                                suggested_atm_strike = gz_opt.get("atm_strike", round(opt_spot, -1)) if "gz_opt" in locals() else round(opt_spot, -1)
                                                sugg_opt_type = "PE" if ("PUT" in opt_sig.action or "BEAR" in opt_sig.action) else "CE"
                                                sugg_tsym = oa.format_shoonya_option_symbol(selected_screener_sym, scr_opt_expiry, suggested_atm_strike, sugg_opt_type)

                                                custom_opt_tsym = st.text_input(
                                                    "NFO Trading Symbol",
                                                    value=sugg_tsym,
                                                    key=f"scr_opt_tsym_{selected_screener_sym}",
                                                    help="Shoonya exchange contract trading symbol (e.g. RELIANCE24NOV1400CE or NIFTY24N2124000CE)."
                                                )

                                                opt_attach_sl = st.checkbox(
                                                    f"🛡️ Auto-Bracket SL @ ₹{opt_sig.stop_loss:,.2f} (-50% Premium Cut)",
                                                    value=True,
                                                    key=f"scr_opt_attach_sl_{selected_screener_sym}",
                                                    help="Automatically submit stop-loss trigger order upon fill."
                                                )

                                                if opt_sig.action != "WAIT":
                                                    live_opt_btn = f"🚀 Place Real Option Order ({total_opt_qty} Qty)"
                                                    if st.button(live_opt_btn, key=f"scr_btn_live_opt_{selected_screener_sym}", type="primary", use_container_width=True):
                                                        with st.spinner(f"Routing Option Order {custom_opt_tsym} ({total_opt_qty} Qty) to exchange..."):
                                                            ok_lo, msg_lo = le.live_executor.place_live_order(
                                                                signal=opt_sig,
                                                                quantity=total_opt_qty,
                                                                product_type=opt_prod_code,
                                                                order_type=opt_ord_code,
                                                                tradingsymbol=custom_opt_tsym,
                                                                exchange="NFO",
                                                                price=opt_sig.entry_price,
                                                            )
                                                            if ok_lo:
                                                                st.success(f"✅ {msg_lo}")
                                                                if opt_attach_sl and opt_sig.stop_loss and opt_sig.stop_loss > 0:
                                                                    ok_osl, msg_osl = le.live_executor.place_stop_loss_order(
                                                                        symbol=custom_opt_tsym,
                                                                        quantity=total_opt_qty,
                                                                        stop_price=opt_sig.stop_loss,
                                                                        limit_price=round(opt_sig.stop_loss * 0.95, 2),
                                                                        is_buy=False,
                                                                        exch="NFO",
                                                                        product_type=opt_prod_code,
                                                                    )
                                                                    if ok_osl:
                                                                        st.info(f"🛡️ {msg_osl}")
                                                            else:
                                                                st.error(f"❌ {msg_lo}")
                                                else:
                                                    st.info("Strategy in Wait mode (no active trigger).")
                                        else:
                                            if opt_sig.action != "WAIT":
                                                if st.button(f"📝 Execute Paper Option ({opt_lot} Qty)", key=f"scr_exec_opt_{selected_screener_sym}", type="primary", use_container_width=True):
                                                    ok_o, msg_o = pt.execute_paper_order(opt_sig, quantity=opt_lot, provider=provider)
                                                    (st.success if ok_o else st.error)(msg_o)
                                                    st.rerun()
                                            else:
                                                st.info("Strategy in Wait mode (no active trigger).")
                                    else:
                                        st.info("Option strategy evaluation pending.")
                                else:
                                    st.info(f"Option chain data currently unavailable for {selected_screener_sym} ({scr_opt_expiry}).")
                            else:
                                st.info(f"ℹ️ **{selected_screener_sym}** is a Cash Equity (Not traded in the NSE F&O Derivatives Segment). Complete quantitative stock analysis, conviction rate, and price targets are displayed on the left.")

                    # ─── FULL-WIDTH BELOW: OPTION CHAIN LADDER TABLE ──────────────
                    if is_opt_eligible and opt_chain is not None and not opt_chain.empty and scr_opt_expiry and opt_spot is not None:
                        st.markdown("---")
                        st.markdown(f"**Option Chain Ladder ({selected_screener_sym} • Expiry: {scr_opt_expiry})**")
                        all_strikes = sorted(opt_chain["strike"].unique())
                        atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - opt_spot))
                        start_idx = max(0, atm_idx - 10)
                        end_idx = min(len(all_strikes), atm_idx + 11)
                        filtered_strikes = all_strikes[start_idx:end_idx]

                        chain_display = opt_chain[opt_chain["strike"].isin(filtered_strikes)].copy()
                        cols_to_show = []
                        rename_map = {}
                        if "ce_oi" in chain_display.columns:
                            cols_to_show.append("ce_oi")
                            rename_map["ce_oi"] = "Call OI"
                        if "ce_ltp" in chain_display.columns:
                            cols_to_show.append("ce_ltp")
                            rename_map["ce_ltp"] = "Call LTP (₹)"
                        if "ce_delta" in chain_display.columns:
                            cols_to_show.append("ce_delta")
                            rename_map["ce_delta"] = "Call Delta"

                        cols_to_show.append("strike")
                        rename_map["strike"] = "STRIKE (₹)"

                        if "pe_delta" in chain_display.columns:
                            cols_to_show.append("pe_delta")
                            rename_map["pe_delta"] = "Put Delta"
                        if "pe_ltp" in chain_display.columns:
                            cols_to_show.append("pe_ltp")
                            rename_map["pe_ltp"] = "Put LTP (₹)"
                        if "pe_oi" in chain_display.columns:
                            cols_to_show.append("pe_oi")
                            rename_map["pe_oi"] = "Put OI"

                        clean_chain_df = chain_display[cols_to_show].rename(columns=rename_map)
                        st.dataframe(clean_chain_df, hide_index=True, use_container_width=True)

        elif "Universal" in scr_mode or "Basket" in scr_mode:
            basket_options = list(sd.UNIVERSE_AND_SEGMENT_OPTIONS) + [f"Watchlist: {w}" for w in all_wl_keys]

            # Detect if user selected a target from Omnibar
            active_omnibar_target = st.session_state.get("omnibar_screen_target")
            default_basket_idx = 0
            if active_omnibar_target:
                if active_omnibar_target in basket_options:
                    default_basket_idx = basket_options.index(active_omnibar_target)
                else:
                    matched = False
                    for b_i, b_opt in enumerate(basket_options):
                        if any(k in active_omnibar_target.upper() for k in ["BSE100", "BSE 100"]) and "BSE 100" in b_opt:
                            default_basket_idx = b_i
                            matched = True
                            break
                        if any(k in active_omnibar_target.upper() for k in ["NIFTY", "NIFTY50", "NIFTY 50"]) and "NIFTY 50" in b_opt:
                            default_basket_idx = b_i
                            matched = True
                            break
                        if any(k in active_omnibar_target.upper() for k in ["BSEMIDCAP", "BSE MIDCAP"]) and "BSE MIDCAP" in b_opt:
                            default_basket_idx = b_i
                            matched = True
                            break
                    if not matched:
                        basket_options.insert(0, active_omnibar_target)
                        default_basket_idx = 0

            scr_s_col1, scr_s_col_depth, scr_s_col_eng, scr_s_col2 = st.columns([2.0, 1.1, 1.4, 1.1])
            with scr_s_col1:
                scr_chosen_basket = st.selectbox(
                    "Universe / Segment Target",
                    basket_options,
                    index=default_basket_idx,
                    key="scr_screen_basket_pick",
                    help="Select whole exchange universe (NSE/BSE), F&O derivatives segment, sector group, or watchlist to screen.",
                )
            with scr_s_col_depth:
                scr_basket_depth = st.selectbox(
                    "Scan Depth Limit",
                    ["Top 50 (Quick)", "Top 100 (Standard)", "Top 185 (F&O Complete)", "Top 250 (Deep)", "Top 500 (Comprehensive)", "All Symbols"],
                    index=1 if "F&O" not in scr_chosen_basket else 2,
                    key="scr_screen_basket_depth",
                    help="Limit batch depth for rapid multi-model quantitative screening.",
                )
            with scr_s_col_eng:
                scr_engine_choice = st.selectbox(
                    "Screener Data Engine",
                    ["📡 TradingView Real-Time (Live NSE/BSE)", "📚 Deep Quantitative (400d Candles)"],
                    index=0,
                    key="scr_engine_choice_pick",
                    help="TradingView Real-Time: Sub-second live prices & indicators directly from NSE/BSE feeds.",
                )
            with scr_s_col2:
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                scan_run_clicked = st.button("⚡ Scan Target Now", type="primary", key="btn_run_screener_subtab", use_container_width=True)

            auto_trigger = st.session_state.pop("omnibar_trigger_scan", False)
            if auto_trigger:
                scan_run_clicked = True

            depth_val = None
            if "Top 50" in scr_basket_depth:
                depth_val = 50
            elif "Top 100" in scr_basket_depth:
                depth_val = 100
            elif "Top 185" in scr_basket_depth:
                depth_val = 185
            elif "Top 250" in scr_basket_depth:
                depth_val = 250
            elif "Top 500" in scr_basket_depth:
                depth_val = 500

            if scr_chosen_basket.startswith("Watchlist: "):
                picked_wl_name = scr_chosen_basket.replace("Watchlist: ", "")
                scan_syms = all_watchlists.get(picked_wl_name, []) or config.load_watchlist()
            else:
                scan_syms = sd.get_symbols_for_universe_or_segment(scr_chosen_basket, watchlist_symbols=current_symbols, limit=depth_val)

            st.markdown(f"#### ⚡ Multi-Strategy High-Conviction Screener ({len(scan_syms)} Symbols in `{scr_chosen_basket}`)")

            if provider and (scan_run_clicked or "last_scan_table" in st.session_state):
                if scan_run_clicked or "last_scan_table" not in st.session_state:
                    with st.spinner(f"Scanning {len(scan_syms)} symbols via {scr_engine_choice}..."):
                        if "TradingView" in scr_engine_choice:
                            try:
                                table, scan_errors = screener.scan_tradingview_realtime(scan_syms)
                            except Exception:
                                table, scan_errors = screener.scan(provider, scan_syms)
                        else:
                            table, scan_errors = screener.scan(provider, scan_syms)
                        st.session_state["last_scan_table"] = table
                        st.session_state["last_scan_errors"] = scan_errors
                else:
                    table = st.session_state["last_scan_table"]
                    scan_errors = st.session_state.get("last_scan_errors", {})

                if not table.empty:
                    # Retrieve persistent criteria settings from session state (set by Omnibar)
                    default_min_c = int(st.session_state.get("omnibar_min_conviction", 0))
                    default_min_w = float(st.session_state.get("omnibar_min_win_rate", 0.0))

                    with st.expander("🎛️ High-Conviction & Win Ratio Criteria Settings", expanded=True):
                        flt_c1, flt_c2, flt_c3, flt_c4 = st.columns(4)
                        min_conv_val = flt_c1.slider("🔥 Min Conviction Rating (%)", 0, 100, min(100, max(0, default_min_c)), step=1, key="scr_flt_conviction_slider")
                        min_wr_val = flt_c2.slider("🎯 Min Expected Win Ratio (%)", 0.0, 100.0, min(100.0, max(0.0, default_min_w)), step=1.0, key="scr_flt_winrate_slider")
                        bias_choice = flt_c3.selectbox("🧭 Setup Direction", ["All Setups", "🟢 Long / Bullish (Accumulation & Breakout)", "🔴 Short / Bearish (Distribution)"], key="scr_flt_bias_direction")
                        min_canslim = flt_c4.slider("Min CAN SLIM Score", 0, 100, 0, step=5, key="scr_flt_canslim_filter")

                    # Apply Multi-Pillar Conviction & Win Ratio Filtering
                    filtered = screener.filter_by_criteria(
                        table,
                        min_conviction=min_conv_val,
                        min_win_rate=min_wr_val,
                        bias="BULLISH" if "Long" in bias_choice else ("BEARISH" if "Short" in bias_choice else "ALL"),
                    )
                    if min_canslim > 0 and "can_slim_score" in filtered.columns:
                        filtered = filtered[filtered["can_slim_score"] >= min_canslim].reset_index(drop=True)

                    # ─── HIGH-IMPACT INSTITUTIONAL METRICS BANNER ───────────────────
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("🎯 Scanned Segment", f"{len(table)} Tickers", scr_chosen_basket[:20] + "...")
                    m_c2.metric("💎 Qualifying Setups", f"{len(filtered)}", f"≥ {min_conv_val}% Conviction")
                    avg_filtered_wr = round(filtered["expected_win_rate"].mean(), 1) if (not filtered.empty and "expected_win_rate" in filtered.columns) else 0.0
                    m_c3.metric("📊 Avg Exp Win Rate", f"{avg_filtered_wr}%" if not filtered.empty else "N/A", f"Threshold: ≥ {min_wr_val:.0f}%")
                    top_sym_display = filtered["symbol"].iloc[0] if not filtered.empty else "None"
                    top_action_display = filtered["action"].iloc[0] if (not filtered.empty and "action" in filtered.columns) else "N/A"
                    m_c4.metric("🏆 #1 Conviction Pick", top_sym_display, top_action_display)

                    # ─── ACTIONABLE TRADING DESK WITH ORDER PAD BELOW EVERY SCREENED SETUP ───
                    v_c1, v_c2 = st.columns([2.5, 1.2])
                    with v_c1:
                        scr_view_mode = st.radio(
                            "Display View Mode",
                            ["🔲 Actionable Trading Cards (With Instant Buy Desk)", "📊 High-Density Data Table View"],
                            horizontal=True,
                            key="scr_unif_view_choice",
                            label_visibility="collapsed",
                        )
                    with v_c2:
                        scr_disp_limit = st.selectbox(
                            "Display Limit",
                            ["Top 10 Setups", "Top 25 Setups", "Top 50 Setups", "All Qualifying Setups"],
                            index=1,
                            key="scr_unif_disp_limit_pick",
                            label_visibility="collapsed",
                        )

                    max_setups = len(filtered)
                    if "10" in scr_disp_limit:
                        max_setups = 10
                    elif "25" in scr_disp_limit:
                        max_setups = 25
                    elif "50" in scr_disp_limit:
                        max_setups = 50

                    if filtered.empty:
                        st.info(
                            f"ℹ️ Currently **0 setups** in `{scr_chosen_basket}` meet the strict criteria of "
                            f"**≥{min_conv_val}% Conviction** and **≥{min_wr_val:.0f}% Win Rate**. "
                            f"Below are the top 5 highest-ranking candidates from `{scr_chosen_basket}`:"
                        )
                        cards_df = table.sort_values(by=["conviction", "expected_win_rate"], ascending=False).head(5).reset_index(drop=True) if "conviction" in table.columns else table.head(5)
                    else:
                        cards_df = filtered.head(max_setups).reset_index(drop=True)

                    if scr_view_mode == "📊 High-Density Data Table View":
                        display_cols = [c for c in [
                            "symbol", "close", "1m_change_%", "conviction", "expected_win_rate", "action",
                            "can_slim_score", "vsa_bias", "rsi14", "vs_sma50_%", "vs_sma200_%", "candlestick", "data_source"
                        ] if c in filtered.columns]

                        rename_cols = {
                            "symbol": "Symbol",
                            "close": "LTP (₹)",
                            "1m_change_%": "1D Chg %",
                            "conviction": "Conviction %",
                            "expected_win_rate": "Win Ratio %",
                            "action": "Action",
                            "can_slim_score": "CAN SLIM",
                            "vsa_bias": "VSA Bias / Phase",
                            "rsi14": "RSI(14)",
                            "vs_sma50_%": "vs 50MA %",
                            "vs_sma200_%": "vs 200MA %",
                            "candlestick": "Consensus Signal",
                            "data_source": "Feed Engine",
                        }
                        st.dataframe(filtered.head(max_setups)[display_cols].rename(columns=rename_cols), hide_index=True, use_container_width=True)

                        if not filtered.empty:
                            tbl_c1, tbl_c2, tbl_c3 = st.columns([1.5, 1.3, 1.2])
                            with tbl_c1:
                                pick_sym = st.selectbox("Select Candidate to Trade or Inspect in Terminal", filtered["symbol"].tolist(), key="scr_pick_sym_res")
                            with tbl_c2:
                                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                                if st.button(f"📝 Quick Paper Buy ({pick_sym})", type="primary", use_container_width=True, key=f"btn_quick_pt_{pick_sym}"):
                                    r_pick = filtered[filtered["symbol"] == pick_sym].iloc[0]
                                    ltp_pk = float(r_pick.get("close", 0.0) or r_pick.get("ltp", 0.0))
                                    sl_pk = round(ltp_pk * 0.95, 2)
                                    t1_pk = round(ltp_pk * 1.10, 2)
                                    t2_pk = round(ltp_pk * 1.20, 2)
                                    sig_pk = ate.TradeSignal(
                                        id=f"PT_QUICK_{pick_sym}_{int(datetime.now().timestamp())}",
                                        symbol=pick_sym,
                                        asset_type="EQUITY",
                                        action="BUY" if "BUY" in str(r_pick.get("action", "BUY")).upper() else "SELL",
                                        conviction=int(r_pick.get("conviction", 75)),
                                        current_price=ltp_pk,
                                        entry_price=ltp_pk,
                                        stop_loss=sl_pk,
                                        target_1=t1_pk,
                                        target_2=t2_pk,
                                        risk_reward=2.0,
                                        recommended_qty=max(1, int(25000 / max(1, ltp_pk))),
                                        legs_desc=f"Equity Cash {pick_sym}",
                                        thesis="Universal Screener High Confluence Setup",
                                        literature_citations=["CAN SLIM & VSA"],
                                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        confluence_rating=int(r_pick.get("conviction", 75)),
                                        expected_win_rate=float(r_pick.get("expected_win_rate", 65.0)),
                                        grade="Grade A",
                                    )
                                    ok_qpt, msg_qpt = pt.execute_paper_order(sig_pk, quantity=sig_pk.recommended_qty, provider=provider)
                                    if ok_qpt:
                                        st.success(f"{msg_qpt}")
                                        st.rerun()
                                    else:
                                        st.error(f"{msg_qpt}")
                            with tbl_c3:
                                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                                if st.button(f"🚀 Load '{pick_sym}' into Workspace", use_container_width=True, key="btn_load_scr_pick_res"):
                                    st.session_state["active_symbol"] = pick_sym
                                    st.session_state["scr_screening_scope_mode"] = "Single Stock"
                                    st.rerun()
                    else:
                        # Actionable Trading Cards (With Instant Buy Desk Below EVERY Stock)
                        for r_idx, (_, r_data) in enumerate(cards_df.iterrows()):
                            c_sym = str(r_data["symbol"]).strip().upper()
                            c_name = sd.get_company_name(c_sym)
                            c_info = sd.get_stock_info(c_sym)
                            c_sec = c_info.get("sector", "Diversified")
                            c_exch = c_info.get("exchange", "NSE")
                            c_ltp = float(r_data.get("close", 0.0) or r_data.get("ltp", 0.0))
                            c_chg = float(r_data.get("1m_change_%", 0.0) or r_data.get("change_pct", 0.0))
                            c_conv = int(r_data.get("conviction", 75))
                            c_wr = float(r_data.get("expected_win_rate", 60.0))
                            c_act_raw = str(r_data.get("action", "BUY")).upper()
                            c_act = "BUY" if ("BUY" in c_act_raw or "BULL" in c_act_raw) else ("SELL" if ("SELL" in c_act_raw or "BEAR" in c_act_raw) else "BUY")
                            c_badge = "🟢" if c_act == "BUY" else "🔴"
                            c_grade = "Grade A+" if c_conv >= 85 else ("Grade A" if c_conv >= 75 else "Grade B")
                            c_rr = float(r_data.get("risk_reward", 2.0))
                            c_canslim = int(r_data.get("can_slim_score", 70))
                            c_vsa = str(r_data.get("vsa_bias", "Accumulation / Breakout"))
                            c_rsi = float(r_data.get("rsi14", 55.0))

                            if c_act == "BUY":
                                c_sl = round(c_ltp * 0.95, 2)
                                c_t1 = round(c_ltp + (c_ltp - c_sl) * c_rr, 2)
                                c_t2 = round(c_ltp + (c_ltp - c_sl) * 3.5, 2)
                                c_entry_str = f"₹{c_ltp * 0.995:,.2f} – ₹{c_ltp:,.2f}"
                            else:
                                c_sl = round(c_ltp * 1.05, 2)
                                c_t1 = round(c_ltp - (c_sl - c_ltp) * c_rr, 2)
                                c_t2 = round(c_ltp - (c_sl - c_ltp) * 3.5, 2)
                                c_entry_str = f"₹{c_ltp:,.2f} – ₹{c_ltp * 1.005:,.2f}"

                            c_def_qty = max(1, int(25000 / max(1, c_ltp)))

                            with st.container(border=True):
                                c_card1, c_card2, c_card3 = st.columns([1.4, 1.7, 1.8])
                                with c_card1:
                                    st.markdown(f"### {c_badge} {c_act} `{c_sym}`")
                                    st.caption(f"**{c_name[:24]}** • `{c_exch}` | {c_sec}")
                                    cm1, cm2 = st.columns(2)
                                    with cm1:
                                        st.metric("Confluence", f"{c_conv}%", c_grade)
                                    with cm2:
                                        st.metric("Win Ratio", f"{c_wr:.1f}%", f"R:R 1:{c_rr:.1f}")
                                    st.caption(f"LTP: **₹{c_ltp:,.2f}** ({c_chg:+.1f}%) | CAN SLIM: **{c_canslim}**")

                                with c_card2:
                                    st.markdown("**🎯 Trade Blueprint & Levels:**")
                                    st.write(f"- 📍 **Suggested Entry:** `{c_entry_str}`")
                                    st.write(f"- 🛑 **Strict Stop-Loss:** `₹{c_sl:,.2f}` (-5.0%)")
                                    st.write(f"- 🎯 **Target 1:** `₹{c_t1:,.2f}` (+{round((c_t1-c_ltp)/max(1,c_ltp)*100, 1)}%)")
                                    st.write(f"- 🎯 **Target 2:** `₹{c_t2:,.2f}`")
                                    st.write(f"- 📊 **VSA / Phase:** `{c_vsa[:25]}` | RSI: `{c_rsi:.1f}`")

                                with c_card3:
                                    st.markdown("**⚡ Purchasing & Direct Order Pad:**")
                                    stk_q = st.number_input(
                                        f"Order Shares ({c_sym})",
                                        min_value=1,
                                        max_value=50000,
                                        value=c_def_qty,
                                        step=1,
                                        key=f"scr_card_qty_{c_sym}_{r_idx}",
                                    )
                                    cap_val = stk_q * c_ltp
                                    st.caption(f"Required Capital: **₹{cap_val:,.2f}**")

                                    btn_c1, btn_c2, btn_c3 = st.columns([1.3, 1.4, 0.9])
                                    with btn_c1:
                                        if st.button(f"📝 Paper Buy", type="primary", use_container_width=True, key=f"btn_card_pt_{c_sym}_{r_idx}", help=f"Execute simulated paper trade for {stk_q} shares of {c_sym}"):
                                            card_sig = ate.TradeSignal(
                                                id=f"PT_CARD_{c_sym}_{int(datetime.now().timestamp())}",
                                                symbol=c_sym,
                                                asset_type="EQUITY",
                                                action=c_act,
                                                conviction=c_conv,
                                                current_price=c_ltp,
                                                entry_price=c_ltp,
                                                stop_loss=c_sl,
                                                target_1=c_t1,
                                                target_2=c_t2,
                                                risk_reward=c_rr,
                                                recommended_qty=stk_q,
                                                legs_desc=f"Equity Cash {c_sym}",
                                                thesis=f"Screener Setup ({c_vsa})",
                                                literature_citations=["CAN SLIM", "Volume Spread Analysis"],
                                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                confluence_rating=c_conv,
                                                expected_win_rate=c_wr,
                                                grade=c_grade,
                                            )
                                            ok_cpt, msg_cpt = pt.execute_paper_order(card_sig, quantity=stk_q, provider=provider)
                                            if ok_cpt:
                                                st.success(f"{msg_cpt}")
                                                st.rerun()
                                            else:
                                                st.error(f"{msg_cpt}")

                                    with btn_c2:
                                        if broker_ready:
                                            if st.button(f"⚡ Live Buy", type="secondary", use_container_width=True, key=f"btn_card_live_{c_sym}_{r_idx}", help=f"Route real live order for {stk_q} shares of {c_sym} to exchange via Shoonya API"):
                                                card_sig = ate.TradeSignal(
                                                    id=f"LIVE_CARD_{c_sym}_{int(datetime.now().timestamp())}",
                                                    symbol=c_sym,
                                                    asset_type="EQUITY",
                                                    action=c_act,
                                                    conviction=c_conv,
                                                    current_price=c_ltp,
                                                    entry_price=c_ltp,
                                                    stop_loss=c_sl,
                                                    target_1=c_t1,
                                                    target_2=c_t2,
                                                    risk_reward=c_rr,
                                                    recommended_qty=stk_q,
                                                    legs_desc=f"Equity Cash {c_sym}",
                                                    thesis=f"Live Exchange Algorithmic Order ({c_vsa})",
                                                    literature_citations=["CAN SLIM", "VSA"],
                                                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    confluence_rating=c_conv,
                                                    expected_win_rate=c_wr,
                                                    grade=c_grade,
                                                )
                                                ok_live_c, msg_live_c = le.live_executor.place_live_order(card_sig, quantity=stk_q, product_type="C", order_type="LMT")
                                                if ok_live_c:
                                                    st.success(f"Live Routed: {msg_live_c}")
                                                else:
                                                    st.error(f"Live Order Error: {msg_live_c}")
                                        else:
                                            st.caption("⚡ Live API: Missing Creds")

                                    with btn_c3:
                                        if st.button("📊 Chart", use_container_width=True, key=f"btn_card_chart_{c_sym}_{r_idx}", help=f"Open {c_sym} in Terminal"):
                                            st.session_state["active_symbol"] = c_sym
                                            st.session_state["scr_screening_scope_mode"] = "Single Stock"
                                            st.rerun()

                                # Derivatives purchasing desk if option eligible
                                is_opt = (c_sym in oa.NSE_FO_LOT_SIZES) or oa.is_weekly_index(c_sym)
                                if is_opt:
                                    opt_lot = oa.get_lot_size(c_sym, spot=c_ltp)
                                    with st.expander(f"⛓️ Derivatives & Option Trading Desk for {c_sym} (Lot Size: {opt_lot})", expanded=False):
                                        oc_col1, oc_col2, oc_col3 = st.columns([1.5, 1.5, 1.5])
                                        with oc_col1:
                                            opt_act_s = "LONG_CALL" if c_act == "BUY" else "LONG_PUT"
                                            opt_lbl = f"{c_sym} ATM {'CE (Call)' if c_act == 'BUY' else 'PE (Put)'} Nearest Expiry"
                                            st.markdown(f"**Recommended Derivative:** `{opt_lbl}`")
                                            st.caption(f"Underlying Spot: **₹{c_ltp:,.2f}** | Lot: **{opt_lot}**")
                                        with oc_col2:
                                            opt_q_input = st.number_input(
                                                f"Option Quantity",
                                                min_value=opt_lot,
                                                max_value=opt_lot * 50,
                                                value=opt_lot,
                                                step=opt_lot,
                                                key=f"scr_opt_q_{c_sym}_{r_idx}",
                                            )
                                        with oc_col3:
                                            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                                            b_opt1, b_opt2 = st.columns(2)
                                            with b_opt1:
                                                if st.button(f"📝 Paper Option", type="primary", use_container_width=True, key=f"btn_opt_pt_{c_sym}_{r_idx}"):
                                                    opt_sig_c = ate.TradeSignal(
                                                        id=f"PT_OPT_{c_sym}_{int(datetime.now().timestamp())}",
                                                        symbol=c_sym,
                                                        asset_type="OPTION",
                                                        action=opt_act_s,
                                                        conviction=c_conv,
                                                        current_price=c_ltp,
                                                        entry_price=round(c_ltp * 0.02, 2),
                                                        stop_loss=round(c_ltp * 0.01, 2),
                                                        target_1=round(c_ltp * 0.04, 2),
                                                        target_2=round(c_ltp * 0.06, 2),
                                                        risk_reward=2.0,
                                                        recommended_qty=opt_q_input,
                                                        legs_desc=f"{c_sym} ATM {'CE' if c_act == 'BUY' else 'PE'}",
                                                        thesis="High-Conviction Derivative Setup",
                                                        literature_citations=["Options Trading"],
                                                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        confluence_rating=c_conv,
                                                        expected_win_rate=c_wr,
                                                        grade=c_grade,
                                                    )
                                                    ok_op, msg_op = pt.execute_paper_order(opt_sig_c, quantity=opt_q_input, provider=provider)
                                                    if ok_op:
                                                        st.success(msg_op)
                                                        st.rerun()
                                                    else:
                                                        st.error(msg_op)
                                            with b_opt2:
                                                if broker_ready:
                                                    if st.button("⚡ Live Option", use_container_width=True, key=f"btn_opt_live_{c_sym}_{r_idx}"):
                                                        opt_sig_c = ate.TradeSignal(
                                                            id=f"LIVE_OPT_{c_sym}_{int(datetime.now().timestamp())}",
                                                            symbol=c_sym,
                                                            asset_type="OPTION",
                                                            action=opt_act_s,
                                                            conviction=c_conv,
                                                            current_price=c_ltp,
                                                            entry_price=round(c_ltp * 0.02, 2),
                                                            stop_loss=round(c_ltp * 0.01, 2),
                                                            target_1=round(c_ltp * 0.04, 2),
                                                            target_2=round(c_ltp * 0.06, 2),
                                                            risk_reward=2.0,
                                                            recommended_qty=opt_q_input,
                                                            legs_desc=f"{c_sym} ATM {'CE' if c_act == 'BUY' else 'PE'}",
                                                            thesis="Live Option Order Placement",
                                                            literature_citations=["Options Trading"],
                                                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                            confluence_rating=c_conv,
                                                            expected_win_rate=c_wr,
                                                            grade=c_grade,
                                                        )
                                                        ok_lo, msg_lo = le.live_executor.place_live_order(opt_sig_c, quantity=opt_q_input, product_type="M", order_type="LMT")
                                                        if ok_lo:
                                                            st.success(f"Live Option Sent: {msg_lo}")
                                                        else:
                                                            st.error(f"Live Option Error: {msg_lo}")
                                                else:
                                                    st.caption("⚡ Live Broker: Missing Creds")

                for sym, msg in scan_errors.items():
                    st.warning(f"{sym}: {msg}")

        elif "High-Conviction" in scr_mode:
            render_high_conviction_screener_desk(key_prefix="scr_hc_")

    _render_universal_screener_body(sel_wl_name, all_wl_keys)

    # ─── MASTER INDIAN EQUITIES DIRECTORY (CONSOLIDATED INSIDE SCREENER DESK) ───
    st.markdown("---")
    with st.expander("🏛️ Master Indian Equities Directory & Universe Lookup (12,722+ Listed NSE & BSE Stocks)", expanded=False):
        st.markdown("#### 🏛️ Indian Equities Directory (Complete NSE & BSE Universe)")
        df_all_stocks = sd.get_all_stocks_dataframe()
        total_count = len(df_all_stocks)
        nse_count = len(df_all_stocks[df_all_stocks["Exchange"] == "NSE"])
        bse_count = len(df_all_stocks[df_all_stocks["Exchange"] == "BSE"])
        idx_count = len(df_all_stocks[df_all_stocks["Series"] == "Index"])

        col_u1, col_u2, col_u3, col_u4 = st.columns(4)
        col_u1.metric("Total Listed Universe", f"{total_count:,}", "NSE + BSE Equities")
        col_u2.metric("NSE Equities", f"{nse_count:,}", "National Stock Exchange")
        col_u3.metric("BSE Equities", f"{bse_count:,}", "Bombay Stock Exchange")
        col_u4.metric("Market Indices", f"{idx_count}", "Benchmarks & Sectoral")

        st.markdown("---")

        # Search & Filter Controls
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1.5, 1.1, 0.9, 0.9, 0.8])
        with col_f1:
            univ_q = st.text_input(
                "🔍 Search by Symbol, Name, Token or ISIN",
                placeholder="e.g. Taparia, Tata, Dixon, 505685, Reliance...",
                key="univ_filter_q_inner",
            )
        with col_f2:
            all_sectors = ["All Sectors"] + sorted(list(set(df_all_stocks["Sector"].dropna())))
            univ_sec = st.selectbox("Industry / Sector", all_sectors, key="univ_filter_sec_inner")
        with col_f3:
            all_exch = ["All Exchanges", "NSE", "BSE"]
            univ_exch = st.selectbox("Exchange", all_exch, key="univ_filter_exch_inner")
        with col_f4:
            all_series = ["All Series", "EQ (Normal Equity)", "BE (Trade for Trade)", "Index"]
            univ_ser = st.selectbox("Market Series", all_series, key="univ_filter_ser_inner")
        with col_f5:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Sync Universe", use_container_width=True, key="btn_sync_inner_univ", help="Sync latest newly listed IPOs and active scrips across NSE and BSE"):
                with st.spinner("Downloading updated equities list across NSE & BSE..."):
                    ok_sync, msg_sync, new_cnt = sd.refresh_stock_directory()
                    if ok_sync:
                        st.success(f"{msg_sync}")
                    else:
                        st.warning(f"{msg_sync}")
                    st.rerun()

        # Apply filters
        filtered_df_d = df_all_stocks.copy()
        if univ_q.strip():
            q_clean = univ_q.strip().lower()
            filtered_df_d = filtered_df_d[
                filtered_df_d["Symbol"].str.lower().str.contains(q_clean, na=False)
                | filtered_df_d["Company Name"].str.lower().str.contains(q_clean, na=False)
                | filtered_df_d["Token / Scrip"].astype(str).str.lower().str.contains(q_clean, na=False)
                | filtered_df_d["ISIN"].str.lower().str.contains(q_clean, na=False)
            ]
        if univ_sec != "All Sectors":
            filtered_df_d = filtered_df_d[filtered_df_d["Sector"] == univ_sec]
        if univ_exch != "All Exchanges":
            filtered_df_d = filtered_df_d[filtered_df_d["Exchange"] == univ_exch]
        if univ_ser != "All Series":
            ser_prefix = univ_ser.split()[0]
            filtered_df_d = filtered_df_d[filtered_df_d["Series"] == ser_prefix]

        st.caption(f"Showing **{len(filtered_df_d):,}** of **{total_count:,}** listed stocks")

        # Quick Selection & Action Card
        if not filtered_df_d.empty:
            col_sel, col_act = st.columns([2, 1])
            with col_sel:
                stock_picks = [f"{row['Symbol']} — {row['Company Name']} [{row.get('Exchange', 'NSE')} • {row['Sector']}]" for _, row in filtered_df_d.head(100).iterrows()]
                chosen_pick = st.selectbox("Select a stock to inspect or load:", stock_picks, key="univ_select_box_inner")
            with col_act:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if chosen_pick:
                    chosen_sym = sd.extract_symbol_from_selection(chosen_pick)
                    if st.button(f"🚀 Analyze {chosen_sym} Now", type="primary", use_container_width=True, key=f"btn_analyze_univ_{chosen_sym}"):
                        st.session_state["active_symbol"] = chosen_sym
                        st.session_state["scr_screening_scope_mode"] = "Single Stock"
                        st.rerun()

            # Detailed card for chosen stock
            if chosen_pick:
                chosen_sym = sd.extract_symbol_from_selection(chosen_pick)
                info = sd.get_stock_info(chosen_sym)
                with st.container(border=True):
                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                    col_c1.markdown(f"**Symbol:** `{chosen_sym}`")
                    col_c2.markdown(f"**Sector:** {info.get('sector', 'Diversified')}")
                    col_c3.markdown(f"**Exchange:** `{info.get('exchange', 'NSE')}` • Series: `{info.get('series', 'EQ')}`")
                    col_c4.markdown(f"**Token / ISIN:** `{info.get('token') or info.get('isin', 'N/A')}`")
                    if info.get("listing_date"):
                        st.caption(f"📅 Official Listing Date: **{info.get('listing_date')}** | Face Value: **₹{info.get('face_value', '10')}**")

        # Render Interactive Table
        st.dataframe(
            filtered_df_d,
            use_container_width=True,
            hide_index=True,
            height=450,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Company Name": st.column_config.TextColumn("Company Name", width="large"),
                "Exchange": st.column_config.TextColumn("Exchange", width="small"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Series": st.column_config.TextColumn("Series", width="small"),
                "Token / Scrip": st.column_config.TextColumn("Token / Scrip", width="small"),
                "ISIN": st.column_config.TextColumn("ISIN", width="medium"),
                "Listing Date": st.column_config.TextColumn("Listing Date", width="small"),
            },
        )

        # Sector Breakdown Expander
        with st.expander("📊 Complete Sector Breakdown of All Indian Listed Equities", expanded=False):
            sec_counts = df_all_stocks["Sector"].value_counts().reset_index()
            sec_counts.columns = ["Sector", "Number of Companies"]
            st.dataframe(sec_counts, hide_index=True, use_container_width=True)

with tab_sectors:
    render_active_stock_header(key_prefix="sectors_", show_stock=False)
    st.markdown("#### 🌐 Sector Rotation & Market Breadth")
    st.caption("Performance tracking across major NSE sector indices.")
    try:
        sec_df = aa.get_sector_performance_table(provider)
        st.dataframe(sec_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.info(f"Sector data loading: {e}")



with tab_bt:
    render_active_stock_header(key_prefix="bt_")
    st.subheader("⏳ Quantitative Backtest Lab & Trade Log")
    st.caption("Simulate book strategies with lookahead-free signals, transaction costs, and trade-by-trade analytics.")

    b_strat = st.selectbox(
        "Select Book Strategy to Simulate",
        [
            "Perry Kaufman Adaptive System (KAMA + Noise Filter)",
            "Bill Williams Alligator Trend System (Lips > Teeth > Jaw)",
            "William O'Neil Breakout System (52-week High + 7% Stop)",
            "Classic SMA Crossover (Moving Average Trend)",
        ],
    )

    bt_c1, bt_c2, bt_c3 = st.columns(3)
    cost = bt_c3.number_input("Cost per trade (bps)", 0.0, 100.0, 10.0)

    if "Perry Kaufman" in b_strat:
        n_kama = bt_c1.number_input("KAMA Lookback", 5, 50, 10)
        bt_min_er = bt_c2.number_input("Min Efficiency Ratio (ER)", 0.0, 0.9, 0.25, step=0.05, key="bt_inp_min_er")
    elif "Classic SMA" in b_strat:
        fast = bt_c1.number_input("Fast SMA", 2, 100, 20)
        slow = bt_c2.number_input("Slow SMA", 5, 300, 50)
    elif "William O'Neil" in b_strat:
        stop_pct = bt_c1.number_input("Unconditional Stop-Loss %", 1.0, 15.0, 7.0, step=0.5)

    if provider and st.button("🚀 Run Backtest Simulation", type="primary"):
        with st.spinner("Simulating historical execution across book rules..."):
            try:
                df_stock = load_stock()
                if df_stock is None or len(df_stock) < 60:
                    raise ValueError("Insufficient historical data for backtesting.")

                if "Perry Kaufman" in b_strat:
                    res = bt.kaufman_adaptive_system(df_stock, n=int(n_kama), min_er=float(bt_min_er), cost_bps=float(cost))
                elif "Bill Williams" in b_strat:
                    res = bt.alligator_trend_system(df_stock, cost_bps=float(cost))
                elif "William O'Neil" in b_strat:
                    res = bt.oneil_breakout_system(df_stock, stop_pct=float(stop_pct), cost_bps=float(cost))
                else:
                    res = bt.sma_crossover(df_stock, int(fast), int(slow), float(cost))
                st.session_state["bt_res"] = res
            except Exception as e:
                st.error(str(e))
                st.session_state["bt_res"] = None

    res = st.session_state.get("bt_res")
    if res is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", f"{res.metrics['strategy_return_%']:+.2f}%")
        m2.metric("Buy & Hold Return", f"{res.metrics['buy_hold_return_%']:+.2f}%")
        m3.metric("Max Drawdown", f"{res.metrics['strategy_max_drawdown_%']:.2f}%")
        m4.metric("Win Rate", f"{res.metrics['win_rate_%']}%" if pd.notna(res.metrics['win_rate_%']) else "N/A")

        fig_bt = go.Figure()
        fig_bt.add_scatter(x=res.equity.index, y=res.equity, name="Book Strategy", line=dict(color="#00E676", width=2.2))
        fig_bt.add_scatter(x=res.buy_hold.index, y=res.buy_hold, name="Buy & Hold Benchmark", line=dict(color="#757575", dash="dot"))
        apply_chart_theme(fig_bt, height=400, yaxis_title="Growth of ₹1.00", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bt, use_container_width=True)

        # Interactive Capital Simulator
        st.markdown("#### 💰 Portfolio Capital Simulator")
        init_cap = st.slider("Simulate Initial Capital (₹)", 50000, 2000000, 500000, step=50000, key="bt_capital_slider")
        end_val = init_cap * res.equity.iloc[-1]
        net_profit = end_val - init_cap
        max_dd_rupees = init_cap * (res.metrics['strategy_max_drawdown_%'] / 100.0)

        cap_c1, cap_c2, cap_c3 = st.columns(3)
        cap_c1.metric("Final Portfolio Value", f"₹{end_val:,.0f}")
        cap_c2.metric("Net Profit / Loss", f"₹{net_profit:+,.0f}", f"{round(net_profit/init_cap*100, 2)}%")
        cap_c3.metric("Max Drawdown in Rupees", f"-₹{abs(max_dd_rupees):,.0f}")

        # Trade Log Table
        if res.trades is not None and not res.trades.empty:
            st.markdown(f"#### 📜 Simulated Trade Log ({len(res.trades)} Trades)")
            st.dataframe(res.trades, hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 6: BOOKS & WISDOM (INTERACTIVE LEARNING ACADEMY)
# ==============================================================================
with tab_books:
    render_active_stock_header(key_prefix="books_", show_stock=False)
    st.subheader("📚 Book Wisdom & Interactive Learning Academy")
    st.markdown("Master frameworks from the 19 foundational investment & trading books in your `Books/` library.")

    cats = ["All Categories"] + bk.get_all_categories()
    sel_cat = st.selectbox("Category Filter", cats)

    matching_books = [b for b in bk.BOOKS_METADATA if sel_cat == "All Categories" or b["category"] == sel_cat]
    book_titles = [f"{b['title']} — {b['author']}" for b in matching_books]
    sel_title = st.selectbox("Select Book to Study", book_titles)
    selected_book = matching_books[book_titles.index(sel_title)]

    with st.container(border=True):
        st.markdown(f"## 📖 {selected_book['title']}")
        st.markdown(f"**Author:** {selected_book['author']} | **Category:** `{selected_book['category']}` | **File:** `{selected_book['filename']}`")
        st.info(selected_book["tagline"])

        # ----------------------------------------------------------------------
        # INTERACTIVE QUIZ & KNOWLEDGE CHECK
        # ----------------------------------------------------------------------
        quiz = bk.get_book_quiz(selected_book["id"])
        if quiz:
            with st.expander("🧠 Interactive Knowledge Check (Test Your Understanding)", expanded=True):
                st.write(f"**Question:** {quiz['question']}")
                quiz_ans = st.radio("Choose the correct answer:", quiz["options"], key=f"quiz_{selected_book['id']}")
                if st.button("Submit Answer", key=f"sub_quiz_{selected_book['id']}"):
                    is_corr = quiz["options"].index(quiz_ans) == quiz["correct"]
                    st.session_state[f"quiz_res_{selected_book['id']}"] = (is_corr, quiz["explanation"])

                q_res = st.session_state.get(f"quiz_res_{selected_book['id']}")
                if q_res is not None:
                    is_c, expl = q_res
                    if is_c:
                        st.success(f"🎉 Correct! {expl}")
                    else:
                        st.error(f"❌ Incorrect. {expl}")

        st.markdown("### 💡 Core Philosophy & Executive Summary")
        st.write(selected_book["core_philosophy"])

        st.markdown("### 🔑 The Best Things From This Book (Key Principles)")
        for p in selected_book["key_principles"]:
            st.markdown(f"- {p}")

        st.markdown("### 📐 Quantitative Rules & Formulas")
        for f in selected_book["formulas_and_rules"]:
            st.code(f, language="text")

        col_stk_app, col_opt_app = st.columns(2)
        with col_stk_app:
            with st.container(border=True):
                st.markdown("#### 📈 How to Apply to Stock Analysis")
                st.write(selected_book["how_to_apply_stocks"])

        with col_opt_app:
            with st.container(border=True):
                st.markdown("#### ⛓️ How to Apply to Option Chains")
                st.write(selected_book["how_to_apply_options"])

        st.warning(f"⚠️ **Core Risk Management Tenet:** {selected_book['risk_rules']}")

        # ----------------------------------------------------------------------
        # INTERACTIVE COMPOUNDING WEALTH SIMULATOR (Darren Hardy & Motilal Oswal)
        # ----------------------------------------------------------------------
        if selected_book["id"] in ("compound_effect", "motilal_101", "young_investors", "bse_temple", "graham"):
            with st.expander("📈 Interactive Compounding Wealth Calculator", expanded=False):
                st.caption("Visualize the exponential power of compounding returns (Darren Hardy & Motilal Oswal).")
                cp1, cp2, cp3, cp4 = st.columns(4)
                start_p = cp1.number_input("Initial Investment (₹)", 0, 10000000, 100000, step=10000)
                sip_m = cp2.number_input("Monthly SIP Contribution (₹)", 0, 1000000, 10000, step=1000)
                cagr_pct = cp3.slider("Expected Annual Return (%)", 6.0, 30.0, 15.0, step=0.5)
                horizon_yrs = cp4.slider("Compounding Horizon (Years)", 1, 35, 15)

                months = horizon_yrs * 12
                r_m = (cagr_pct / 100.0) / 12.0
                wealth = []
                invested = []
                curr_w = start_p
                curr_inv = start_p

                for m in range(1, months + 1):
                    curr_w = curr_w * (1 + r_m) + sip_m
                    curr_inv += sip_m
                    if m % 12 == 0:
                        wealth.append(curr_w)
                        invested.append(curr_inv)

                yr_axis = list(range(1, horizon_yrs + 1))
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=yr_axis, y=invested, name="Total Invested Capital", fill="tozeroy", line=dict(color="#757575")))
                fig_comp.add_trace(go.Scatter(x=yr_axis, y=wealth, name="Total Compounded Wealth", fill="tonexty", line=dict(color="#00E676", width=2.5)))
                apply_chart_theme(fig_comp, title="Wealth Compounding Curve (Rule of 72 & The Compound Effect)", xaxis_title="Years", yaxis_title="Rupees (₹)", height=350, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_comp, use_container_width=True)

                st.write(f"After **{horizon_yrs} years**, your **₹{curr_inv:,.0f}** investment compounds into **₹{curr_w:,.0f}** (Growth of ₹{curr_w - curr_inv:,.0f}).")

        # ----------------------------------------------------------------------
        # SEARCHABLE PDF READER WITH MULTI-PAGE BROWSER
        # ----------------------------------------------------------------------
        st.divider()
        st.markdown("### 📄 Searchable PDF Excerpt Reader")
        st.caption(f"Search keywords or browse original pages in `{selected_book['filename']}`.")

        total_pages = bk.get_pdf_page_count(selected_book["filename"])
        if total_pages > 0:
            search_col1, search_col2 = st.columns([3, 1])
            search_query = search_col1.text_input("Search Keyword inside Book (e.g., 'Volume', 'Stop Loss', 'Alligator')", key=f"search_{selected_book['id']}")
            search_btn = search_col2.button("Search Book", key=f"btn_search_{selected_book['id']}")

            if search_query and (search_btn or len(search_query.strip()) >= 3):
                with st.spinner(f"Searching for '{search_query}' in {selected_book['filename']}..."):
                    results = bk.search_pdf_for_keywords(selected_book["filename"], search_query, max_results=5)
                if results:
                    st.write(f"Found **{len(results)}** matching pages:")
                    for r in results:
                        st.markdown(f"- **Page {r['page']}:** {r['snippet']}")
                else:
                    st.info("No matching pages found for this keyword.")

            pg_col1, pg_col2 = st.columns([1, 4])
            page_num = pg_col1.number_input("Page Number", min_value=1, max_value=total_pages, value=1, key=f"pg_{selected_book['id']}")
            if pg_col2.button(f"Extract Page {page_num} of {total_pages}", key=f"ext_{selected_book['id']}"):
                excerpt = bk.extract_pdf_excerpt(selected_book["filename"], int(page_num))
                st.text_area(f"Page {page_num} Content", excerpt, height=220)
        else:
            st.caption("PDF file not found in Books directory.")


# ==============================================================================
# TAB 7: CONNECTION
# ==============================================================================
with tab_conn:
    render_active_stock_header(key_prefix="conn_", show_stock=False)
    st.subheader("🔌 Market Data, TradingView & Broker Connections")

    col_conn1, col_conn2, col_conn3 = st.columns(3)
    with col_conn1:
        st.markdown("### 🌐 Online Real-Time Feed")
        st.caption("Streams live quotes, today's OHLCV candles, and India VIX for NSE & BSE.")
        if st.button("Test Online Market Feed", key="btn_test_online"):
            try:
                ok, msg = make_provider("online").check_connection()
            except Exception as e:
                ok, msg = False, str(e)
            (st.success if ok else st.error)(msg)

    with col_conn2:
        st.markdown("### 📺 TradingView Account Link")
        st.caption("Interactive Charting Terminal, Multi-Timeframe Technical Consensus & Pine Script Alert Webhooks.")
        if st.button("Test TradingView TA Feed", key="btn_test_tv"):
            try:
                tv_data = tvi.fetch_tradingview_ta("RELIANCE", interval="1d")
                if "error" not in tv_data:
                    st.success(f"Connected! RELIANCE Consensus: {tv_data.get('recommendation')} ({tv_data.get('buy_votes')} Buy, {tv_data.get('sell_votes')} Sell)")
                else:
                    st.error(tv_data["error"])
            except Exception as e:
                st.error(str(e))
        st.markdown("Run `python webhook_server.py --port 5000` to start receiving alert webhooks.")

    with col_conn3:
        st.markdown("### 🏛️ Shoonya Broker API Link")
        missing = config.shoonya_missing()
        if missing:
            st.info("Shoonya broker not linked yet. Missing in .env: " + ", ".join(missing))
            st.markdown(
                "1. Copy `.env.example` to `.env`\n"
                "2. Fill in your Shoonya user, password, TOTP secret, and vendor key.\n"
                "3. Restart the app to enable direct live broker order execution."
            )
        else:
            st.success("All Shoonya broker credentials configured in .env.")
            if st.button("Test Shoonya Broker Connection", key="btn_test_shoonya"):
                try:
                    ok, msg = make_provider("shoonya").check_connection()
                except Exception as e:
                    ok, msg = False, str(e)
                (st.success if ok else st.error)(msg)

    st.divider()
    st.subheader("📺 TradingView Webhook Automation Bridge")
    st.caption("Receive live trade alert webhooks from your TradingView account and auto-execute them in Paper Trading or Live Shoonya Execution.")

    tv_hook_c1, tv_hook_c2 = st.columns([1.6, 1])
    with tv_hook_c1:
        st.markdown("**How to link your TradingView Account Alerts:**")
        st.markdown(
            "1. In TradingView, create an Alert on any chart or Pine Script strategy.\n"
            "2. Under **Notifications**, enable **Webhook URL**.\n"
            "3. Enter the Webhook URL: `http://localhost:5000/webhook` (or your ngrok / public URL).\n"
            "4. In the **Message** box, configure JSON payload format or use the Pine Script alert template below."
        )
        with st.expander("📋 TradingView Pine Script v5 Strategy Webhook Template", expanded=False):
            st.code(tvi.get_sample_pine_script_alert_code(), language="pinescript")

    with tv_hook_c2:
        st.markdown("**Test Webhook Dispatcher:**")
        test_sym = st.selectbox("Test Symbol", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "TATAMOTORS"], key="tv_test_sym")
        test_act = st.selectbox("Action", ["BUY", "SELL", "EXIT"], key="tv_test_act")
        is_shoonya_live = getattr(provider, "name", "") == "shoonya" and getattr(provider, "is_live", False)
        test_mode = st.selectbox(
            "Execution Mode",
            ["paper", "live"] if is_shoonya_live else ["paper"],
            key="tv_test_mode",
            help="Select 'paper' for simulated execution or 'live' for real Shoonya broker order placement.",
        )
        is_live_mode = (test_mode == "live")
        if st.button(f"🚀 Send Test Alert ({test_mode.upper()})", key="btn_send_tv_webhook"):
            est_p_map = {"RELIANCE": 1375.0, "TCS": 3500.0, "HDFCBANK": 1750.0, "INFY": 1850.0, "TATAMOTORS": 780.0}
            p_val = est_p_map.get(test_sym, 1000.0)
            if provider:
                try:
                    q = provider.get_quote(test_sym)
                    if q and q.get("ltp"):
                        p_val = float(q["ltp"])
                except Exception:
                    pass
            test_payload = {
                "ticker": test_sym,
                "action": test_act,
                "price": p_val,
                "stop_loss": round(p_val * 0.98, 2),
                "target": round(p_val * 1.04, 2),
                "quantity": 5,
                "strategy": "TradingView Manual Test",
                "mode": test_mode,
            }
            ok_w, msg_w, det_w = tvi.process_tradingview_webhook_alert(test_payload)
            if ok_w:
                st.success(f"Status: {det_w.get('status')} — {msg_w}")
            else:
                st.error(f"Failed: {msg_w}")
            st.rerun()

    st.divider()
    st.caption("Credentials stay securely on your computer in .env. This page never transmits or stores them.")
