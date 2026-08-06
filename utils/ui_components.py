import streamlit as st


def apply_custom_theme():
    """Injects high-end glassmorphic dark theme CSS into Streamlit."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.98) 0%, rgba(10, 15, 30, 1) 90%);
        color: #f8fafc;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }

    .metric-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .status-badge-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.25);
    }

    .status-badge-real {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.25);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #38bdf8 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Renders sleek top header."""
    title_style = (
        "font-size: 2.8rem; font-weight: 800; background: "
        "linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); "
        "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
        "margin-bottom: 8px;"
    )
    desc_style = (
        "font-size: 1.1rem; color: #94a3b8; max-width: 750px; "
        "margin: 0 auto 20px auto;"
    )
    header_html = f"""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="{title_style}">
            🛡️ Deepfake Forensic Intelligence Suite
        </h1>
        <p style="{desc_style}">
            Enterprise Multi-Modal Forensic Platform powered by Transformer
            Neural Networks, 2D FFT Spectral Analysis, Error Level Analysis,
            & EXIF Metadata Verification.
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_verdict_card(verdict: str, confidence: float, details: str = ""):
    """Renders glowing verdict summary banner."""
    is_fake = verdict.lower() == "fake"
    badge_class = "status-badge-fake" if is_fake else "status-badge-real"
    icon = "⚠️ MANIPULATED / FAKE" if is_fake else "✅ AUTHENTIC / REAL"
    border_color = "#ef4444" if is_fake else "#22c55e"
    bg_gradient = (
        "rgba(239, 68, 68, 0.08)" if is_fake else "rgba(34, 197, 94, 0.08)"
    )

    card_html = f"""
    <div style="background: {bg_gradient}; border: 1.5px solid {border_color}; border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 24px; box-shadow: 0 4px 20px {bg_gradient};">
        <div style="margin-bottom: 8px;">
            <span class="{badge_class}">{icon}</span>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-top: 8px;">
            Confidence Score: <span style="color: {border_color};">{confidence * 100:.1f}%</span>
        </div>
        <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 6px;">
            {details}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
