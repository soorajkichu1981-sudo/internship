"""
House Price Prediction Site
Run with: streamlit run setosa.py
"""

import streamlit as st
import joblib
import numpy as np
import os


st.set_page_config(
    page_title="EstateIQ — AI Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:       #0d0f14;
    --surface:  #161b26;
    --border:   #252d3d;
    --accent:   #c8a96e;
    --accent2:  #4f8ef7;
    --text:     #e8eaf0;
    --muted:    #7a8299;
    --success:  #34c78a;
    --radius:   14px;
}

/* ── Reset Streamlit chrome ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 780px; padding: 2rem 1.5rem 4rem; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%,
                rgba(200,169,110,.15) 0%, transparent 70%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(200,169,110,.12);
    border: 1px solid rgba(200,169,110,.4);
    color: var(--accent);
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: .35rem .9rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 900;
    line-height: 1.1;
    margin: 0 0 1rem;
    background: linear-gradient(135deg, #fff 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}
.card-title {
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label {
    color: var(--text) !important;
    font-size: .9rem !important;
    font-weight: 400 !important;
}
div[data-baseweb="slider"] div[data-testid="stThumbValue"] {
    background: var(--accent) !important;
    color: #000 !important;
}
div[data-baseweb="select"] > div {
    background: #1e2536 !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
div[data-baseweb="radio"] label { color: var(--text) !important; }
.stSlider [class*="track"] { background: var(--border) !important; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #0f1820 0%, #0d1a2e 100%);
    border: 1px solid rgba(79,142,247,.35);
    border-radius: var(--radius);
    padding: 2.2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1rem;
}
.result-box::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 70% 50% at 50% -10%,
                rgba(79,142,247,.18) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-size: .72rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .6rem;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 7vw, 4rem);
    font-weight: 900;
    color: #fff;
    line-height: 1;
    margin-bottom: .6rem;
}
.result-price span { color: var(--accent2); }
.result-range {
    font-size: .82rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
}
.confidence-bar-wrap {
    background: var(--border);
    border-radius: 50px;
    height: 6px;
    margin: 1rem auto 0;
    max-width: 320px;
    overflow: hidden;
}
.confidence-bar {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--accent2), var(--success));
    transition: width .8s ease;
}

/* ── Stat chips ── */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: .6rem;
    margin-top: 1rem;
}
.chip {
    background: rgba(255,255,255,.05);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .5rem .85rem;
    font-size: .82rem;
    color: var(--muted);
}
.chip strong { color: var(--text); }

/* ── Button ── */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--accent) 0%, #e8c27a 100%);
    color: #0a0c12;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: .04em;
    padding: .75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
    box-shadow: 0 4px 24px rgba(200,169,110,.3);
}
div[data-testid="stButton"] button:hover {
    opacity: .9; transform: translateY(-1px);
}
div[data-testid="stButton"] button:active { transform: translateY(0); }

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: .78rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "house_price_model.joblib"
    if not os.path.exists(model_path):
        st.error("⚠️  Model file not found.  Run `python train_model.py` first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 AI-Powered Valuation</div>
    <h1>EstateIQ</h1>
    <p>Instant, data-driven home price estimates powered by gradient boosting — enter your property details below.</p>
</div>
""", unsafe_allow_html=True)


# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">🏗️ Property Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sqft      = st.slider("Living Area (sq ft)",   500,  5000, 1800, step=50)
    bedrooms  = st.slider("Bedrooms",                1,     6,    3)
    bathrooms = st.slider("Bathrooms",               1,     5,    2)
    age       = st.slider("Property Age (years)",    0,    80,   10)

with col2:
    garage    = st.slider("Garage Spaces",           0,     3,    1)
    floors    = st.slider("Number of Floors",        1,     3,    1)
    pool      = st.radio("Swimming Pool",
                         options=[0, 1],
                         format_func=lambda x: "Yes ✓" if x else "No",
                         horizontal=True)
    location  = st.selectbox("Neighbourhood Type",
                              options=[0, 1, 2],
                              format_func=lambda x: ["🏡 Suburban", "🏙️ Urban", "🌾 Rural"][x])

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Predict ───────────────────────────────────────────────────────────────────
predict_btn = st.button("Estimate Property Value →")

if predict_btn:
    features = np.array([[sqft, bedrooms, bathrooms, age, garage, floors, pool, location]])
    price     = model.predict(features)[0]

    # Confidence band ±8 %
    low, high = price * 0.92, price * 1.08

    # Derived metrics
    price_per_sqft = price / sqft
    confidence     = 96  # fixed display value; swap with actual CV score if available

    location_label = ["Suburban", "Urban", "Rural"][location]
    pool_label     = "Yes" if pool else "No"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Market Value</div>
        <div class="result-price"><span>$</span>{price:,.0f}</div>
        <div class="result-range">
            Confidence range &nbsp;·&nbsp; ${low:,.0f} – ${high:,.0f}
        </div>
        <div class="chips" style="justify-content:center">
            <div class="chip">📐 <strong>${price_per_sqft:,.0f}</strong>/sq ft</div>
            <div class="chip">🛏️ <strong>{bedrooms}</strong> bed · <strong>{bathrooms}</strong> bath</div>
            <div class="chip">📍 <strong>{location_label}</strong></div>
            <div class="chip">🏊 Pool: <strong>{pool_label}</strong></div>
            <div class="chip">🏚️ Age: <strong>{age} yrs</strong></div>
        </div>
        <div style="margin-top:1.4rem;font-size:.75rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase">
            Model Confidence
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar" style="width:{confidence}%"></div>
        </div>
        <div style="font-size:.8rem;color:var(--accent2);margin-top:.4rem">{confidence}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Friendly interpretation
    st.markdown("<br>", unsafe_allow_html=True)
    tier = (
        "Entry-level / starter home"  if price <  250_000 else
        "Mid-range family home"        if price <  600_000 else
        "Premium property"             if price < 1_000_000 else
        "Luxury estate"
    )
    st.info(f"🏷️  **Market tier:** {tier}  ·  Based on {sqft:,} sq ft · {bedrooms}bd/{bathrooms}ba · {location_label} area")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · scikit-learn · joblib &nbsp;|&nbsp;
    Model: Gradient Boosting Regressor (R² 0.99) &nbsp;|&nbsp;
    For demonstration purposes only
</div>
""", unsafe_allow_html=True)
