import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
import os

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from groq import Groq


# --------------------------------------------------
# GROQ GENAI CONFIGURATION
# --------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-120b"

ADVISORY_DISCLAIMER = (
    "This AI-generated information is for educational purposes only. "
    "It is not a medical diagnosis or treatment recommendation. "
    "Please consult a qualified healthcare professional."
)


def get_groq_client():

    key = os.environ.get("GROQ_API_KEY")

    # Read from Streamlit Cloud Secrets if not in environment
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None

    if not key:
        return None

    return Groq(api_key=key)


# --------------------------------------------------
# CLINICAL AI ADVISORY
# --------------------------------------------------

def generate_clinical_advisory(
    subtype, confidence, patient_features, shap_features
):

    client = get_groq_client()

    if client is None:
        return "Groq API key not found. Please check Streamlit Cloud Secrets."

    prompt = f"""
    You are an educational PCOS health information assistant.

    Clinical model predicted subtype: {subtype}
    Prediction confidence: {confidence:.2f}

    Patient input features:
    {patient_features}

    Top SHAP contributing features:
    {shap_features}

    Explain the model prediction cautiously.
    SHAP values indicate model contributions, not medical causes.

    Provide:
    1. Simple explanation of the model result.
    2. General dietary guidance.
    3. Physical activity suggestions.
    4. Sleep and stress management suggestions.
    5. Tests or follow-up topics to discuss with a doctor.

    Do not diagnose, prescribe medicines or supplements, suggest doses,
    or infer unmeasured hormone levels.

    Keep the response clear and concise.
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


# --------------------------------------------------
# ULTRASOUND AI ADVISORY
# --------------------------------------------------

def generate_ultrasound_advisory(
    prediction, pcos_probability, comparison_probability
):

    client = get_groq_client()

    if client is None:
        return "Groq API key not found. Please check Streamlit Cloud Secrets."

    prompt = f"""
    You are an educational PCOS health information assistant.

    CNN ultrasound classification: {prediction}
    PCO probability: {pcos_probability:.2%}
    Comparison Class probability: {comparison_probability:.2%}

    Explain the image classification in simple language.

    Provide:
    1. Explanation of the CNN result.
    2. General information about ultrasound findings.
    3. Limitations of AI-based ultrasound classification.
    4. Questions to discuss with a qualified healthcare professional.

    Do not diagnose PCOS or infer clinical subtypes or hormone levels.
    Do not prescribe medicines, supplements or doses.

    Keep the response clear and concise.
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PCOS Insight",
    page_icon="♀️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #F8F5FF, #F1ECFA);
}

[data-testid="stSidebar"] {
    background-color: #EAE2F7;
}

h1, h2, h3 {
    color: #604582 !important;
}

p, label {
    color: #493B5D;
}

[data-testid="stMetric"],
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E5D9F3;
    border-radius: 14px;
    padding: 15px;
}

/* Lavender action buttons */

.stButton > button,
.stFormSubmitButton > button {
    background-color: #8064A8 !important;
    color: #000000 !important;
    border: 1px solid #8064A8 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #674B91 !important;
    color: #000000 !important;
    border: 1px solid #674B91 !important;
}

/* Keep PCOS Insight in its lavender light theme */

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F8F5FF, #F1ECFA) !important;
}

h1, h2, h3,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
label {
    color: #604582 !important;
}

/* Hide Streamlit toolbar and theme selector */

[data-testid="stToolbar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
}

/* Make normal app text readable */

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #493B5D !important;
}

/* Clinical prediction - red */

.prediction-red {
    color: #D32F2F !important;
    font-size: 23px;
    font-weight: 700;
}

/* Ultrasound prediction - red or green */

.prediction-green {
    color: #218838 !important;
    font-size: 23px;
    font-weight: 700;
}

/* Top navigation styling — scoped to the main navigation only */
.st-key-main_navigation [role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 5px !important;
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.st-key-main_navigation [role="radiogroup"] label {
    background: transparent !important;
    border: 0 !important;
    border-radius: 7px !important;
    padding: 8px 10px !important;
    color: #3F3151 !important;
    font-size: 13px;
    font-weight: 650;
}
.st-key-main_navigation [role="radiogroup"] label:has(input:checked) {
    background: #8060B2 !important;
}
.st-key-main_navigation [role="radiogroup"] label:has(input:checked) p {
    color: white !important;
}

/* Advisory content */

.advisory-text {
    color: #493B5D !important;
    font-size: 16px;
}

/* GenAI advisory readable text */

.stMarkdown p,
.stMarkdown li,
.stMarkdown strong,
.stMarkdown em,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #493B5D !important;
}


/* PCOS Insight brand header */
.brand-row {
    display:flex;
    align-items:center;
    gap:12px;
    padding: 2px 0 10px 0;
}
.brand-mark {
    width:48px;
    height:48px;
    flex:0 0 48px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:transparent;
    box-shadow:none;
    border-radius:0;
}
.brand-name {
    color:#302640;
    font-size:25px;
    font-weight:800;
    letter-spacing:.2px;
    line-height:1.05;
}
.brand-subtitle {
    color:#715C89;
    font-size:10px;
    letter-spacing:1.8px;
    text-transform:uppercase;
    margin-top:4px;
}
.st-key-nav_home button, .st-key-nav_clinical_analysis button,
.st-key-nav_ultrasound_analysis button, .st-key-nav_about_us button,
.st-key-nav_contact_us button {
    background: transparent !important;
    border: none !important;
    color: #493B5D !important;
    padding: 5px 2px !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    font-size: 13px !important;
}
.st-key-nav_home button:hover, .st-key-nav_clinical_analysis button:hover,
.st-key-nav_ultrasound_analysis button:hover, .st-key-nav_about_us button:hover,
.st-key-nav_contact_us button:hover {
    color: #7652A4 !important;
    text-decoration: underline !important;
}
.st-key-home_hero {
    background: linear-gradient(112deg, #B69ADD 0%, #C6B0E5 58%, #D4C4EB 100%);
    padding: 20px 30px 0 30px;
    margin-top: 2px;
    overflow: hidden;
}
.st-key-home_hero [data-testid="stHorizontalBlock"] {
    align-items: center;
}
.st-key-home_hero .brand-name {
    color: #302640 !important;
}
.st-key-home_hero .brand-subtitle {
    color: #5B4771 !important;
}
.st-key-home_hero div[role="radiogroup"] label {
    color: #302640 !important;
}
.st-key-home_hero .hero-panel {
    background: transparent !important;
    padding: 34px 20px 32px 10px;
    min-height: 390px;
}
.st-key-home_hero [data-testid="stImage"] img {
    border-radius: 0 !important;
}
.hero-panel {
    background:linear-gradient(112deg,#B69ADD 0%,#C6B0E5 58%,#D4C4EB 100%);
    border-radius:0;
    padding:34px 34px 28px 34px;
    min-height:390px;
    overflow:hidden;
}
.hero-kicker {
    color:#5D437E;
    font-size:12px;
    font-weight:800;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:14px;
}
.hero-title {
    color:#30243F !important;
    font-size:clamp(30px,3.1vw,46px);
    line-height:1.13;
    font-weight:750;
    margin:0 0 16px 0;
}
.hero-copy {
    color:#40334F !important;
    font-size:16px;
    line-height:1.55;
    max-width:560px;
}
.hero-note {
    color:#5B4771 !important;
    font-size:12px;
    margin-top:14px;
}
@media (max-width: 900px) {
  .brand-name { font-size:21px; }
  .st-key-home_hero { padding: 14px 14px 0 14px; }
  .st-key-home_hero .hero-panel { min-height:unset; padding:22px 10px; }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PCOS INSIGHT AUTHENTICATION (SUPABASE)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from supabase import create_client


def get_supabase_client():
    """Create a Supabase client using Streamlit Cloud Secrets."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        return create_client(url, key)
    except Exception:
        return None


def show_auth_page():
    """Show the lavender login / create-account screen."""
    st.markdown("""
    <style>
      [data-testid="stSidebar"] {background:#EAE2F7 !important;}
      .auth-card {
        background:rgba(255,255,255,.88); border:1px solid #E5D9F3;
        border-radius:22px; padding:30px 34px 25px 34px;
        box-shadow:0 12px 35px rgba(92,65,128,.10);
      }
      .auth-title {text-align:center;color:#604582;font-size:30px;
        font-weight:800;margin-bottom:3px;}
      .auth-subtitle {text-align:center;color:#806A99;font-size:14px;
        margin-bottom:22px;}
      .auth-link {text-align:center;color:#7652A4;font-size:14px;}
      .auth-side-title {font-size:20px;font-weight:800;color:#604582;}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        logo_path = os.path.join(BASE_DIR, "pcos_insight_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=72)
        st.markdown('<div class="auth-side-title">PCOS Insight</div>', unsafe_allow_html=True)
        st.caption("Women's health research")
        st.markdown("---")
        st.markdown(
            "*“Small steps towards better understanding create meaningful change.”*"
        )
        st.caption("AI-powered PCOS research platform")

    supabase = get_supabase_client()
    if supabase is None:
        st.error(
            "Login is not configured yet. Add SUPABASE_URL and "
            "SUPABASE_ANON_KEY to your Streamlit Cloud Secrets."
        )
        st.info("The setup instructions are included with the updated app files.")
        st.stop()

    left, center, right = st.columns([1, 1.65, 1])
    with center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">PCOS Insight</div>', unsafe_allow_html=True)

        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "login"

        if st.session_state.auth_mode == "login":
            st.markdown('<div class="auth-subtitle">Login to PCOS Insight</div>', unsafe_allow_html=True)
            with st.form("pcos_login_form"):
                email = st.text_input("Email", placeholder="Enter your email address")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_clicked = st.form_submit_button("Login", use_container_width=True)

            if login_clicked:
                if not email.strip() or not password:
                    st.warning("Please enter both your email and password.")
                else:
                    try:
                        response = supabase.auth.sign_in_with_password(
                            {"email": email.strip(), "password": password}
                        )
                        if response.user:
                            st.session_state.authenticated = True
                            st.session_state.user_email = email.strip()
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error("Login failed. Please check your details and try again.")
                    except Exception as exc:
                        message = str(exc).lower()
                        if "invalid login credentials" in message or "invalid password" in message:
                            st.error("Incorrect email or password. Please try again.")
                        elif "email not confirmed" in message:
                            st.error("Please verify your email using the confirmation link sent to your inbox.")
                        else:
                            st.error("Unable to log in. Please check your email and password, then try again.")

            st.markdown('<div class="auth-link">New user?</div>', unsafe_allow_html=True)
            if st.button("Create Account", key="go_to_signup", use_container_width=True, type="tertiary"):
                st.session_state.auth_mode = "signup"
                st.rerun()

        else:
            st.markdown('<div class="auth-subtitle">Create your PCOS Insight account</div>', unsafe_allow_html=True)
            with st.form("pcos_signup_form"):
                new_email = st.text_input("Email", placeholder="Enter your email address")
                new_password = st.text_input("Create password", type="password", placeholder="At least 8 characters")
                confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password")
                signup_clicked = st.form_submit_button("Create Account", use_container_width=True)

            if signup_clicked:
                if not new_email.strip() or not new_password or not confirm_password:
                    st.warning("Please complete all fields.")
                elif new_password != confirm_password:
                    st.error("The passwords do not match. Please try again.")
                elif len(new_password) < 8:
                    st.error("Please choose a password with at least 8 characters.")
                else:
                    try:
                        response = supabase.auth.sign_up(
                            {"email": new_email.strip(), "password": new_password}
                        )
                        if response.session:
                            st.session_state.authenticated = True
                            st.session_state.user_email = new_email.strip()
                            st.session_state.auth_mode = "login"
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.success("Account created. Please check your email and click the confirmation link, then log in.")
                            st.session_state.auth_mode = "login"
                    except Exception:
                        st.error("We couldn't create this account. The email may already be registered, or the details may be invalid.")

            st.markdown('<div class="auth-link">Already have an account?</div>', unsafe_allow_html=True)
            if st.button("Back to Login", key="back_to_login", use_container_width=True, type="tertiary"):
                st.session_state.auth_mode = "login"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# Require authentication before displaying any application page.
if not st.session_state.get("authenticated", False):
    show_auth_page()
    st.stop()

with st.sidebar:
    st.caption(f"Signed in as {st.session_state.get('user_email', '')}")
    if st.button("Log out", use_container_width=True):
        try:
            auth_client = get_supabase_client()
            if auth_client is not None:
                auth_client.auth.sign_out()
        except Exception:
            pass
        st.session_state.authenticated = False
        st.session_state.pop("user_email", None)
        st.session_state.auth_mode = "login"
        st.rerun()


# --------------------------------------------------
# MODEL AND DATA PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# XGBoost model is in the same folder as app.py
CLINICAL_MODEL_PATH = os.path.join(
    BASE_DIR, "xgboost_model.json"
)

# CNN model is downloaded from Hugging Face
ULTRASOUND_MODEL_PATH = "/tmp/cnn_model.keras"


FEATURES = [
    "Hair_Growth", "Blood_Sugar", "Hormonal_Disruption", "AMH",
    "Androgen_Score", "HOMA_IR", "FSH_LH_raw", "Total_Follicles",
    "Follicle_L", "LH", "FSH", "LH_FSH_Ratio", "Cycle_Length",
    "Fast_Food", "TSH", "BP_Diastolic", "BMI", "Progesterone",
    "Marriage_Years", "RR"
]

CLASSES = [
    "Insulin Resistant",
    "Inflammatory",
    "Adrenal",
    "Post-Pill"
]


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_clinical_model():

    model = xgb.Booster()

    model.load_model(CLINICAL_MODEL_PATH)

    return model


@st.cache_resource
def load_ultrasound_model():

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="MahaHuggy/pcos-insight-models",
        filename="cnn_model.keras"
    )

    return tf.keras.models.load_model(model_path)


# --------------------------------------------------
# BRANDED HEADER + TOP NAVIGATION
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

NAV_PAGES = [
    "Home",
    "Clinical Analysis",
    "Ultrasound Analysis",
    "About Us",
    "Contact Us"
]

# Keep the navigation widget and the active page in sync. This prevents
# the radio tab from reverting to an old selection after a card/back click.
if "main_navigation" not in st.session_state:
    st.session_state.main_navigation = st.session_state.page


def set_page(page_name):
    st.session_state.page = page_name
    st.session_state.main_navigation = page_name


def sync_page_from_navigation():
    st.session_state.page = st.session_state.main_navigation


def render_navigation():
    header_left, header_right = st.columns(
        [1.0, 2.8], vertical_alignment="center"
    )

    with header_left:
        logo_col, brand_text_col = st.columns(
            [1, 3.8], vertical_alignment="center"
        )

        with logo_col:
            logo_path = os.path.join(BASE_DIR, "pcos_insight_logo.png")
            if os.path.exists(logo_path):
                st.image(logo_path, width=48)

        with brand_text_col:
            st.markdown("""
            <div class="brand-name">PCOS Insight</div>
            <div class="brand-subtitle">Women's health research</div>
            """, unsafe_allow_html=True)

    with header_right:
        nav_slots = st.columns([1.0, 0.12, 1.65, 0.12, 1.8, 0.12, 1.0, 0.12, 1.15])
        nav_items = [
            (0, "Home"), (2, "Clinical Analysis"),
            (4, "Ultrasound Analysis"), (6, "About Us"), (8, "Contact Us")
        ]
        for slot, label in nav_items:
            with nav_slots[slot]:
                st.button(
                    label,
                    key=f"nav_{label.lower().replace(' ', '_')}",
                    type="tertiary",
                    use_container_width=True,
                    on_click=set_page,
                    args=(label,)
                )
        for slot in [1, 3, 5, 7]:
            with nav_slots[slot]:
                st.markdown("<div style='text-align:center;color:#8064A8;padding-top:8px;'>|</div>", unsafe_allow_html=True)

    return st.session_state.page


# ==================================================
# HOME PAGE — HEADER AND HERO IN ONE SECTION
# ==================================================

if st.session_state.page == "Home":

    with st.container(key="home_hero"):
        page = render_navigation()

        if page == "Home":
            hero_text, hero_image = st.columns(
                [1.0, 1.05], gap="small", vertical_alignment="center"
            )

            with hero_text:
                st.markdown("""
                <div class="hero-panel">
                  <div class="hero-kicker">AI-powered PCOS research</div>
                  <div class="hero-title">
                    A smarter perspective<br>
                    on women's health
                  </div>
                  <div class="hero-copy">
                    Explore PCOS-related clinical biomarkers and ultrasound
                    image patterns through machine learning, deep learning,
                    and explainable AI.
                  </div>
                  <div class="hero-note">
                    Research-focused insights • Multimodal analysis • Explainable results
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with hero_image:
                hero_path = os.path.join(BASE_DIR, "hero_women.png")
                if os.path.exists(hero_path):
                    st.image(hero_path, use_container_width=True)
                else:
                    st.warning(
                        "Add hero_women.png beside app.py to display the hero photograph."
                    )

    page = st.session_state.page

    if page == "Home":
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        st.markdown("## Explore PCOS Insight")
        clinical_card, ultrasound_card = st.columns(2, gap="large")

        with clinical_card:
            st.markdown("""
            <div style="background:#F0E8FA;padding:22px;border-radius:16px;
                        border:1px solid #E2D5F2;min-height:170px;">
              <div style="font-size:25px;color:#7652A4;font-weight:800;">Clinical Analysis</div>
              <p style="color:#51445F;">Explore clinical biomarkers, four-class subtype predictions,
              and SHAP-based model explanations.</p>
            </div>
            """, unsafe_allow_html=True)
            st.button(
                "Open Clinical Analysis",
                key="home_clinical",
                use_container_width=True,
                on_click=set_page,
                args=("Clinical Analysis",)
            )

        with ultrasound_card:
            st.markdown("""
            <div style="background:#F0E8FA;padding:22px;border-radius:16px;
                        border:1px solid #E2D5F2;min-height:170px;">
              <div style="font-size:25px;color:#7652A4;font-weight:800;">Ultrasound Analysis</div>
              <p style="color:#51445F;">Upload an ovarian ultrasound image and view the CNN-based
              PCO or Comparison Class prediction.</p>
            </div>
            """, unsafe_allow_html=True)
            st.button(
                "Open Ultrasound Analysis",
                key="home_ultrasound",
                use_container_width=True,
                on_click=set_page,
                args=("Ultrasound Analysis",)
            )

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.markdown("## Research at a glance")

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown("""
            <div style="background:#F0E8FA;padding:22px 18px;border-radius:16px;
            text-align:center;min-height:125px;">
              <div style="font-size:34px;font-weight:800;color:#7656A3;">4</div>
              <div style="font-weight:700;color:#443451;">Clinical subtypes</div>
              <div style="font-size:12px;color:#766783;">XGBoost classification</div>
            </div>
            """, unsafe_allow_html=True)
        with stat2:
            st.markdown("""
            <div style="background:#F0E8FA;padding:22px 18px;border-radius:16px;
            text-align:center;min-height:125px;">
              <div style="font-size:34px;font-weight:800;color:#7656A3;">2</div>
              <div style="font-weight:700;color:#443451;">Ultrasound classes</div>
              <div style="font-size:12px;color:#766783;">CNN image classification</div>
            </div>
            """, unsafe_allow_html=True)
        with stat3:
            st.markdown("""
            <div style="background:#F0E8FA;padding:22px 18px;border-radius:16px;
            text-align:center;min-height:125px;">
              <div style="font-size:34px;font-weight:800;color:#7656A3;">XAI</div>
              <div style="font-weight:700;color:#443451;">Explainable insights</div>
              <div style="font-size:12px;color:#766783;">SHAP-based interpretation</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("## About PCOS Insight")
        st.write(
            "PCOS Insight is a research-focused platform that explores "
            "clinical biomarkers and ultrasound image patterns using "
            "machine learning, deep learning and explainable AI."
        )
        st.info(
            "For educational and research purposes only. It is not a "
            "medical diagnostic tool."
        )

else:
    page = render_navigation()
    st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)


if page == "About Us":

    st.title("About PCOS Insight")
    st.write(
        "PCOS Insight is a research-oriented application designed to "
        "explore PCOS-related clinical information and ovarian ultrasound "
        "images using machine-learning and deep-learning techniques."
    )

    st.subheader("Project Overview")
    st.write(
        "The application contains two independent analysis modules. "
        "The Clinical Analysis module uses an XGBoost model for four-class "
        "PCOS subtype classification and provides patient-specific SHAP "
        "explanations. The Ultrasound Analysis module uses a CNN to classify "
        "uploaded images into PCO and Comparison Class."
    )

    st.subheader("Research Objectives")
    st.markdown("""
    - Explore clinical biomarker patterns for PCOS subtype classification.
    - Analyse ovarian ultrasound images using deep learning.
    - Provide model explainability for clinical predictions.
    - Present cautious, educational AI-generated result summaries.
    """)

    st.subheader("Technologies Used")
    st.markdown("""
    - Python and Streamlit
    - XGBoost for clinical classification
    - TensorFlow / Keras for ultrasound image classification
    - SHAP for clinical model explainability
    - Groq API for educational AI advisories
    """)

    st.info(
        "The clinical and ultrasound models perform separate tasks. "
        "The ultrasound model does not predict the four clinical subtypes."
    )


# ==================================================
# CONTACT US
# ==================================================

elif page == "Contact Us":

    st.title("Contact Us")
    st.write("For questions or feedback about the PCOS Insight research project, contact our team.")

    st.subheader("Project Team")
    st.markdown("""
    **Team Members**

    - Madhu Parameswari Ganesan
    - Mahalakshmi M
    - Dhivyalakshmi S

    **Department of CSE**  
    SASTRA Deemed University  
    Thanjavur, India

    **Project Guide**  
    Dr. Anushiadevi R  
    Department of ICT  
    SASTRA Deemed University  
    Thanjavur, India
    """)

    st.subheader("Contact Email")
    st.markdown("[triotech1203@gmail.com](mailto:triotech1203@gmail.com)")


# ==================================================
# CLINICAL ANALYSIS
# ==================================================

elif page == "Clinical Analysis":

    st.button("← Back to Home", on_click=set_page, args=("Home",))

    st.title("Clinical Analysis")
    st.write("Enter the clinical measurements below.")

    with st.form("clinical_form"):
        col1, col2 = st.columns(2)

        with col1:
            hair = st.selectbox("Hair Growth", ["No", "Yes"])
            blood_sugar = st.number_input("Blood Sugar", min_value=0.0, value=100.0)
            hormonal = st.number_input("Hormonal Disruption", min_value=0.0, value=1.0)
            amh = st.number_input("AMH", min_value=0.0, value=3.0)
            androgen = st.number_input("Androgen Score", min_value=0.0, value=1.0)
            homa = st.number_input("HOMA_IR", min_value=0.0, value=1.0)
            fsh_lh_raw = st.number_input("FSH_LH_raw", min_value=0.0, value=1.0)
            total_follicles = st.number_input("Total Follicles", min_value=0.0, value=20.0)
            follicle_l = st.number_input("Follicle_L", min_value=0.0, value=10.0)
            lh = st.number_input("LH", min_value=0.0, value=8.0)

        with col2:
            fsh = st.number_input("FSH", min_value=0.0, value=5.0)
            lh_fsh_ratio = st.number_input("LH_FSH_Ratio", min_value=0.0, value=1.6)
            cycle_length = st.number_input("Cycle Length", min_value=0.0, value=35.0)
            fast_food = st.selectbox("Fast Food Consumption", ["No", "Yes"])
            tsh = st.number_input("TSH", min_value=0.0, value=2.5)
            bp = st.number_input("BP Diastolic", min_value=0.0, value=80.0)
            bmi = st.number_input("BMI", min_value=0.0, value=25.0)
            progesterone = st.number_input("Progesterone", min_value=0.0, value=1.0)
            marriage_years = st.number_input("Marriage Years", min_value=0.0, value=0.0)
            rr = st.number_input("RR", min_value=0.0, value=18.0)

        submitted = st.form_submit_button("Predict Subtype", use_container_width=True)

    if submitted:
        values = {
            "Hair_Growth": 1 if hair == "Yes" else 0,
            "Blood_Sugar": blood_sugar,
            "Hormonal_Disruption": hormonal,
            "AMH": amh,
            "Androgen_Score": androgen,
            "HOMA_IR": homa,
            "FSH_LH_raw": fsh_lh_raw,
            "Total_Follicles": total_follicles,
            "Follicle_L": follicle_l,
            "LH": lh,
            "FSH": fsh,
            "LH_FSH_Ratio": lh_fsh_ratio,
            "Cycle_Length": cycle_length,
            "Fast_Food": 1 if fast_food == "Yes" else 0,
            "TSH": tsh,
            "BP_Diastolic": bp,
            "BMI": bmi,
            "Progesterone": progesterone,
            "Marriage_Years": marriage_years,
            "RR": rr
        }

        try:
            with st.spinner("Analyzing clinical measurements..."):
                model = load_clinical_model()
                input_df = pd.DataFrame([values], columns=FEATURES)
                probabilities = model.predict(xgb.DMatrix(input_df))

                if probabilities.ndim == 1:
                    probabilities = probabilities.reshape(1, -1)

                probabilities = probabilities[0]
                predicted_index = int(np.argmax(probabilities))

                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = np.asarray(explainer.shap_values(input_df))
                patient_shap = shap_values[0, :, predicted_index]

                shap_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "SHAP Value": patient_shap
                })
                shap_df["Absolute SHAP"] = np.abs(shap_df["SHAP Value"])
                shap_df = shap_df.sort_values(
                    "Absolute SHAP", ascending=False
                ).head(5)

                advisory = None
                advisory_error = None
                try:
                    advisory = generate_clinical_advisory(
                        subtype=CLASSES[predicted_index],
                        confidence=float(probabilities[predicted_index]),
                        patient_features=values,
                        shap_features=shap_df[
                            ["Feature", "SHAP Value"]
                        ].to_dict(orient="records")
                    )
                except Exception as e:
                    advisory_error = str(e)

                st.session_state["clinical_result"] = {
                    "values": values,
                    "probabilities": probabilities.tolist(),
                    "predicted_index": predicted_index,
                    "shap_df": shap_df,
                    "advisory": advisory,
                    "advisory_error": advisory_error
                }

        except Exception as e:
            st.error(f"Clinical prediction error: {e}")
            st.session_state.pop("clinical_result", None)

    clinical_result = st.session_state.get("clinical_result")

    if clinical_result:
        probabilities = np.asarray(clinical_result["probabilities"])
        predicted_index = clinical_result["predicted_index"]
        probability_df = pd.DataFrame({
            "Subtype": CLASSES,
            "Probability": probabilities
        })
        probability_df["Probability (%)"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.markdown(
            f"""
            <div style='color:#D32F2F !important;font-size:23px;font-weight:700;'>
            Predicted PCOS Subtype: {CLASSES[predicted_index]}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric(
            "Prediction Confidence",
            f"{probabilities[predicted_index] * 100:.2f}%"
        )

        st.subheader("Subtype Probability Distribution")
        st.bar_chart(
            probability_df.set_index("Subtype")["Probability (%)"]
        )
        st.dataframe(
            probability_df[["Subtype", "Probability (%)"]],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("🔍 SHAP Explainability", expanded=False):
            st.caption(
                "Top five features contributing to this model prediction. "
                "SHAP values represent model contributions, not medical causes."
            )
            shap_df = clinical_result["shap_df"]
            st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])
            st.dataframe(
                shap_df[["Feature", "SHAP Value"]],
                use_container_width=True,
                hide_index=True
            )

        with st.expander("💡 Clinical AI Advisory", expanded=False):
            if clinical_result["advisory"]:
                st.markdown(clinical_result["advisory"])
                st.info(ADVISORY_DISCLAIMER)
            else:
                st.warning(
                    "The clinical advisory could not be generated. "
                    f"{clinical_result['advisory_error'] or ''}"
                )

        with st.expander("📋 Input Summary", expanded=False):
            input_summary = pd.DataFrame(
                list(clinical_result["values"].items()),
                columns=["Clinical Feature", "Entered Value"]
            )
            st.dataframe(
                input_summary,
                use_container_width=True,
                hide_index=True
            )


# ==================================================
# ULTRASOUND ANALYSIS
# ==================================================

elif page == "Ultrasound Analysis":

    st.button("← Back to Home", on_click=set_page, args=("Home",))

    st.title("Ultrasound Analysis")
    st.write("Upload an ultrasound image for model-based classification.")

    uploaded_file = st.file_uploader(
        "Upload ultrasound image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(
            image,
            caption="Uploaded Ultrasound Image",
            width=350
        )

        if st.button("Analyze Ultrasound", use_container_width=True):
            try:
                with st.spinner("Analyzing ultrasound image..."):
                    model = load_ultrasound_model()
                    resized_image = image.resize(
                        (224, 224),
                        Image.Resampling.BICUBIC
                    )
                    image_array = np.array(
                        resized_image.convert("RGB"),
                        dtype=np.float32
                    )
                    image_array = np.expand_dims(image_array, axis=0)
                    image_array = preprocess_input(image_array)

                    output = float(
                        model.predict(image_array, verbose=0)[0][0]
                    )

                    # CNN output represents P(PCO).
                    pcos_probability = output
                    comparison_probability = 1.0 - output
                    prediction = (
                        "PCO"
                        if pcos_probability >= comparison_probability
                        else "Comparison Class"
                    )

                    advisory = None
                    advisory_error = None
                    try:
                        advisory = generate_ultrasound_advisory(
                            prediction=prediction,
                            pcos_probability=pcos_probability,
                            comparison_probability=comparison_probability
                        )
                    except Exception as e:
                        advisory_error = str(e)

                    st.session_state["ultrasound_result"] = {
                        "prediction": prediction,
                        "pcos_probability": pcos_probability,
                        "comparison_probability": comparison_probability,
                        "advisory": advisory,
                        "advisory_error": advisory_error,
                        "filename": uploaded_file.name,
                        "image_size": image.size
                    }

            except Exception as e:
                st.error(f"Ultrasound prediction error: {e}")
                st.session_state.pop("ultrasound_result", None)

    ultrasound_result = st.session_state.get("ultrasound_result")

    if ultrasound_result:
        # Show the result only for the currently uploaded image.
        current_filename = uploaded_file.name if uploaded_file is not None else None
        if current_filename == ultrasound_result["filename"]:
            prediction = ultrasound_result["prediction"]
            prediction_color = "#D32F2F" if prediction == "PCO" else "#218838"

            st.markdown(
                f"""
                <div style='color:{prediction_color} !important;
                font-size:23px;font-weight:700;'>
                Predicted class: {prediction}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction Confidence",
                f"{max(ultrasound_result['pcos_probability'], ultrasound_result['comparison_probability']) * 100:.2f}%"
            )

            probability_df = pd.DataFrame({
                "Class": ["PCO", "Comparison Class"],
                "Probability (%)": [
                    round(ultrasound_result["pcos_probability"] * 100, 2),
                    round(ultrasound_result["comparison_probability"] * 100, 2)
                ]
            })

            st.subheader("Prediction Probabilities")
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            with st.expander("📊 Probability Analysis", expanded=False):
                st.bar_chart(
                    probability_df.set_index("Class")["Probability (%)"]
                )

            with st.expander("💡 Ultrasound AI Advisory", expanded=False):
                if ultrasound_result["advisory"]:
                    st.markdown(ultrasound_result["advisory"])
                    st.info(ADVISORY_DISCLAIMER)
                else:
                    st.warning(
                        "The ultrasound advisory could not be generated. "
                        f"{ultrasound_result['advisory_error'] or ''}"
                    )

            with st.expander("🖼️ Image Details", expanded=False):
                st.write(f"**File name:** {ultrasound_result['filename']}")
                st.write(
                    f"**Original image dimensions:** "
                    f"{ultrasound_result['image_size'][0]} × "
                    f"{ultrasound_result['image_size'][1]} pixels"
                )
                st.caption(
                    "The CNN classifies the uploaded image as PCO or "
                    "Comparison Class only. It does not predict the four "
                    "clinical PCOS subtypes."
                )
        else:
            st.info("Upload and analyze an image to view its prediction.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.caption(
    "Research prototype | Not for medical diagnosis"
)
