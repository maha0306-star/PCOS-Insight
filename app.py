import os

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from groq import Groq
from supabase import create_client


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PCOS Insight",
    page_icon="♀️",
    layout="wide",
    initial_sidebar_state="auto"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROQ_MODEL = "openai/gpt-oss-120b"
ADVISORY_DISCLAIMER = (
    "This AI-generated information is for educational purposes only. "
    "It is not a medical diagnosis or treatment recommendation. "
    "Please consult a qualified healthcare professional."
)

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

CNN_CLASSES = ["Normal", "PCO", "Dominant_Follicle"]

CLINICAL_MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.json")


# ==================================================
# RESPONSIVE LAVENDER THEME
# ==================================================

st.markdown("""
<style>
/* Remove the entire empty Streamlit top header strip, not just its icons */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Keep Streamlit's native sidebar toggle available as the mobile hamburger menu. */
header[data-testid="stHeader"] {
    display: block !important;
    visibility: visible !important;
    height: 2.75rem !important;
    min-height: 2.75rem !important;
    background: transparent !important;
    z-index: 1001 !important;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: none !important;
}
/* Make the native sidebar open/close control visible and easy to tap. */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebar"] button[kind="header"] {
    color: #604582 !important;
    min-width: 40px !important;
    min-height: 40px !important;
}

/* Prevent a leftover dark band where the header used to be */
[data-testid="stAppViewContainer"] > .main {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ---------- Global page ---------- */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FAF8FE 0%, #F1ECFA 100%) !important;
}
.block-container {
    max-width: 1480px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    padding-left: clamp(1.5rem, 4vw, 4rem);
    padding-right: clamp(1.5rem, 4vw, 4rem);
}
h1, h2, h3, h4 {
    color: #604582 !important;
}
p, li, label, [data-testid="stMarkdownContainer"] {
    color: #493B5D;
}
[data-testid="stSidebar"] {
    background: #EAE2F7 !important;
}

/* ---------- Inputs: login, signup and analysis ---------- */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {
    color: #30243D !important;
    background-color: #FFFFFF !important;
    border: 1px solid #CDBBE2 !important;
    border-radius: 10px !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #786B86 !important;
    opacity: 1 !important;
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border: 1.5px solid #8064A8 !important;
    box-shadow: 0 0 0 2px rgba(128, 100, 168, .13) !important;
}
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stFileUploader label,
.stRadio label {
    color: #493B5D !important;
    font-weight: 600 !important;
}
/* Selectbox selected value and dropdown text */
[data-baseweb="select"] *,
[role="listbox"] *,
[role="option"] {
    color: #30243D !important;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stFormSubmitButton > button {
    background: #8064A8 !important;
    color: #FFFFFF !important;
    border: 1px solid #8064A8 !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
    min-height: 42px;
    transition: background .15s ease, transform .15s ease;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #674B91 !important;
    border-color: #674B91 !important;
    color: #FFFFFF !important;
}
.stButton > button:focus,
.stFormSubmitButton > button:focus {
    box-shadow: 0 0 0 3px rgba(128,100,168,.22) !important;
}

/* ---------- Cards and expanders ---------- */
[data-testid="stMetric"],
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E5D9F3;
    border-radius: 14px;
}
[data-testid="stMetric"] {
    padding: 14px;
}
[data-testid="stExpander"] summary {
    color: #604582 !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] details {
    border-radius: 14px !important;
}

/* ---------- Brand and hero ---------- */
.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0 12px;
}
.brand-name {
    color: #302640;
    font-size: 25px;
    font-weight: 800;
    line-height: 1.1;
}
.brand-subtitle {
    color: #715C89;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}
.hero-panel {
    background: linear-gradient(112deg, #B69ADD 0%, #C6B0E5 58%, #D4C4EB 100%);
    border-radius: 20px;
    padding: clamp(22px, 4vw, 42px);
    min-height: 300px;
}
.hero-kicker {
    color: #5D437E;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    color: #30243F !important;
    font-size: clamp(29px, 4vw, 46px);
    line-height: 1.13;
    font-weight: 800;
    margin: 0 0 16px;
}
.hero-copy {
    color: #40334F !important;
    font-size: 16px;
    line-height: 1.6;
}
.hero-note {
    color: #5B4771 !important;
    font-size: 12px;
    margin-top: 14px;
}
.info-card {
    background: #F0E8FA;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #E2D5F2;
    min-height: 150px;
}
.info-card h3 {
    margin-top: 0;
    color: #7652A4 !important;
}
.stat-card {
    background: #F0E8FA;
    padding: 20px 14px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #E2D5F2;
    min-height: 118px;
}
.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #7656A3;
}
.stat-label {
    font-weight: 700;
    color: #443451;
}
.stat-note {
    font-size: 12px;
    color: #766783;
}

/* ---------- Navigation ---------- */
.nav-wrap {
    background: rgba(255,255,255,.72);
    border: 1px solid #E5D9F3;
    border-radius: 16px;
    padding: 12px 20px 4px;
    margin: 8px 0 26px;
    box-shadow: 0 5px 18px rgba(83, 59, 112, .055);
}
.st-key-main_navigation [role="radiogroup"] {
    display: flex;
    flex-wrap: wrap !important;
    gap: 10px !important;
    align-items: center;
    justify-content: center !important;
    width: 100%;
}
.st-key-main_navigation [role="radiogroup"] label {
    background: transparent !important;
    border: 0 !important;
    border-radius: 9px !important;
    padding: 9px 17px !important;
    color: #493B5D !important;
    font-size: 14px !important;
    font-weight: 650 !important;
}
.st-key-main_navigation [role="radiogroup"] label:has(input:checked) {
    background: #8060B2 !important;
    color: #FFFFFF !important;
}
.st-key-main_navigation [role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important;
}

/*
   Responsive dashboard navigation:
   - Desktop: account/branding sidebar + horizontal top navigation.
   - Mobile/tablet: top navigation becomes a hamburger-triggered Streamlit drawer;
     the drawer contains branding, signed-in email, quote, page links and logout.
*/
@media (min-width: 769px) {
    [data-testid="stSidebar"] [data-testid="stRadio"] { display: none !important; }
    .nav-wrap { display: block !important; }
}
@media (max-width: 768px) {
    .nav-wrap { display: none !important; }
    header[data-testid="stHeader"] {
        height: 3rem !important;
        min-height: 3rem !important;
    }
    .block-container {
        padding-top: .75rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .brand-name { font-size: 22px !important; }
    .hero-panel { min-height: auto !important; }
    [data-testid="stSidebar"] {
        min-width: min(84vw, 340px) !important;
        max-width: min(84vw, 340px) !important;
        background: #EAE2F7 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        min-height: 42px !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        min-height: 46px !important;
    }
}

/* ---------- Result headings and footer ---------- */
.prediction-title {
    font-size: clamp(20px, 3vw, 25px);
    font-weight: 800;
    color: #604582;
    overflow-wrap: anywhere;
}
.app-footer {
    border-top: 1px solid #E2D5F2;
    margin-top: 32px;
    padding-top: 14px;
    color: #786B86;
    font-size: 12px;
    text-align: center;
}

/* ---------- Remove unwanted dark outline/shadow around login card ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: none !important;
    outline: none !important;
    border-color: #E5D9F3 !important;
}
[data-testid="stForm"] {
    box-shadow: none !important;
}

/* ---------- Login / signup split-screen design ---------- */
.auth-brand-panel {
    /* A true vertical branding sidebar, not a floating rounded card */
    min-height: 560px;
    height: 100%;
    width: 100%;
    box-sizing: border-box;
    padding: 48px 34px;
    border-radius: 0 !important;
    background: linear-gradient(180deg, #EAE2F7 0%, #D8C8EE 100%);
    border: 0 !important;
    border-right: 1px solid #D1BFE8;
    box-shadow: none !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    overflow-wrap: normal;
}
.auth-lock {
    width: 62px;
    height: 62px;
    border-radius: 18px;
    background: rgba(255,255,255,.78);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    margin-bottom: 30px;
    box-shadow: 0 8px 22px rgba(82,55,115,.10);
}
.auth-brand-logo {
    display: block;
    width: 64px;
    height: 64px;
    object-fit: contain;
    border-radius: 18px;
    margin: 0 0 18px 0;
    background: rgba(255,255,255,.78);
    padding: 8px;
    box-sizing: border-box;
    box-shadow: 0 8px 22px rgba(82,55,115,.10);
}
.auth-login-lock {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    background: #F0E8FA;
    border: 1px solid #E1D4F0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 27px;
    margin: 4px auto 10px;
}
.auth-brand-title {
    color: #30243F;
    font-size: clamp(28px, 2.2vw, 36px);
    font-weight: 850;
    line-height: 1.12;
    letter-spacing: -.8px;
    max-width: 100%;
    white-space: normal;
}
.auth-brand-subtitle {
    color: #674C89;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.2px;
    margin-top: 12px;
}
.auth-brand-divider {
    width: 68px;
    height: 4px;
    border-radius: 8px;
    background: #8064A8;
    margin: 24px 0 18px;
}
.auth-quote {
    color: #453454;
    font-size: clamp(16px, 1.25vw, 19px);
    font-weight: 550;
    line-height: 1.55;
    max-width: 360px;
}
.auth-brand-foot {
    color: #6D5785;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-top: 30px;
}
.auth-form-title {
    text-align: center;
    margin-top: 12px;
}
.auth-form-subtitle {
    text-align: center;
    color: #786B86 !important;
    margin-bottom: 28px;
}
@media (min-width: 1100px) {
    .auth-brand-panel { min-height: 500px; }
}
@media (max-width: 768px) {
    .auth-brand-panel {
        min-height: 250px;
        padding: 26px 24px;
        border-radius: 0 !important;
        border-right: 0;
        border-bottom: 1px solid #D1BFE8;
        margin-bottom: 10px;
    }
    .auth-lock { width: 48px; height: 48px; font-size: 24px; margin-bottom: 16px; }
    .auth-brand-title { font-size: 32px; }
    .auth-brand-subtitle { font-size: 9px; letter-spacing: 1.5px; }
    .auth-brand-divider { margin: 18px 0 14px; }
    .auth-quote { font-size: 16px; }
    .auth-brand-foot { margin-top: 20px; font-size: 9px; }
}

/* ---------- Desktop and large-screen refinement ---------- */
@media (min-width: 1100px) {
    .block-container {
        max-width: 1480px !important;
        padding-left: clamp(2.5rem, 4.2vw, 4.5rem) !important;
        padding-right: clamp(2.5rem, 4.2vw, 4.5rem) !important;
    }
    .brand-name { font-size: 28px; }
    .brand-subtitle { font-size: 11px; }
    .hero-panel {
        min-height: 350px;
        padding: 46px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hero-title { font-size: clamp(38px, 3.2vw, 52px); }
    .hero-copy { max-width: 620px; font-size: 17px; }
    .info-card { min-height: 170px; padding: 28px; }
    .stat-card { min-height: 132px; padding: 24px 18px; }
    .nav-wrap { padding-left: 24px; padding-right: 24px; }
}

/* ---------- Small screens ---------- */
@media (max-width: 768px) {
    .block-container {
        max-width: 100% !important;
        padding-top: .65rem !important;
        padding-left: .85rem !important;
        padding-right: .85rem !important;
    }
    .brand-name { font-size: 21px; }
    .brand-subtitle { font-size: 9px; letter-spacing: 1px; }
    .hero-panel {
        min-height: unset;
        padding: 22px 19px;
        border-radius: 16px;
    }
    .hero-title { font-size: 30px; }
    .hero-copy { font-size: 14px; }
    .info-card { min-height: unset; padding: 18px; }
    .stat-card { min-height: 105px; padding: 15px 10px; }
    .stat-number { font-size: 27px; }
    .st-key-main_navigation [role="radiogroup"] {
        gap: 2px !important;
        justify-content: flex-start !important;
    }
    .st-key-main_navigation [role="radiogroup"] label {
        padding: 6px 7px !important;
        font-size: 12px !important;
    }
    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        max-width: 100%;
        overflow-x: auto;
    }
    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 46px;
    }
}
@media (max-width: 480px) {
    .brand-name { font-size: 19px; }
    .hero-title { font-size: 27px; }
    .hero-kicker { font-size: 10px; letter-spacing: 1.4px; }
    .st-key-main_navigation [role="radiogroup"] label {
        padding: 5px 6px !important;
        font-size: 11px !important;
    }
}

/* Hide decorative Streamlit chrome; retain app content */
[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ==================================================
# GROQ CLIENT AND ADVISORIES
# ==================================================

def get_groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return Groq(api_key=key) if key else None


def generate_clinical_advisory(subtype, confidence, patient_features, shap_features):
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

Explain the model prediction cautiously. SHAP values indicate model
contributions, not medical causes.

Provide:
1. Simple explanation of the model result.
2. General dietary guidance.
3. Physical activity suggestions.
4. Sleep and stress management suggestions.
5. Tests or follow-up topics to discuss with a doctor.

Do not diagnose, prescribe medicines or supplements, suggest doses,
or infer unmeasured hormone levels. Keep the response clear and concise.
"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content


def generate_ultrasound_advisory(prediction, class_probabilities):
    client = get_groq_client()
    if client is None:
        return "Groq API key not found. Please check Streamlit Cloud Secrets."

    probability_text = "\\n".join(
        f"{class_name}: {probability:.2%}"
        for class_name, probability in zip(CNN_CLASSES, class_probabilities)
    )

    prompt = f"""
You are an educational PCOS health information assistant.

CNN ultrasound classification: {prediction}

Class probabilities:
{probability_text}

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
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content


# ==================================================
# SUPABASE AUTHENTICATION
# ==================================================

def get_supabase_client():
    try:
        return create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    except Exception:
        return None


def show_auth_page():
    supabase = get_supabase_client()
    if supabase is None:
        st.error(
            "Login is not configured. Add SUPABASE_URL and "
            "SUPABASE_ANON_KEY to Streamlit Cloud Secrets."
        )
        st.stop()

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    # Two-panel authentication layout: brand panel on the left, form on the right.
    # Keep the branding panel compact; give the login form more room.
    brand_col, form_col = st.columns([0.85, 1.55], gap="large")

    with brand_col:
        # Reuse the exact logo asset displayed in the Home page navigation.
        # Embed it as a data URI so it renders reliably inside the branded panel.
        import base64
        logo_path = os.path.join(BASE_DIR, "pcos_insight_logo.png")
        logo_html = ""
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as logo_file:
                logo_data = base64.b64encode(logo_file.read()).decode("utf-8")
            logo_html = (
                f'<img class="auth-brand-logo" '
                f'src="data:image/png;base64,{logo_data}" '
                f'alt="PCOS Insight logo">'
            )

        st.markdown(
            f"""
            <div class="auth-brand-panel">
                {logo_html}
                <div class="auth-brand-title">PCOS Insight</div>
                <div class="auth-brand-subtitle">WOMEN'S HEALTH RESEARCH</div>
                <div class="auth-brand-divider"></div>
                <div class="auth-quote">
                    “Turning clinical data and ultrasound insights into
                    clearer, explainable research.”
                </div>
                <div class="auth-brand-foot">MULTIMODAL AI · EXPLAINABLE INSIGHTS</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with form_col:
        with st.container(border=False):
            st.markdown(
                "<div class='auth-login-lock'>🔒</div><h2 class='auth-form-title'>Welcome back</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p class='auth-form-subtitle'>Sign in to continue to PCOS Insight</p>",
                unsafe_allow_html=True
            )

            if st.session_state.auth_mode == "login":
                with st.form("pcos_login_form"):
                    email = st.text_input(
                        "Email", placeholder="Enter your email address", key="login_email"
                    )
                    password = st.text_input(
                        "Password", type="password",
                        placeholder="Enter your password", key="login_password"
                    )
                    login_clicked = st.form_submit_button(
                        "🔒  Login securely", use_container_width=True
                    )

                if login_clicked:
                    if not email.strip() or not password:
                        st.warning("Please enter both your email and password.")
                    else:
                        try:
                            response = supabase.auth.sign_in_with_password({
                                "email": email.strip(), "password": password
                            })
                            if response.user:
                                st.session_state.authenticated = True
                                st.session_state.user_email = email.strip()
                                st.rerun()
                            else:
                                st.error("Login failed. Please check your details.")
                        except Exception as exc:
                            message = str(exc).lower()
                            if "email not confirmed" in message:
                                st.error(
                                    "Please verify your email using the confirmation "
                                    "link sent to your inbox."
                                )
                            elif "invalid login credentials" in message:
                                st.error("Incorrect email or password.")
                            else:
                                st.error(
                                    "Unable to log in. Please check your details and try again."
                                )

                st.caption("New to PCOS Insight?")
                if st.button("Create Account", use_container_width=True, key="go_to_signup"):
                    st.session_state.auth_mode = "signup"
                    st.rerun()

            else:
                with st.form("pcos_signup_form"):
                    new_email = st.text_input(
                        "Email", placeholder="Enter your email address", key="signup_email"
                    )
                    new_password = st.text_input(
                        "Create password", type="password",
                        placeholder="At least 8 characters", key="signup_password"
                    )
                    confirm_password = st.text_input(
                        "Confirm password", type="password",
                        placeholder="Re-enter your password", key="confirm_password"
                    )
                    signup_clicked = st.form_submit_button(
                        "Create Account", use_container_width=True
                    )

                if signup_clicked:
                    if not new_email.strip() or not new_password or not confirm_password:
                        st.warning("Please complete all fields.")
                    elif new_password != confirm_password:
                        st.error("The passwords do not match.")
                    elif len(new_password) < 8:
                        st.error("Please choose a password with at least 8 characters.")
                    else:
                        try:
                            response = supabase.auth.sign_up({
                                "email": new_email.strip(), "password": new_password
                            })
                            if response.session:
                                st.session_state.authenticated = True
                                st.session_state.user_email = new_email.strip()
                                st.success("Account created successfully.")
                                st.rerun()
                            else:
                                st.success(
                                    "Account created. Please check your email, "
                                    "verify your account, and then log in."
                                )
                                st.session_state.auth_mode = "login"
                        except Exception:
                            st.error(
                                "We couldn't create this account. The email may "
                                "already be registered, or the details may be invalid."
                            )

                st.caption("Already have an account?")
                if st.button("← Back to Login", use_container_width=True, key="back_to_login"):
                    st.session_state.auth_mode = "login"
                    st.rerun()

    st.markdown(
        "<div class='app-footer'>PCOS Insight · Research and education only · "
        "Not for medical diagnosis</div>",
        unsafe_allow_html=True
    )


if not st.session_state.get("authenticated", False):
    show_auth_page()
    st.stop()


# ==================================================
# MODEL LOADING
# ==================================================

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


# ==================================================
# NAVIGATION
# ==================================================

NAV_PAGES = [
    "Home",
    "Clinical Analysis",
    "Ultrasound Analysis",
    "About Us",
    "Contact Us"
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "main_navigation" not in st.session_state:
    st.session_state.main_navigation = st.session_state.page
if "mobile_navigation" not in st.session_state:
    st.session_state.mobile_navigation = st.session_state.page


def set_page(page_name):
    st.session_state.page = page_name
    st.session_state.main_navigation = page_name
    st.session_state.mobile_navigation = page_name


def logout_user():
    try:
        auth_client = get_supabase_client()
        if auth_client is not None:
            auth_client.auth.sign_out()
    except Exception:
        pass
    st.session_state.authenticated = False
    st.session_state.pop("user_email", None)
    st.session_state.pop("clinical_result", None)
    st.session_state.pop("ultrasound_result", None)
    st.session_state.auth_mode = "login"
    st.rerun()


def render_navigation():
    # Brand header is shared by desktop and mobile layouts.
    logo_path = os.path.join(BASE_DIR, "pcos_insight_logo.png")
    logo_col, title_col = st.columns([0.09, 0.91], vertical_alignment="center")
    with logo_col:
        if os.path.exists(logo_path):
            st.image(logo_path, width=46)
        else:
            st.markdown("### ♀️")
    with title_col:
        st.markdown(
            "<div class='brand-name'>PCOS Insight</div>"
            "<div class='brand-subtitle'>Women's health research</div>",
            unsafe_allow_html=True
        )

    # Desktop account panel; on narrow screens Streamlit turns this into the
    # hamburger-triggered drawer containing account details and page navigation.
    with st.sidebar:
        if os.path.exists(logo_path):
            st.image(logo_path, width=76)
        else:
            st.markdown("## ♀️")
        st.markdown("## PCOS Insight")
        st.caption("Women's health research")
        st.divider()
        st.caption("SIGNED IN AS")
        st.markdown(f"**{st.session_state.get('user_email', '')}**")
        st.markdown(
            "<div style='background:#F7F2FC;border-left:4px solid #8064A8;"
            "padding:13px;border-radius:8px;color:#493B5D;font-style:italic;'>"
            "Knowledge empowers better health.</div>",
            unsafe_allow_html=True
        )
        st.markdown(" ")
        st.radio(
            "Dashboard navigation",
            NAV_PAGES,
            key="mobile_navigation",
            label_visibility="collapsed",
            on_change=lambda: set_page(st.session_state.mobile_navigation)
        )
        st.divider()
        if st.button("Log out", key="sidebar_logout", use_container_width=True):
            logout_user()

    # Horizontal top navigation is shown on desktop and moved into the drawer on mobile.
    st.markdown("<div class='nav-wrap'>", unsafe_allow_html=True)
    selected = st.radio(
        "Main navigation",
        NAV_PAGES,
        key="main_navigation",
        horizontal=True,
        label_visibility="collapsed",
        on_change=lambda: set_page(st.session_state.main_navigation)
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.page = selected
    return selected


page = render_navigation()


# ==================================================
# HOME
# ==================================================

if page == "Home":
    hero_text, hero_image = st.columns([1.12, 0.88], gap="large",
                                       vertical_alignment="center")
    with hero_text:
        st.markdown("""
        <div class="hero-panel">
          <div class="hero-kicker">AI-powered PCOS research</div>
          <div class="hero-title">A smarter perspective<br>on women's health</div>
          <div class="hero-copy">
            Explore PCOS-related clinical biomarkers and ultrasound image
            patterns through machine learning, deep learning, and explainable AI.
          </div>
          <div class="hero-note">
            Research-focused insights · Multimodal analysis · Explainable results
          </div>
        </div>
        """, unsafe_allow_html=True)
    with hero_image:
        hero_path = os.path.join(BASE_DIR, "hero_women.png")
        if os.path.exists(hero_path):
            st.image(hero_path, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-card">
              <h3>Clinical + Ultrasound Intelligence</h3>
              <p>Two independent AI analysis pathways for research exploration.</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.header("Explore PCOS Insight")
    card1, card2 = st.columns(2, gap="large")
    with card1:
        st.markdown("""
        <div class="info-card">
          <h3>Clinical Analysis</h3>
          <p>Explore clinical biomarkers, four-class subtype predictions,
          probability scores and SHAP-based model explanations.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open Clinical Analysis", key="home_clinical",
                  use_container_width=True,
                  on_click=set_page, args=("Clinical Analysis",))
    with card2:
        st.markdown("""
        <div class="info-card">
          <h3>Ultrasound Analysis</h3>
          <p>Upload an ovarian ultrasound image and view the CNN-based
          Normal, PCO or Dominant_Follicle prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open Ultrasound Analysis", key="home_ultrasound",
                  use_container_width=True,
                  on_click=set_page, args=("Ultrasound Analysis",))

    st.write("")
    st.header("Research at a glance")
    s1, s2, s3 = st.columns(3)
    for col, number, label, note in [
        (s1, "4", "Clinical subtypes", "XGBoost classification"),
        (s2, "3", "Ultrasound classes", "CNN image classification"),
        (s3, "XAI", "Explainable insights", "SHAP-based interpretation")
    ]:
        with col:
            st.markdown(
                f"<div class='stat-card'><div class='stat-number'>{number}</div>"
                f"<div class='stat-label'>{label}</div>"
                f"<div class='stat-note'>{note}</div></div>",
                unsafe_allow_html=True
            )

    st.write("")
    st.header("About PCOS Insight")
    st.write(
        "PCOS Insight is a research-focused platform that explores clinical "
        "biomarkers and ultrasound image patterns using machine learning, "
        "deep learning and explainable AI."
    )
    st.info(
        "For educational and research purposes only. It is not a medical "
        "diagnostic tool."
    )


# ==================================================
# ABOUT US
# ==================================================

elif page == "About Us":
    st.title("About PCOS Insight")
    st.write(
        "PCOS Insight is a research-oriented application designed to explore "
        "PCOS-related clinical information and ovarian ultrasound images "
        "using machine-learning and deep-learning techniques."
    )
    st.subheader("Project Overview")
    st.write(
        "The application contains two independent analysis modules. The "
        "Clinical Analysis module uses an XGBoost model for four-class PCOS "
        "subtype classification and provides patient-specific SHAP explanations. "
        "The Ultrasound Analysis module uses a CNN to classify uploaded images "
        "into Normal, PCO and Dominant_Follicle."
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
    - Supabase for user authentication
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
    st.write(
        "For questions or feedback about the PCOS Insight research project, "
        "contact our team."
    )
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
    st.title("Clinical Analysis")
    st.write("Enter the clinical measurements below.")

    with st.form("clinical_form"):
        st.subheader("Clinical and Biomarker Information")
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
                except Exception as exc:
                    advisory_error = str(exc)

                st.session_state["clinical_result"] = {
                    "values": values,
                    "probabilities": probabilities.tolist(),
                    "predicted_index": predicted_index,
                    "shap_df": shap_df,
                    "advisory": advisory,
                    "advisory_error": advisory_error
                }
        except Exception as exc:
            st.error(f"Clinical prediction error: {exc}")
            st.session_state.pop("clinical_result", None)

    clinical_result = st.session_state.get("clinical_result")
    if clinical_result:
        probabilities = np.asarray(clinical_result["probabilities"])
        predicted_index = clinical_result["predicted_index"]

        st.markdown(
            f"<div class='prediction-title'>Predicted PCOS Subtype: "
            f"{CLASSES[predicted_index]}</div>",
            unsafe_allow_html=True
        )
        st.metric(
            "Prediction Confidence",
            f"{probabilities[predicted_index] * 100:.2f}%"
        )

        probability_df = pd.DataFrame({
            "Subtype": CLASSES,
            "Probability (%)": (probabilities * 100).round(2)
        })

        with st.expander("📊 Subtype Probability Analysis", expanded=True):
            st.bar_chart(
                probability_df.set_index("Subtype")["Probability (%)"],
                color="#9B7BC4"
            )
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

        with st.expander("🔍 SHAP Explainability", expanded=False):
            st.caption(
                "Top five features contributing to this model prediction. "
                "SHAP values represent model contributions, not medical causes."
            )
            shap_df = clinical_result["shap_df"]
            st.bar_chart(
                shap_df.set_index("Feature")["SHAP Value"],
                color="#9B7BC4"
            )
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
    st.title("Ultrasound Analysis")
    st.write("Upload an ovarian ultrasound image for model-based classification.")

    uploaded_file = st.file_uploader(
        "Upload ultrasound image",
        type=["jpg", "jpeg", "png"],
        help="Choose a JPG, JPEG or PNG ultrasound image."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(
            image,
            caption="Uploaded Ultrasound Image",
            use_container_width=True
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

                    class_probabilities = np.asarray(
                        model.predict(image_array, verbose=0)[0],
                        dtype=float
                    )

                    if class_probabilities.size != len(CNN_CLASSES):
                        raise ValueError(
                            "The CNN returned "
                            f"{class_probabilities.size} outputs, but "
                            f"{len(CNN_CLASSES)} classes are configured. "
                            "Please verify the model and class order."
                        )

                    predicted_index = int(np.argmax(class_probabilities))
                    prediction = CNN_CLASSES[predicted_index]
                    confidence = float(class_probabilities[predicted_index])

                    advisory = None
                    advisory_error = None
                    try:
                        advisory = generate_ultrasound_advisory(
                            prediction, class_probabilities
                        )
                    except Exception as exc:
                        advisory_error = str(exc)

                    st.session_state["ultrasound_result"] = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "class_probabilities": class_probabilities.tolist(),
                        "advisory": advisory,
                        "advisory_error": advisory_error,
                        "filename": uploaded_file.name,
                        "image_size": image.size
                    }
            except Exception as exc:
                st.error(f"Ultrasound prediction error: {exc}")
                st.session_state.pop("ultrasound_result", None)

    ultrasound_result = st.session_state.get("ultrasound_result")
    if ultrasound_result:
        current_filename = (
            uploaded_file.name if uploaded_file is not None else None
        )
        if current_filename == ultrasound_result["filename"]:
            prediction = ultrasound_result["prediction"]
            st.markdown(
                f"<div class='prediction-title'>Predicted Class: "
                f"{prediction}</div>",
                unsafe_allow_html=True
            )

            confidence = ultrasound_result["confidence"]
            st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")

            probability_df = pd.DataFrame({
                "Class": CNN_CLASSES,
                "Probability (%)": (
                    np.asarray(ultrasound_result["class_probabilities"]) * 100
                ).round(2)
            })

            with st.expander("📊 Probability Analysis", expanded=True):
                st.bar_chart(
                    probability_df.set_index("Class")["Probability (%)"],
                    color="#9B7BC4"
                )
                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
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
                    "The CNN classifies the uploaded image as Normal, PCO or "
                    "Dominant_Follicle. It does not predict the four "
                    "clinical PCOS subtypes."
                )
        else:
            st.info("Upload and analyze an image to view its prediction.")


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    "<div class='app-footer'>PCOS Insight · Research and educational use only · "
    "Not a medical diagnostic tool</div>",
    unsafe_allow_html=True
)
