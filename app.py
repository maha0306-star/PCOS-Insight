
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

/* Top navigation styling */

div[role="radiogroup"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.78);
    padding: 10px 14px;
    border: 1px solid #E5D9F3;
    border-radius: 14px;
}

div[role="radiogroup"] label {
    background: #F3EDFA;
    border: 1px solid #E5D9F3;
    border-radius: 9px;
    padding: 7px 13px;
}

.nav-divider {
    border-bottom: 1px solid #E5D9F3;
    margin: 4px 0 22px 0;
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
div[role="radiogroup"] {
    background:transparent !important;
    border:0 !important;
    padding:0 !important;
    justify-content:flex-end;
    gap:3px !important;
}
div[role="radiogroup"] label {
    background:transparent !important;
    border:0 !important;
    border-radius:0 !important;
    padding:9px 10px !important;
    color:#3F3151 !important;
    font-size:13px;
    font-weight:650;
}
div[role="radiogroup"] label:has(input:checked) {
    background:#8060B2 !important;
    border-radius:7px !important;
}
div[role="radiogroup"] label:has(input:checked) p {
    color:white !important;
}


@media (max-width: 900px) {
  .brand-name { font-size:21px; }
  .hero-panel { min-height:unset; padding:26px 22px; border-radius:16px; }
}

</style>
""", unsafe_allow_html=True)


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

header_left, header_right = st.columns([1.15, 2.4], vertical_alignment="center")

with header_left:
    st.markdown("""
    <div class="brand-mark">
    <img src="pcos_insight_logo.png"
         alt="PCOS Insight Logo"
         style="width:48px;height:48px;object-fit:contain;">
</div>
      <div>
        <div class="brand-name">PCOS Insight</div>
        <div class="brand-subtitle">Women's health research</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with header_right:
    current_index = (
        NAV_PAGES.index(st.session_state.page)
        if st.session_state.page in NAV_PAGES
        else 0
    )
    selected_page = st.radio(
        "Main navigation",
        NAV_PAGES,
        index=current_index,
        horizontal=True,
        label_visibility="collapsed"
    )

st.session_state.page = selected_page
page = st.session_state.page
st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    hero_text, hero_image = st.columns([1.05, 1.0], gap="large")

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
            st.markdown("""
            <div style="
              min-height:390px; height:100%;
              border-radius:0 18px 18px 0;
              background:linear-gradient(135deg,#D8C8EB,#B99ADD);
              display:flex;align-items:center;justify-content:center;
              color:#604582;text-align:center;padding:25px;">
              <div>
                <div style="font-size:54px;">♡</div>
                <b>Women's health, supported by research</b>
                <p style="font-size:13px;">
                  Add hero_women.png beside app.py to display the hero photograph.
                </p>
              </div>
            </div>
            """, unsafe_allow_html=True)

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
          <div style="font-size:34px;font-weight:800;color:#7656A3;">AI</div>
          <div style="font-weight:700;color:#443451;">Explainable insights</div>
          <div style="font-size:12px;color:#766783;">SHAP + educational advisories</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("## Explore the analysis modules")
    st.write(
        "Choose a research module to explore clinical biomarker patterns "
        "or ultrasound image classification."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style="background:#FFFFFF;padding:23px;border-radius:16px;
        border:1px solid #E5D9F3;min-height:160px;">
          <h3 style="color:#604582;">Clinical Analysis</h3>
          <p style="color:#594B6D;">
            Explore clinical measurements, four-class subtype predictions,
            probability scores, and patient-specific SHAP explanations.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Clinical Analysis", use_container_width=True):
            st.session_state.page = "Clinical Analysis"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background:#FFFFFF;padding:23px;border-radius:16px;
        border:1px solid #E5D9F3;min-height:160px;">
          <h3 style="color:#604582;">Ultrasound Analysis</h3>
          <p style="color:#594B6D;">
            Explore CNN-based ultrasound image classification into
            PCO and Comparison Class, with probability analysis.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Ultrasound Analysis", use_container_width=True):
            st.session_state.page = "Ultrasound Analysis"
            st.rerun()

    st.markdown("""
    <div style="background:#FFFFFF;padding:20px 22px;border-radius:15px;
    border-left:5px solid #A78BCB;margin-top:24px;">
      <h4 style="color:#604582;">About this research</h4>
      <p style="color:#594B6D;">
        PCOS Insight brings together clinical machine learning,
        ultrasound deep learning, and explainable AI in a research
        and educational prototype.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Research and educational prototype only. "
        "AI predictions are not medical diagnoses."
    )


# ==================================================
# ABOUT US
# ==================================================

elif page == "About Us":

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
    st.markdown(
        "- Explore clinical biomarker patterns for PCOS subtype classification.\\n"
        "- Analyse ovarian ultrasound images using deep learning.\\n"
        "- Provide model explainability for clinical predictions.\\n"
        "- Present cautious, educational AI-generated result summaries."
    )

    st.subheader("Technologies Used")
    st.markdown(
        "- Python and Streamlit\\n"
        "- XGBoost for clinical classification\\n"
        "- TensorFlow / Keras for ultrasound image classification\\n"
        "- SHAP for clinical model explainability\\n"
        "- Groq API for educational AI advisories"
    )

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
        "For questions about this research prototype, collaboration, "
        "or technical feedback, please contact the project team through "
        "your institution or the official project contact channel."
    )

    st.subheader("Project Information")
    st.write("Project: PCOS Insight")
    st.write(
        "Research area: Multimodal PCOS analysis using clinical biomarkers, "
        "ultrasound features, and explainable AI."
    )

    st.subheader("Project Team")
    st.write(
        "Team member names, department, institution, project guide, "
        "and official contact email can be added here after confirming "
        "the details intended for public display."
    )


# ==================================================
# CLINICAL ANALYSIS
# ==================================================

elif page == "Clinical Analysis":

    if st.button("← Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

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

    if st.button("← Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

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
