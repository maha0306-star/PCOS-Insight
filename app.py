
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
import os

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------------------------------
# GROQ GENAI CONFIGURATION
# --------------------------------------------------
from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"

ADVISORY_DISCLAIMER = (
    "This AI-generated information is for educational purposes only. "
    "It is not a medical diagnosis or treatment recommendation. "
    "Please consult a qualified healthcare professional."
)

def get_groq_client():
    key = os.environ.get("GROQ_API_KEY")

    if not key:
        return None

    return Groq(api_key=key)


# --------------------------------------------------
# CLINICAL AI ADVISORY
# --------------------------------------------------

def generate_clinical_advisory(subtype, confidence, patient_features, shap_features):

    client = get_groq_client()

    if client is None:
        return "Groq API key not found. Please check Colab Secrets."

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

def generate_ultrasound_advisory(prediction, pcos_probability,
                                 comparison_probability):

    client = get_groq_client()

    if client is None:
        return "Groq API key not found. Please check Colab Secrets."

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
    page_icon="🌸",
    layout="wide"
)


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

.stButton > button {
    background-color: #8064A8;
    color: white;
    border: none;
    border-radius: 10px;
}

.stButton > button:hover {
    background-color: #674B91;
    color: white;
}

[data-testid="stAlert"] {
    border-radius: 12px;
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

/* Preserve white text on purple buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #8064A8 !important;
    color: #000000 !important;
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

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL AND DATA PATHS
# --------------------------------------------------

CLINICAL_MODEL_PATH = "models/xgboost_model.json"

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

# --------------------------------------------------
# PAGE NAVIGATION — NO SIDEBAR
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

# HOME PAGE
# ==================================================



if page == "Home":

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#E9DDF8,#F5EFFF,#E5DDF5);
        padding:35px 25px;
        border-radius:22px;
        text-align:center;
    ">
        <div style="font-size:35px;">🌸</div>
        <h1 style="color:#604582;">PCOS Insight</h1>
        <h3 style="color:#8064A8;">
            Understand your body. Embrace your journey.
        </h3>
        <p style="color:#594B6D;font-size:17px;">
            Every woman's health journey is unique.
            Explore your health insights with care and confidence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Welcome to your health insight space 💜")
    st.write(
        "Choose an analysis to explore your health-related "
        "clinical features or ultrasound image patterns."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:#F0E9FA;padding:22px;
        border-radius:16px;min-height:145px;">
            <h3 style="color:#604582;">🩷 Clinical Analysis</h3>
            <p style="color:#594B6D;">
                Explore clinical measurements, subtype predictions
                and patient-specific SHAP explanations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore Clinical Analysis", use_container_width=True):
            st.session_state.page = "Clinical Analysis"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background:#F7EDF8;padding:22px;
        border-radius:16px;min-height:145px;">
            <h3 style="color:#604582;">🔬 Ultrasound Analysis</h3>
            <p style="color:#594B6D;">
                Explore AI-assisted ultrasound image classification
                and prediction probabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore Ultrasound Analysis", use_container_width=True):
            st.session_state.page = "Ultrasound Analysis"
            st.rerun()

    st.markdown("""
    <div style="background:white;padding:20px;
    border-radius:15px;border-left:5px solid #A78BCB;
    margin-top:25px;">
        <h4 style="color:#604582;">A gentle reminder 💜</h4>
        <p style="color:#594B6D;font-style:italic;">
            You are more than your diagnosis. Be kind to yourself
            and celebrate every small step.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Research and educational prototype only. "
        "AI predictions are not medical diagnoses."
    )


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

        submitted = st.form_submit_button("Predict Subtype")

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

            model = load_clinical_model()

            input_df = pd.DataFrame([values], columns=FEATURES)

            probabilities = model.predict(xgb.DMatrix(input_df))

            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(1, -1)

            probabilities = probabilities[0]

            predicted_index = int(np.argmax(probabilities))



            # 1. Display predicted subtype first
            st.markdown(
                f"<div style='color:#D32F2F !important; font-size:23px; font-weight:700;'>Predicted PCOS Subtype: {CLASSES[predicted_index]}</div>",
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction Confidence",
                f"{probabilities[predicted_index] * 100:.2f}%"
            )

            # 2. Probability distribution
            probability_df = pd.DataFrame({
                "Subtype": CLASSES,
                "Probability": probabilities
            })

            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.subheader("Subtype Probability Distribution")

            st.bar_chart(
                probability_df.set_index("Subtype")["Probability (%)"]
            )

            st.dataframe(
                probability_df[["Subtype", "Probability (%)"]],
                use_container_width=True,
                hide_index=True
            )

            # 3. Patient-specific SHAP explanation
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

            st.subheader("Patient-Specific SHAP Explanation")
            st.caption(
                "Top five features contributing to this model prediction. "
                "SHAP values represent model contributions, not medical causes."
            )

            st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])

            st.dataframe(
                shap_df[["Feature", "SHAP Value"]],
                hide_index=True
            )

            # 4. Automatically generate clinical advisory
            st.divider()
            st.subheader("Clinical AI Advisory")

            with st.spinner("Generating your clinical advisory..."):
                try:
                    advisory = generate_clinical_advisory(
                        subtype=CLASSES[predicted_index],
                        confidence=float(probabilities[predicted_index]),
                        patient_features=values,
                        shap_features=shap_df[
                            ["Feature", "SHAP Value"]
                        ].to_dict(orient="records")
                    )

                    st.markdown(advisory)
                    st.info(ADVISORY_DISCLAIMER)

                except Exception as e:
                    st.error(f"Clinical advisory error: {e}")

        except Exception as e:

            st.error(f"Clinical prediction error: {e}")


# ==================================================
# SHAP EXPLAINABILITY
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

        if st.button("Analyze Ultrasound"):

            try:

                model = load_ultrasound_model()

                resized_image = image.resize(
                    (224, 224),
                    Image.Resampling.BICUBIC
                )

                image_array = np.array(
                    resized_image.convert("RGB"),
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                image_array = preprocess_input(image_array)

                output = float(
                    model.predict(image_array, verbose=0)[0][0]
                )

                # CNN output represents P(PCO)
                pcos_probability = output
                comparison_probability = 1.0 - output

                if pcos_probability >= comparison_probability:
                    prediction = "PCO"
                else:
                    prediction = "Comparison Class"

                prediction_color = "#D32F2F" if prediction == "PCO" else "#218838"

                st.markdown(
                    f"<div style='color:{prediction_color} !important; font-size:23px; font-weight:700;'>Predicted class: {prediction}</div>",
                    unsafe_allow_html=True
                )

                probability_df = pd.DataFrame({
                    "Class": ["PCO", "Comparison Class"],
                    "Probability (%)": [
                        round(pcos_probability * 100, 2),
                        round(comparison_probability * 100, 2)
                    ]
                })

                st.subheader("Prediction Probabilities")

                st.bar_chart(
                    probability_df.set_index("Class")
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

                st.divider()
                st.subheader("Ultrasound AI Advisory")

                with st.spinner("Generating ultrasound advisory..."):
                    try:
                        advisory = generate_ultrasound_advisory(
                            prediction=prediction,
                            pcos_probability=pcos_probability,
                            comparison_probability=comparison_probability
                        )

                        st.markdown(advisory)
                        st.info(ADVISORY_DISCLAIMER)

                    except Exception as e:
                        st.error(f"Ultrasound advisory error: {e}")

            except Exception as e:

                st.error(f"Ultrasound prediction error: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.caption(
    "Research prototype | Not for medical diagnosis"
)
