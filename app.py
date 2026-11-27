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
