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
    initial_sidebar_state="collapsed"
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

CLINICAL_MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.json")


# ==================================================
# RESPONSIVE LAVENDER THEME
# ==================================================

st.markdown("""
<style>
/* ---------- Global page ---------- */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FAF8FE 0%, #F1ECFA 100%) !important;
}
.block-container {
    max-width: 1200px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    padding-left: clamp(1rem, 3vw, 2.5rem);
    padding-right: clamp(1rem, 3vw, 2.5rem);
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
