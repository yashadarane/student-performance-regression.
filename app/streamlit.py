import streamlit as st
import pandas as pd
import sys
import os

# --- FIX FOR MODULENOTFOUND / IMPORT ERROR ---
# Adds the project root to the system path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.data_loader import load_data
from src.preprocessing import handle_missing_values, encode_categorical, select_features
from src.model import PerformancePredictor
from src.config import path, features, target

st.set_page_config(page_title="Student Performance AI", page_icon="🎓")

@st.cache_resource
def get_model():
    """Loads data, preprocesses, and trains the model once."""
    df = load_data(path)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    X, y = select_features(df, features, target)

    predictor = PerformancePredictor()
    predictor.train(X, y)
    return predictor

try:
    predictor = get_model()

    st.title("🎓 Student Performance Predictor")
    st.write("Adjust the sliders below to predict your Performance Index.")

    # User Input Controls
    hours = st.slider("Hours Studied", 0, 10, 5)
    prev_score = st.number_input("Previous Test Score", 0, 100, 70)
    sleep = st.slider("Sleep Hours", 4, 10, 8)
    papers = st.number_input("Practice Papers Solved", 0, 10, 3)
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    # Convert "Yes/No" to 1/0 (Must match training logic)
    extra_val = 1 if extra == "Yes" else 0

    if st.button("Calculate Prediction", use_container_width=True):
        # IMPORTANT: Columns must be in the exact order as FEATURES list in config.py
        input_data = pd.DataFrame([[hours, prev_score, extra_val, sleep, papers]], 
                                 columns=features)
        
        prediction = predictor.predict(input_data)[0]
        
        st.divider()
        st.metric(label="Predicted Exam Score", value=f"{prediction:.2f}")

except Exception as e:
    st.error(f"Critical App Error: {e}")