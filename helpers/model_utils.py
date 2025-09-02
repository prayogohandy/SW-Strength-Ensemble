import os, joblib
import streamlit as st
from config import model_folder

def load_model(model_num, variant):
    model_path = os.path.join(model_folder, f"{model_num}_{variant}.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.session_state.model_loaded = True
        st.session_state.current_model_info = (model_num, variant)
        st.session_state.model = model
        return model
    else:
        st.error(f"Model file not found: {model_path}")
        return None

