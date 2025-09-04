import os, joblib
import streamlit as st
from config import model_folder, model_param_keys, model_abbreviations

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

def extract_model_params(model):
    """
    Extracts only the tuned hyperparameters from a trained model,
    detecting the type from model.__class__.__name__.
    """
    class_name = model.base_model.__class__.__name__
    
    param_keys = model_param_keys.get(class_name, [])
    model_params = model.base_model.get_params()
    
    extracted = {}
    scaler_used = model.scaler # can be None or a scaler object, if a list then should be Custom 
    if scaler_used is None:
        extracted['scaler'] = 'none'
    elif isinstance(scaler_used, dict):
        extracted['scaler'] = 'custom'
    else:
        extracted['scaler'] = scaler_used

    feature_mask = model.feature_mask # if None, all features used
    if feature_mask is None or sum(feature_mask) == len(feature_mask):
        extracted['feature_excluded'] = 'none'
    else:
        indices = [i for i, x in enumerate(feature_mask) if x == 0]
        extracted['feature_excluded'] = ','.join(str(i) for i in indices)

    extracted.update({k: model_params[k] for k in param_keys if k in model_params})
    return extracted

def get_model_names(models):
    """
    Returns a list of abbreviated model names.
    If duplicates exist, adds an index suffix: e.g., RF_0, RF_1
    """
    counts = {}
    result = []

    for model in models:
        base_name = model.base_model.__class__.__name__
        abbrev = model_abbreviations.get(base_name, base_name)

        # Get current count (0 if not seen before)
        index = counts.get(abbrev, 0)
        counts[abbrev] = index + 1
        result.append(f"{abbrev}_{index}")
    return result