# main.py
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from graphviz import Source

from config import (feature_bounds, default_values, 
                    feature_steps, rename_dict, variant_options, 
                    model_num_options, model_abbreviations)
from helpers.feature_utils import compute_derived_features
from helpers.plot_utils import draw_cross_section, draw_elevation, visualize_last_layer_ensemble
from helpers.input_utils import make_synced_input
from helpers.model_utils import load_model, extract_model_params, get_model_names

# ------------------ App Title ------------------
st.set_page_config(
    page_title="Ensemble Predictor",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.title("Shear Wall Shear Strength Prediction")

raw_features = list(feature_bounds.keys())
input_data = {}

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=raw_features + ['Prediction'])

# ------------------ Prediction ------------------
# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.current_model_info = None


# ------------------ Tabs ------------------
tabs = st.tabs(["Shear Wall Details", "Ensemble Explorer", "Scenario Analysis", "History"])

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Input Parameters")
with st.sidebar.expander("Main Geometry", expanded=False):
    container = st.container()
    # Main Geometry Inputs
    main_geometry = ['Hw', 'Lw', 'tw', 'tf', 'bf']
    for i, f in enumerate(main_geometry):
        lb, ub = feature_bounds[f]
        input_data[f] = make_synced_input(f, lb, ub, feature_steps[f], container, rename_dict, 
                                          default_values=default_values)
        
with st.sidebar.expander("Section Properties", expanded=False):
    # Section Properties Inputs
    container = st.container()
    section_properties = ['fyh', 'rho_h', 'fyv', 'rho_v', 'fyb', 'rho_b', "fc'", 'P']
    for i, f in enumerate(section_properties):
        lb, ub = feature_bounds[f]
        scale = 100 if f.startswith('rho') else 1
        input_data[f] = make_synced_input(f, lb, ub, feature_steps[f], container, rename_dict, 
                                          scale=scale, default_values=default_values)

st.sidebar.markdown("---")

# ------------------ Inputs & Prediction ------------------
with tabs[0]:
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    # ------------------ Derived Features ------------------
    st.subheader("Derived Features")
    df = compute_derived_features(df)
    cols_to_drop = ['rho_h', 'rho_v', 'rho_b', 'fyh', 'fyv', 'fyb']
    df_display = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    derived_cols = ['Hw/Lw','Lw/tw','rhofyh','rhofyv','rhofyb','Ab','Ag','Pc','ALR']
    columns = st.columns(3)
    for i, col_name in enumerate(derived_cols):
        if col_name in ['Ab','Ag']:
            format_str = "{:d}"
        else:
            format_str = "{:.2f}"
        columns[i % 3].metric(label=rename_dict[col_name], value=format_str.format(df[col_name].values[0]))

    X_input = df_display.values
    st.markdown("---")
    # ------------------ Visualization ------------------
    fig_container = st.container()
    with fig_container:
        st.subheader("Shear Wall Visualization")
        col1, col2 = st.columns(2)
        fig = draw_cross_section(input_data['Lw'], input_data['tw'], input_data['tf'], 
                                 input_data['bf'], arrow_offset=100)
        col1.pyplot(fig)
        fig2 = draw_elevation(input_data['Hw'], input_data['Lw'], input_data['tf'], 
                              arrow_offset=100)
        col2.pyplot(fig2)


# ------------------ Sidebar Model Selection ------------------

st.sidebar.header("Prediction")
with st.sidebar.expander("Model Selection", expanded=False):
    # Variant and model number selection
    variant = st.radio(
        "Choose model variant",
        options=list(variant_options.keys()),
        format_func=lambda x: variant_options[x]
    )
    model_num = st.selectbox(
        "Number of base models",
        options=model_num_options[variant]
    )

# Load model button
col1, col2 = st.sidebar.columns(2)
load_disabled = st.session_state.current_model_info == f"{model_num}-{variant}"
load_label = "Load Model" if not load_disabled else "Model Loaded"
if col1.button(load_label, disabled=load_disabled):
    if not load_disabled:
        start_time = time.time()
        with st.spinner("Loading..."):
            model = load_model(model_num, variant)
        elapsed = time.time() - start_time
        st.sidebar.success(f"Model loaded successfully in {elapsed:.2f} seconds")
        st.session_state.model_loaded = True
        st.session_state.current_model_info = f"{model_num}-{variant}"
        st.session_state.model = model

# Predict button
predict_disabled = not st.session_state.get("model_loaded", False)
if col2.button("Predict", disabled=predict_disabled):
    start_time = time.time()
    with st.spinner("Predicting..."):
        prediction = st.session_state.model.predict(X_input)[0]
    elapsed = time.time() - start_time
    st.sidebar.markdown(f"**Prediction:** {prediction:.2f} kN  \n**Time:** {elapsed:.2f} seconds")

    # Update prediction history
    new_row = input_data.copy()
    new_row['Prediction'] = prediction
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new_row])],
        ignore_index=True
    )

# ------------------ Ensemble Structure ------------------
with tabs[1]:
    if not st.session_state.get("model_loaded", False):
        st.warning("Load a model in the Prediction tab first.")
    else:
        view_mode = st.selectbox(
            "Select View Mode",
            ["Ensemble Structure", "Individual Model Details", "Model Types", "Model Importance"],
        )
        st.markdown("---")
        model = st.session_state.model
        num_levels = len(model.all_models)

        if view_mode == "Ensemble Structure":
            max_model_number = len(model.all_models[-1])
            max_models = st.slider("Max models shown:", 2, min(50, max_model_number), min(5, max_model_number))
            html_content = visualize_last_layer_ensemble(ensembler=model.ensembler, node_size=15, max_models=max_models)
            st.components.v1.html(html_content, height=600)

        elif view_mode == "Individual Model Details":
            # Level selection
            level_idx = 0 # st.selectbox("Select Level", range(num_levels), key="indiv_level")
            models_at_level = model.all_models[level_idx]

            # Get model types at this level
            model_types = [m.base_model.__class__.__name__ for m in models_at_level]
            unique_types = sorted(list(set(model_types)))
            type_options = ["All Models"] + unique_types

            # Model type selection
            col1, col2 = st.columns(2)
            selected_type = col1.selectbox("Select Model Type", type_options, key="indiv_model_type")

            # Filter models based on type
            if selected_type == "All Models":
                filtered_models = models_at_level
            else:
                filtered_models = [m for m in models_at_level if m.base_model.__class__.__name__ == selected_type]
            
            # Model index selection within filtered models
            model_idx = col2.slider(
                "Select Model Index",
                min_value=0,
                max_value=len(filtered_models) - 1,
                value=0,
                step=1,
                key="indiv_model_idx"
            )
            individual_model = filtered_models[model_idx]

            # Show model details
            model_params = extract_model_params(individual_model)
            params_str = "\n".join(f"{k}: {v}" for k, v in model_params.items())

            st.markdown(f"**Model Type:** {individual_model.base_model.__class__.__name__}")
            st.markdown(f"**Validation RMSE:** {getattr(individual_model, 'oof_score', np.nan):.2f} kN")
            st.markdown(f"**Test RMSE:** {getattr(individual_model, 'test_score', np.nan):.2f} kN")
            st.text_area("Hyperparameters", value=params_str, height=200)


        elif view_mode == "Model Types":
            level_idx = 0 # st.selectbox("Select Level", range(num_levels), key="types_level")
            model_types = [
                model_abbreviations.get(m.base_model.__class__.__name__, m.base_model.__class__.__name__)
                for m in model.all_models[level_idx]
            ]
            type_counts = pd.Series(model_types).value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index)
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "Model Importance":
            final_models = model.all_models[-1]
            names = get_model_names(final_models)

            # Aggregate model coefficients or fallback to uniform importance
            coefs = [abs(model.coef_) for model in model.ensembler.model.models]
            model_importance = np.mean(coefs, axis=0)

            fig = px.bar(x=names, y=model_importance, labels={"x": "Model", "y": "|Coefficient|"})
            st.plotly_chart(fig, use_container_width=True)

# ------------------ Scenario Analysis ------------------
with tabs[2]:
    if not st.session_state.get("model_loaded", False):
        st.warning("Load a model in the Prediction tab first.")
    else:
        feature_to_vary = st.selectbox("Select feature to vary", options=raw_features)
        st.markdown("---")
        lb, ub = feature_bounds[feature_to_vary]
        vary_range = st.slider(f"Set range for {rename_dict[feature_to_vary]}", 
                               min_value=lb, max_value=ub, value=(lb, ub))
        n_points = st.slider("Number of points in scenario", min_value=5, max_value=100, 
                             value=20, step=5)
        vary_values = np.linspace(vary_range[0], vary_range[1], n_points)

        scenario_df = pd.DataFrame([input_data]*n_points)
        scenario_df[feature_to_vary] = vary_values
        scenario_df = compute_derived_features(scenario_df)

        cols_to_drop = ['rho_h', 'rho_v', 'rho_b', 'fyh', 'fyv', 'fyb']
        X_scenario = scenario_df.drop(columns=[col for col in cols_to_drop 
                                               if col in scenario_df.columns])
        predictions = st.session_state.model.predict(X_scenario.values)
        
        fig = px.line(x=vary_values, y=predictions)
        fig.update_layout(xaxis_title=f"{feature_to_vary} [mm]", 
                          yaxis_title="Predicted Shear Strength (kN)")
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Prediction History ------------------
with tabs[3]:
    history = st.session_state.history

    if history.empty:
        st.warning("No prediction history available.")
    else:
        if st.button("Clear Prediction History"):
            st.session_state.history = pd.DataFrame(columns=raw_features + ['Prediction'])
            history = st.session_state.history

        st.dataframe(history)
        st.markdown("---")
        # Feature vs Prediction
        feature_to_plot = st.selectbox("Select feature to plot", options=raw_features)
        fig_feature = px.scatter(history, x=feature_to_plot, y="Prediction", 
                                 hover_data=history.columns,
                                 labels={feature_to_plot: feature_to_plot, 
                                         "Prediction": "Shear Strength (kN)"},
                                 title=f"{feature_to_plot} vs Predicted Shear Strength")
        st.plotly_chart(fig_feature, use_container_width=True)
        
        st.markdown("---")
        # Historical Shear Strength Plot
        fig3 = px.scatter(history, x=history.index, y="Prediction", hover_data=history.columns,
                          labels={"x": "Prediction Index", "Prediction": "Shear Strength (kN)"},
                          title="Prediction History")
        st.plotly_chart(fig3, use_container_width=True)
