# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from config import feature_bounds, default_values, feature_steps, rename_dict, variant_options, model_num_options
from helpers.feature_utils import compute_derived_features
from helpers.plot_utils import draw_cross_section, draw_elevation
from helpers.input_utils import make_synced_input
from helpers.model_utils import load_model

# ------------------ App Title ------------------
st.title("Shear Wall Shear Strength Prediction")

raw_features = list(feature_bounds.keys())
input_data = {}

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=raw_features + ['Prediction'])

# ------------------ Tabs ------------------
tabs = st.tabs(["Shear Strength Prediction", "Scenario Analysis", "History"])

# ------------------ Inputs & Prediction ------------------
with tabs[0]:
    fig_container = st.container()

    st.subheader("Main Geometry")
    num_col = 2
    columns = st.columns(num_col)

    # Main Geometry Inputs
    main_geometry = ['Hw', 'Lw', 'tw', 'tf', 'bf']
    for i, f in enumerate(main_geometry):
        lb, ub = feature_bounds[f]
        col = columns[i % num_col]
        input_data[f] = make_synced_input(f, lb, ub, feature_steps[f], col, rename_dict, default_values=default_values)

    # Section Properties Inputs
    st.subheader("Section Properties")
    columns = st.columns(num_col)
    section_properties = ['fyh', 'rho_h', 'fyv', 'rho_v', 'fyb', 'rho_b', "fc'", 'P']
    for i, f in enumerate(section_properties):
        lb, ub = feature_bounds[f]
        col = columns[i % num_col]
        scale = 100 if f.startswith('rho') else 1
        input_data[f] = make_synced_input(f, lb, ub, feature_steps[f], col, rename_dict, scale=scale, default_values=default_values)

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # ------------------ Visualization ------------------
    st.subheader("Visualization")
    with fig_container:
        col1, col2 = st.columns(2)
        fig = draw_cross_section(input_data['Lw'], input_data['tw'], input_data['tf'], input_data['bf'], arrow_offset=100)
        col1.pyplot(fig)
        fig2 = draw_elevation(input_data['Hw'], input_data['Lw'], input_data['tf'], arrow_offset=100)
        col2.pyplot(fig2)

    # ------------------ Derived Features ------------------
    st.subheader("Derived Features")
    df = compute_derived_features(df)
    cols_to_drop = ['rho_h', 'rho_v', 'rho_b', 'fyh', 'fyv', 'fyb']
    df_display = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    derived_cols = ['Hw/Lw','Lw/tw','rhofyh','rhofyv','rhofyb','Ab','Ag','Pc','ALR']
    columns = st.columns(3)
    for i, col_name in enumerate(derived_cols):
        columns[i % 3].metric(label=rename_dict[col_name], value=f"{df[col_name].values[0]:.2f}")

    X_input = df_display.values

    # ------------------ Prediction ------------------
    st.subheader("Prediction")
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.current_model_info = None

    variant = st.radio("Choose model variant", options=list(variant_options.keys()),
                       format_func=lambda x: variant_options[x])
    model_num = st.selectbox("Select number of base models", options=model_num_options[variant])

    col1, col2 = st.columns(2)
    load_disabled = st.session_state.current_model_info == (model_num, variant)

    if col1.button("Load Model", disabled=load_disabled):
        with st.spinner("Loading..."):
            if st.session_state.model_loaded:
                st.info(f"Model {st.session_state.current_model_info} is already loaded")
            else:
                model = load_model(model_num, variant)
                st.success(f"Model loaded successfully")

    predict_disabled = not st.session_state.get("model_loaded", False)
    if col2.button("Predict", disabled=predict_disabled):
        with st.spinner("Predicting..."):
            prediction = st.session_state.model.predict(X_input)[0]
            st.success(f"Prediction: {prediction:.2f} kN")

            # Update prediction history
            new_row = input_data.copy()
            new_row['Prediction'] = prediction
            st.session_state.history = pd.concat(
                [st.session_state.history, pd.DataFrame([new_row])],
                ignore_index=True
            )

# ------------------ Scenario Analysis ------------------
with tabs[1]:
    st.header("Scenario Analysis")
    if not st.session_state.get("model_loaded", False):
        st.warning("Load a model in the Prediction tab first.")
    else:
        feature_to_vary = st.selectbox("Select feature to vary", options=raw_features)
        lb, ub = feature_bounds[feature_to_vary]
        vary_range = st.slider(f"Set range for {rename_dict[feature_to_vary]}", min_value=lb, max_value=ub, value=(lb, ub))
        n_points = st.slider("Number of points in scenario", min_value=5, max_value=100, value=20, step=5)
        vary_values = np.linspace(vary_range[0], vary_range[1], n_points)

        scenario_df = pd.DataFrame([input_data]*n_points)
        scenario_df[feature_to_vary] = vary_values
        scenario_df = compute_derived_features(scenario_df)

        cols_to_drop = ['rho_h', 'rho_v', 'rho_b', 'fyh', 'fyv', 'fyb']
        X_scenario = scenario_df.drop(columns=[col for col in cols_to_drop if col in scenario_df.columns])
        predictions = st.session_state.model.predict(X_scenario.values)
        
        st.markdown(f"### Scenario Analysis: ${feature_to_vary}$")
        fig = px.line(x=vary_values, y=predictions)
        fig.update_layout(xaxis_title=f"{feature_to_vary} [mm]", yaxis_title="Predicted Shear Strength (kN)")
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Prediction History ------------------
with tabs[2]:
    st.header("Prediction History")
    history = st.session_state.history
    if st.button("Clear Prediction History"):
        st.session_state.history = pd.DataFrame(columns=raw_features + ['Prediction'])
        history = st.session_state.history

    if history.empty:
        st.warning("No prediction history available.")
    else:
        st.dataframe(history)

        # Feature vs Prediction
        feature_to_plot = st.selectbox("Select feature to plot", options=raw_features)
        fig_feature = px.scatter(history, x=feature_to_plot, y="Prediction", hover_data=history.columns,
                                 labels={feature_to_plot: feature_to_plot, "Prediction": "Shear Strength (kN)"},
                                 title=f"{feature_to_plot} vs Predicted Shear Strength")
        st.plotly_chart(fig_feature, use_container_width=True)

        # Historical Shear Strength Plot
        fig3 = px.scatter(history, x=history.index, y="Prediction", hover_data=history.columns,
                          labels={"x": "Prediction Index", "Prediction": "Shear Strength (kN)"},
                          title="Prediction History")
        st.plotly_chart(fig3, use_container_width=True)
