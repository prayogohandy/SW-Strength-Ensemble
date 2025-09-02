import streamlit as st
from .feature_utils import step_to_format
from .plot_utils import draw_dimension  # if needed for helper annotations

def make_synced_input(f, lb, ub, step, col, rename_dict, default_values, scale=1):
    """
    Slider + text input synced.
    Preserves int/float type, clamps to bounds, reverts invalid input to previous value,
    and formats text input according to step precision.
    """
    if f not in st.session_state:
        st.session_state[f] = default_values[f]

    orig_type = type(default_values[f])
    value_disp = st.session_state[f] * scale
    lb_disp, ub_disp, step_disp = lb * scale, ub * scale, step * scale
    fmt = step_to_format(step_disp)

    slider_col, input_col = col.columns([3,1])

    def update_from_slider():
        val = st.session_state[f"{f}_slider"] / scale
        val = max(min(val, ub), lb)
        st.session_state[f] = orig_type(val) if orig_type is int else val
        st.session_state[f"{f}_text"] = fmt % (st.session_state[f]*scale)

    def update_from_text():
        prev_val = st.session_state[f]
        try:
            val = float(st.session_state[f"{f}_text"])
            val = max(min(val/scale, ub), lb)
            st.session_state[f] = orig_type(val) if orig_type is int else val
        except ValueError:
            st.session_state[f] = prev_val
        st.session_state[f"{f}_slider"] = st.session_state[f]*scale
        st.session_state[f"{f}_text"] = fmt % (st.session_state[f]*scale)

    slider_col.slider(f"{rename_dict[f]}", min_value=lb_disp, max_value=ub_disp,
                      value=value_disp, step=step_disp, key=f"{f}_slider",
                      on_change=update_from_slider)

    input_col.text_input("", value=fmt % value_disp, key=f"{f}_text", on_change=update_from_text)

    return st.session_state[f]
