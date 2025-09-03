import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
import networkx as nx
from helpers.model_utils import extract_model_params
from config import model_abbreviations

def draw_dimension(
    ax, p1, p2, offset=(0, 0), text=None, rotation='horizontal',
    color="black", lw=1, text_offset=10, arrowstyle="<|-|>"
):
    """
    Draws a dimension line with extension lines and label.
    The label is placed above the line with a customizable offset.

    ax         : matplotlib axis
    p1, p2     : tuples (x,y) → endpoints on the object
    offset     : (dx, dy) → shift of dimension line away from object
    text       : label (if None, distance is shown)
    rotation   : text rotation ('horizontal' or 'vertical')
    color      : line/text color
    lw         : line width
    text_offset: shift of the label from the line (in data units)
    arrowstyle : matplotlib arrow style string
    """
    import numpy as np

    # Compute offset points (dimension line endpoints)
    p1_off = (p1[0] + offset[0], p1[1] + offset[1])
    p2_off = (p2[0] + offset[0], p2[1] + offset[1])

    # Draw extension lines from object to dimension line
    for pt, pt_off in zip([p1, p2], [p1_off, p2_off]):
        ax.plot([pt[0], pt_off[0]], [pt[1], pt_off[1]], color=color, lw=lw, linestyle='dotted', alpha=0.7)

    # Default label = numeric distance
    if text is None:
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        text = f"{dist:.2f}"
        
    # Handle custom outside arrows
    if arrowstyle == "|>-<|":
        # Default text = distance
        if rotation == 'horizontal':
            # Left arrow (pointing right)
            ax.annotate(
                "", xy=p1_off, xytext=(p1_off[0]-0.001, p1_off[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw)
            )
            # Right arrow (pointing left)
            ax.annotate(
                "", xy=p2_off, xytext=(p2_off[0]+0.001, p2_off[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw)
            )
            # Main dimension line
            ax.plot([p1_off[0], p2_off[0]], [p1_off[1], p2_off[1]], color=color, lw=lw)
            # Text above
            ax.text((p1_off[0]+p2_off[0])/2, p1_off[1]+text_offset, text,
                    ha='center', va='bottom', color=color, fontsize=10)

        else:  # vertical
            # Bottom arrow (pointing up)
            ax.annotate(
                "", xy=p1_off, xytext=(p1_off[0], p1_off[1]-0.001),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw)
            )
            # Top arrow (pointing down)
            ax.annotate(
                "", xy=p2_off, xytext=(p2_off[0], p2_off[1]+0.001),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw)
            )
            # Main vertical line
            ax.plot([p1_off[0], p2_off[0]], [p1_off[1], p2_off[1]], color=color, lw=lw)
            # Text left
            ax.text(p1_off[0]-text_offset, (p1_off[1]+p2_off[1])/2, text,
                    ha='right', va='center', color=color, fontsize=10)
    else:
        ax.annotate(
            "", xy=p1_off, xytext=p2_off,
            arrowprops=dict(arrowstyle=arrowstyle, lw=lw, color=color, shrinkA=0, shrinkB=0)
        )

        # Determine text position
        mid_x = (p1_off[0] + p2_off[0]) / 2
        mid_y = (p1_off[1] + p2_off[1]) / 2
        if rotation == 'horizontal':
            text_x = mid_x
            text_y = mid_y + text_offset
            va, ha = "bottom", "center"
            rot = 0
        elif rotation == 'vertical':
            text_x = mid_x - text_offset
            text_y = mid_y
            ha, va = "right", "center"
            rot = 90
        else:
            # Allow arbitrary angle
            angle = np.degrees(np.arctan2(p2_off[1] - p1_off[1], p2_off[0] - p1_off[0]))
            text_x = mid_x
            text_y = mid_y + text_offset
            ha, va = "center", "center"
            rot = angle

        # Place the label
        ax.text(text_x, text_y, text, ha=ha, va=va, rotation=rot, color=color, fontsize=10, clip_on=True)
        
def draw_cross_section(Lw, tw, tf, bf, arrow_offset):
    fig, ax = plt.subplots(figsize=(6,4), dpi=200)
    
    # Define flange polygon
    flange_x = [0, tf, tf, 0]
    flange_y = [0, 0, bf, bf]
    ax.fill(flange_x, flange_y, color='lightblue', alpha=0.7, label='Flange')

    # Web
    web_x = [tf, Lw-tf, Lw-tf, tf]
    web_height = (bf - tw)/2
    web_y = [web_height, web_height, web_height+tw, web_height+tw]
    ax.fill(web_x, web_y, color='lightgrey', alpha=0.7, label='Web')

    # Define flange polygon
    offset = Lw-tf
    flange_x = [offset, offset + tf, offset + tf, offset]
    flange_y = [0, 0, bf, bf]
    ax.fill(flange_x, flange_y, color='lightblue', alpha=0.7, label='Flange')
    
    # ===== Annotations =====
    # Lw (horizontal length of wall)
    draw_dimension(ax, (0, 0), (Lw, 0), offset=(0,-arrow_offset), text="$L_w$", arrowstyle="<|-|>")

    # bf (flange width, vertical)
    draw_dimension(ax, (0, 0), (0, bf), offset=(-arrow_offset,0), text="$b_f$", rotation='vertical', arrowstyle="<|-|>")

    # tw (web thickness, vertical)
    draw_dimension(ax, (Lw/2, web_height), (Lw/2, web_height+tw),
                   offset=(0, 0), text="$t_w$", rotation='vertical', arrowstyle="|>-<|")

    # tf (flange thickness, horizontal)
    draw_dimension(ax, (0, bf), (tf, bf), offset=(0,arrow_offset), text="$t_f$", arrowstyle="<|-|>")

    # ===== Plot settings =====
    ax.set_aspect('equal')
    ax.axis('off')
    pad_x, pad_y = 3 * arrow_offset, 3 * arrow_offset
    ax.set_xlim(-pad_x, Lw + pad_x)
    ax.set_ylim(-pad_y, max(bf, tw) + pad_y)
    return fig

def draw_elevation(Hw, Lw, tf, arrow_offset):
    fig, ax = plt.subplots(figsize=(6,4), dpi=200)
    
    # Define flange polygon
    flange_x = [0, tf, tf, 0]
    flange_y = [0, 0, Hw, Hw]
    ax.fill(flange_x, flange_y, color='lightblue', alpha=0.7, label='Flange')

    # Web
    web_x = [tf, Lw-tf, Lw-tf, tf]
    web_y = [0, 0, Hw, Hw]
    ax.fill(web_x, web_y, color='lightgrey', alpha=0.7, label='Web')

    # Define flange polygon
    offset = Lw-tf
    flange_x = [offset, offset + tf, offset + tf, offset]
    flange_y = [0, 0, Hw, Hw]
    ax.fill(flange_x, flange_y, color='lightblue', alpha=0.7, label='Flange')
    
    # ===== Annotations =====
    # Lw (horizontal length of wall)
    draw_dimension(ax, (0, 0), (Lw, 0), offset=(0,-arrow_offset), text="$L_w$", arrowstyle="<|-|>")

    # Hw (vertical height of wall)
    draw_dimension(ax, (0, 0), (0, Hw), offset=(-arrow_offset,0), text="$H_w$", rotation='vertical', arrowstyle="<|-|>")

    # tf (flange thickness, horizontal)
    draw_dimension(ax, (0, Hw), (tf, Hw), offset=(0, arrow_offset), text="$t_f$", arrowstyle="<|-|>")

    # ===== Plot settings =====
    ax.set_aspect('equal')
    ax.axis('off')
    pad_x, pad_y = 3 * arrow_offset, 3 * arrow_offset
    ax.set_xlim(-pad_x, Lw + pad_x)
    ax.set_ylim(-pad_y, Hw + pad_y)
    return fig

def plot_prediction(y_train, pred_train, y_test, pred_test):
    fig = plt.figure(figsize=(6, 6))
    all_vals = np.concatenate([y_train, pred_train, y_test, pred_test])
    min_val = np.min(all_vals)
    max_val = np.max(all_vals)
    x_vals = np.linspace(min_val, max_val, 100)
    plt.plot(x_vals, x_vals, 'k--')
    perc = 0.25
    color = 'black'
    upper = x_vals * (1 + perc)
    lower = x_vals / (1 + perc)
    plt.plot(x_vals, upper, linestyle='--', color=color, alpha=0.6)
    plt.plot(x_vals, lower, linestyle='--', color=color, alpha=0.6)
    plt.text(0.9 / 1.25  * max_val, 0.9 * max_val, f'+{int(perc*100)}%', color=color, ha='right', fontsize=10)
    plt.text(0.9 * max_val, 0.9 / 1.3 * max_val, f'-{int(perc*100)}%', color=color, ha='left', fontsize=10)
    plt.scatter(y_train, pred_train, label='Train', alpha=0.6, s=30, edgecolors='blue', marker='o', facecolors='none')
    plt.scatter(y_test, pred_test, label='Test', alpha=0.6, s=30, edgecolors='red', marker='^', facecolors='none')
    plt.xlabel('Experiment Shear Strength (kN)')
    plt.ylabel('Predicted Shear Strength (kN)')
    plt.grid(False)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left')
    plt.tight_layout()

    return fig

def visualize_last_layer_ensemble(ensembler, max_models=None, node_size=15, height=600, width="100%"):
    """
    Returns an HTML string for a circular visualization of the last layer and ensemble node.
    
    ensembler: fitted ensemble object
    max_models: int, truncate last layer if too large
    """
    G = nx.DiGraph()
    
    # Truncate if needed
    last_models = ensembler.prev_models
    if max_models and len(last_models) > max_models + 1:
        half = max_models // 2
        last_models_display = last_models[:half] + last_models[-(max_models - half):]
        hidden_count = len(last_models) - len(last_models_display)
    else:
        last_models_display = last_models
        hidden_count = 0
    
    # Add last layer nodes
    for i, model in enumerate(last_models_display):
        node_id = f"model_{i}"
        name = model_abbreviations.get(type(model.base_model).__name__)
        label = f"{name}\nVal: {model.oof_score:.3f}\nTest: {model.test_score:.3f}"
        G.add_node(node_id, label=label, size=node_size)

        model_params = extract_model_params(model.base_model)
        params_str = "\n".join(f"{k}: {v}" for k, v in model_params.items())
        G.nodes[node_id]['title'] = params_str  # hover info
    
    # Add ensemble node
    G.add_node("ensemble", shape="box", size=node_size*1.5, x=0, y=0, 
               label=f"Ensemble\nVal: {ensembler.oof_score:.3f}\nTest: {ensembler.test_score:.3f}")
    
    # Connect last layer nodes to ensemble with weights
    for i, model in enumerate(last_models_display):
        node_id = f"model_{i}"
        weight = ensembler.weights[i]
        G.add_edge(node_id, "ensemble", label=f"{weight:.2f}")
    
    # Ellipsis node if truncated
    if hidden_count > 0:
        ellipsis_id = "ellipsis"
        G.add_node(ellipsis_id, label=f"Hidden MLs\n (+{hidden_count})", size=node_size*1.5, shape="ellipse")
        G.add_edge(ellipsis_id, "ensemble")

    # Create PyVis network
    net = Network(height=f"{height}px", width=width, directed=True)
    net.from_nx(G)
    return net.generate_html()