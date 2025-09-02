import os

feature_bounds = {
    'Hw': (100, 5000), 'Lw': (100, 6000), 'tw': (10, 300),
    'tf': (10, 500), 'bf': (10, 3500), "fc'": (10.0, 180.0),
    'fyh': (0, 1500), 'fyv': (0, 1500), 'fyb': (0, 1500),
    'rho_h': (0.0, 0.04), 'rho_v': (0.0, 0.04), 'rho_b': (0.0, 0.16),
    'P': (0, 6500)
}

default_values = {
    'Hw': 600, 'Lw': 1000, 'tw': 60, 'tf': 100, 'bf': 200,
    "fc'": 34.0, 'fyh': 327, 'fyv': 327, 'fyb': 460,
    'rho_h': 0.0063, 'rho_v': 0.0063, 'rho_b': 0.0314, 'P': 0
}

feature_steps = {
    'Hw': 1, 'Lw': 1, 'tw': 1, 'tf': 1, 'bf': 1,
    "fc'": 0.1, 'fyh': 1, 'fyv': 1, 'fyb': 1,
    'rho_h': 0.0001, 'rho_v': 0.0001, 'rho_b': 0.0001, 'P': 1
}

rename_dict = {
    'Hw': r'$H_w$ [mm]', 'Lw': r'$L_w$ [mm]', 'tw': r'$t_w$ [mm]', 'tf': r'$t_f$ [mm]',
    'bf': r'$b_f$ [mm]', "fc'": r"$f'_c$ [MPa]", 'fyh': r'$f_{yh}$ [MPa]', 'fyv': r'$f_{yv}$ [MPa]',
    'fyb': r'$f_{yb}$ [MPa]', 'rho_h': r'$\rho_h$ [%]', 'rho_v': r'$\rho_v$ [%]',
    'rho_b': r'$\rho_b$ [%]', 'P': r'$P$ [kN]', 'Hw/Lw': r'$H_w / L_w$ [-]', 
    'Lw/tw': r'$L_w / t_w$ [-]', 'Hw/tw': r'$H_w / t_w$ [-]', 'bf/tw': r'$b_f / t_w$ [-]', 
    'tf/tw': r'$t_f / t_w$ [-]', 'rhofyh': r'$\rho_h \cdot f_{yh}$ [MPa]', 
    'rhofyv': r'$\rho_v \cdot f_{yv}$ [MPa]', 'rhofyb': r'$\rho_b \cdot f_{yb}$ [MPa]',
    'Ab': r'$A_b$ [mm²]', 'Ag': r'$A_g$ [mm²]', 'Ab/Ag': r'$A_b / A_g$ [-]', 
    'Pc': r'$P_c$ [kN]', 'ALR': r'$\mathrm{ALR}$ [-]'
}

variant_options = {0: "All Features + Standard Scaler",
                   1: "Masked Features + Standard Scaler",
                   2: "All Features + Optimized Scaler",
                   3: "Masked Features + Optimized Scaler"}

model_folder = "model"

# Build model number options automatically
model_num_options = {}
for filename in os.listdir(model_folder):
    if filename.endswith(".pkl"):
        parts = filename.replace(".pkl", "").split("_")
        if len(parts) == 2:
            model_num, variant = map(int, parts)
            model_num_options.setdefault(variant, []).append(model_num)
for k in model_num_options:
    model_num_options[k] = sorted(model_num_options[k])
