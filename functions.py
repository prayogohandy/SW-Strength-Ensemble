# -------------------- Imports --------------------
import numpy as np
import pandas as pd
from copy import deepcopy

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- Utility Functions --------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return (abs((y_true - y_pred) / y_true)).mean() * 100

def compute_metrics(y_true, y_pred):
    """Helper to compute all metrics at once."""
    return {
        'R2': r2_score(y_true, y_pred) * 100,
        'RMSE': rmse(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }

# -------------------- Transformer Builders --------------------
def get_scaler_map(numerical_cols, scaler, default='standard'):
    if isinstance(scaler, dict):
        scaler_map = {col: scaler.get(col, default) for col in numerical_cols}
    elif isinstance(scaler, list):
        if len(scaler) != len(numerical_cols):
            raise ValueError("Scaler list length must match number of numerical features")
        scaler_map = dict(zip(numerical_cols, scaler))
    else:
        scaler_map = {col: scaler for col in numerical_cols}
    return scaler_map

def build_numerical_transformer(scaler_map):
    scaler_dict = {
        'none': None,
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'power': PowerTransformer(method='box-cox'),
        'yeo-johnson': PowerTransformer(method='yeo-johnson'),
        'quantile-normal': QuantileTransformer(n_quantiles=200, output_distribution='normal'),
        'quantile-uniform': QuantileTransformer(n_quantiles=200, output_distribution='uniform'),
    }
    scaler_groups = {}
    for col, sc_name in scaler_map.items():
        if sc_name not in scaler_dict:
            raise ValueError(f"Unknown scaler: {sc_name}")
        scaler_groups.setdefault(sc_name, []).append(col)
    transformers = []
    for sc_name, cols in scaler_groups.items():
        if scaler_dict[sc_name] is not None:
            transformers.append((sc_name, scaler_dict[sc_name], cols))
    return ColumnTransformer(transformers, remainder='passthrough')

def build_categorical_transformer(df, categorical_cols, encoder_type, fit, existing_encoders=None):
    if not categorical_cols:
        return None, existing_encoders if existing_encoders else [], []
    encoders = [] if existing_encoders is None else existing_encoders
    len_ohes = []
    encoded_dfs = []
    if encoder_type == 'onehot':
        for i, col in enumerate(categorical_cols):
            if fit:
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = enc.fit_transform(df[[col]])
                encoders.append(enc)
            else:
                enc = encoders[i]
                encoded = enc.transform(df[[col]])
            ohe_cols = enc.get_feature_names_out([col])
            ohe_df = pd.DataFrame(encoded, columns=ohe_cols, index=df.index)
            encoded_dfs.append(ohe_df)
            len_ohes.append(ohe_df.shape[1])
        encoded_df = pd.concat(encoded_dfs, axis=1)
    elif encoder_type == 'ordinal':
        if fit:
            enc = OrdinalEncoder()
            encoded = enc.fit_transform(df[categorical_cols])
            encoders = enc
        else:
            enc = encoders
            encoded = enc.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=categorical_cols, index=df.index)
        len_ohes = []
    else:
        raise ValueError("categorical_encoder must be 'onehot' or 'ordinal'.")
    return encoded_df, encoders, len_ohes

# -------------------- Data Processing --------------------
def process_array(X, feature_info, scaler='standard', categorical_encoder='onehot', existing_info=None, return_df=False, feature_names=None):
    fit = existing_info is None
    numerical_indices = feature_info['numerical_indices']
    categorical_indices = feature_info['categorical_indices']
    numerical_cols = feature_info['numerical_cols']
    categorical_cols = feature_info['categorical_cols']

    if feature_names is None:
        feature_names = numerical_cols + categorical_cols
    df = pd.DataFrame(X, columns=feature_names)

    df_numerical = df[numerical_cols]
    df_categorical = df[categorical_cols] if categorical_cols else pd.DataFrame(index=df.index)

    if fit:
        scaler_map = get_scaler_map(numerical_cols, scaler)
        numerical_transformer = build_numerical_transformer(scaler_map)
        df_numerical_scaled = pd.DataFrame(
            numerical_transformer.fit_transform(df_numerical),
            columns=numerical_cols,
            index=df.index
        )
    else:
        numerical_transformer = existing_info['numerical_transformer']
        df_numerical_scaled = pd.DataFrame(
            numerical_transformer.transform(df_numerical),
            columns=numerical_cols,
            index=df.index
        )

    encoded_df, encoders, len_ohes = build_categorical_transformer(
        df,
        categorical_cols,
        categorical_encoder,
        fit,
        existing_encoders=existing_info['encoders'] if not fit else None
    ) if categorical_cols else (None, [] if categorical_encoder == 'onehot' else None, [])

    if encoded_df is not None:
        final_df = pd.concat([df_numerical_scaled, encoded_df], axis=1)
    else:
        final_df = df_numerical_scaled

    X_final = final_df if return_df else final_df.to_numpy(dtype=np.float32)

    additional_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'numerical_indices': numerical_indices,
        'categorical_indices': categorical_indices,
        'numerical_transformer': numerical_transformer,
        'encoders': encoders,
        'encoder_type': categorical_encoder,
        'len_ohes': len_ohes if categorical_encoder == 'onehot' else None
    }
    return X_final, additional_info
