from copy import deepcopy
import graphviz
import numpy as np
from scipy.optimize import minimize
import shap
from sklearn.base import clone
from sklearn.model_selection import KFold
from tqdm import tqdm
import time
from datetime import timedelta
from functions import (compute_metrics, process_array)

# Regression
class L0BaggingModelRegressor:
    def __init__(self, base_model, score_func, n_splits=5, random_state=42,
                 scaler=None, feature_info=None, feature_mask=None):
        self.base_model = base_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.score_func = score_func
        self.models = []
        self.oof_preds = None
        self.oof_score = None
        self.test_preds = None
        self.test_score = None
        self.scaler = scaler
        self.feature_info = feature_info  # Feature information for scaling
        self.feature_mask = feature_mask  # Mask to skip features if needed
        self.scaler_info = []  # Store scaler information if needed

    def _clone_model(self):
        try:
            return clone(self.base_model)
        except Exception:
            return deepcopy(self.base_model)
    
    def fit(self, X_train, y_train, X_test, y_test):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_preds = np.zeros(len(X_train))
        self.test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val = X_train[val_idx]
            X_te = X_test

            # Apply scaler if specified
            if self.scaler is not None and self.feature_info is not None:
                X_tr, infos = process_array(X_tr, scaler=self.scaler, feature_info=self.feature_info, existing_info=None)
                X_val, _ = process_array(X_val, scaler=self.scaler, feature_info=self.feature_info, existing_info=infos)
                X_te, _ = process_array(X_test, scaler=self.scaler, feature_info=self.feature_info, existing_info=infos)
                self.scaler_info.append(infos)
            else:
                self.scaler_info.append(None)

            # Apply feature mask if provided
            if self.feature_mask is not None:
                X_tr = X_tr[:, self.feature_mask]
                X_val = X_val[:, self.feature_mask]
                X_te = X_te[:, self.feature_mask]

            model = self._clone_model()
            model.fit(X_tr, y_tr)
            self.models.append(model)

            self.oof_preds[val_idx] = model.predict(X_val)
            self.test_preds += model.predict(X_te) / self.n_splits
            
        if self.score_func:
            self.oof_score = self.score_func(y_train, self.oof_preds)
            self.test_score = self.score_func(y_test, self.test_preds)
        return self

    def predict(self, X_original):
        if not self.models:
            raise ValueError("No models trained. Call `.fit()` first.")
        preds = []
        for i, model in enumerate(self.models):
            if self.scaler is not None and self.feature_info is not None and self.scaler_info[i] is not None:
                X, _ = process_array(X_original, scaler=self.scaler, feature_info=self.feature_info, existing_info=self.scaler_info[i])
            else:
                X = X_original
            if self.feature_mask is not None:
                X = X[:, self.feature_mask]
            preds.append(model.predict(X))
        return np.mean(preds, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.score_func(y, y_pred)

    def strip_model(self):
        self.models = []  # remove trained models
        return self

class L1BaggingModelRegressor:
    def __init__(self, base_model, prev_models, score_func,  
                 feature_skip=True, n_splits=5, random_state=42,
                 scaler=None, feature_info=None, feature_mask=None):
        self.base_model = base_model
        self.prev_models = prev_models
        self.n_meta_feature = len(prev_models)
        self.feature_skip = feature_skip
        self.n_splits = n_splits
        self.random_state = random_state
        self.score_func = score_func
        self.models = []
        self.scaler = scaler
        self.feature_info = feature_info
        self.feature_mask = feature_mask
        self.scaler_info = []
        self.X_train = None
        self.X_test = None
        self.oof_preds = None
        self.oof_score = None
        self.test_preds = None
        self.test_score = None
        self.meta_feature_importance = None

    def _clone_model(self):
        try:
            return clone(self.base_model)
        except Exception:
            return deepcopy(self.base_model)
    
    def fit(self, X_train, y_train, X_test, y_test):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_preds = np.zeros(len(X_train))
        self.test_preds = np.zeros(len(X_test))

        prev_oof_feats = [model.oof_preds.reshape(-1, 1) for model in self.prev_models]
        prev_test_feats = [model.test_preds.reshape(-1, 1) for model in self.prev_models]
        prev_oof_stack = np.hstack(prev_oof_feats)
        prev_test_stack = np.hstack(prev_test_feats)

        self.X_train = X_train
        self.X_test = X_test
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val = X_train[val_idx]
            prev_oof_train = prev_oof_stack[train_idx]
            prev_oof_val = prev_oof_stack[val_idx]

            # Feature processing
            if self.feature_skip:
                if self.scaler is not None and self.feature_info is not None:
                    X_tr, infos = process_array(X_tr, scaler=self.scaler, feature_info=self.feature_info, existing_info=None)
                    X_val, _ = process_array(X_val, scaler=self.scaler, feature_info=self.feature_info, existing_info=infos)
                    X_te, _ = process_array(X_test, scaler=self.scaler, feature_info=self.feature_info, existing_info=infos)
                    self.scaler_info.append(infos)
                else:
                    X_te = X_test
                    self.scaler_info.append(None)

                if self.feature_mask is not None:
                    X_tr = X_tr[:, self.feature_mask]
                    X_val = X_val[:, self.feature_mask]
                    X_te = X_te[:, self.feature_mask]

                X_tr = np.hstack([X_tr, prev_oof_train])
                X_val = np.hstack([X_val, prev_oof_val])
                X_te = np.hstack([X_te, prev_test_stack])
            else:
                X_tr = prev_oof_train
                X_val = prev_oof_val
                X_te = prev_test_stack

            model = self._clone_model()
            model.fit(X_tr, y_tr)
            self.models.append(model)
            self.oof_preds[val_idx] = model.predict(X_val)
            self.test_preds += model.predict(X_te) / self.n_splits
            
        if self.score_func:
            self.oof_score = self.score_func(y_train, self.oof_preds)
            self.test_score = self.score_func(y_test, self.test_preds)
        return self

    def predict(self, X_original):
        if not self.models:
            raise ValueError("No models trained. Call `.fit()` first.")
        meta_features = np.hstack([model.predict(X_original).reshape(-1, 1) for model in self.prev_models])
        preds = []
        for i, model in enumerate(self.models):
            if self.feature_skip:
                if self.scaler is not None and self.feature_info is not None and self.scaler_info[i] is not None:
                    X, _ = process_array(X_original, scaler=self.scaler, feature_info=self.feature_info, existing_info=self.scaler_info[i])
                else:
                    X = X_original
                if self.feature_mask is not None:
                    X = X[:, self.feature_mask]
                X_meta = np.hstack([X, meta_features])
            else:
                X_meta = meta_features
            preds.append(model.predict(X_meta).reshape(-1, 1))
        return np.mean(preds, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.score_func(y, y_pred)

    def calculate_meta_feature_importance(self):
        """Compute average normalized importance of each meta-feature using SHAP."""
        if self.meta_feature_importance is not None:
            return self.meta_feature_importance

        all_shap_values = []
        for model in self.models:
            try:
                explainer = shap.Explainer(model)
                shap_vals = explainer(self.X_train)
                shap_array = np.abs(shap_vals.values[:, -self.n_meta_feature:])
            except Exception:
                shap_array = np.ones((self.X_train.shape[0], self.n_meta_feature))
            all_shap_values.append(shap_array)

        avg_shap_values = np.mean(np.stack(all_shap_values), axis=0)
        normalized_values = avg_shap_values / (np.max(avg_shap_values, axis=1, keepdims=True) + 1e-8)
        self.meta_feature_importance = np.mean(normalized_values, axis=0)
        return self.meta_feature_importance

class WeightedEnsemblerRegressor:
    def __init__(self, bagged_models, score_func, maximize_score=False):
        self.prev_models = bagged_models
        self.n_models = len(bagged_models)
        self.score_func = score_func
        self.weights = None
        self.oof_preds = None
        self.test_preds = None
        self.oof_score = None
        self.test_score = None
        self.y_train = None
        self.y_test = None
        self.maximize_score = maximize_score

    def _objective(self, weights):
        weighted_preds = sum(w * m.oof_preds for w, m in zip(weights, self.prev_models))
        sign = -1 if self.maximize_score else 1
        return sign * self.score_func(self.y_train, weighted_preds) 

    def fit(self, y_train, y_test):
        self.y_train = y_train
        self.y_test = y_test
        # Initial equal weights
        init_weights = np.ones(self.n_models) / self.n_models
        bounds = [(0.0, 1.0)] * self.n_models
        constraints = [{'type': 'eq', 'fun': lambda w: 1 - sum(w)}]

        # Optimize weights
        result = minimize(self._objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights = result.x / np.sum(result.x) # ensure sum(weight) = 1

        # Compute final weighted predictions
        self.oof_preds = sum(w * m.oof_preds for w, m in zip(self.weights, self.prev_models))
        self.test_preds = sum(w * m.test_preds for w, m in zip(self.weights, self.prev_models))

        score_func = self.prev_models[0].score_func
        self.oof_score = score_func(self.y_train, self.oof_preds)
        self.test_score = score_func(self.y_test, self.test_preds)
        return self
        
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call `.fit()` first to compute ensemble weights.")
        preds = [model.predict(X) for model in self.prev_models]
        preds = np.array(preds)
        weighted_preds = np.average(preds, axis=0, weights=self.weights)
        return weighted_preds

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.score_func(y, y_pred)

class MetaModelEnsemblerRegressor:
    def __init__(self, bagged_models, meta_model, score_func):
        self.prev_models = bagged_models
        self.n_models = len(bagged_models)
        self.meta_model = clone(meta_model)
        self.score_func = score_func
        self.oof_preds = None
        self.test_preds = None
        self.oof_score = None
        self.test_score = None
        self.weights = None

    def fit(self, y_train, y_test):
        # use previous model oof stack as a meta feature
        prev_oof_feats = [model.oof_preds.reshape(-1, 1) for model in self.prev_models]
        prev_test_feats = [model.test_preds.reshape(-1, 1) for model in self.prev_models]
        X_train = np.hstack(prev_oof_feats)
        X_test = np.hstack(prev_test_feats)

        self.meta_model.fit(X_train, y_train)

        # weight for plotting
        self.weights = self.meta_model.coef_
        
        self.oof_preds = self.meta_model.predict(X_train)
        self.test_preds = self.meta_model.predict(X_test)

        if self.score_func:
            self.oof_score = self.score_func(y_train, self.oof_preds)
            self.test_score = self.score_func(y_test, self.test_preds)
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call `.fit()` first to compute ensemble weights.")
        preds = [model.predict(X).reshape(-1, 1) for model in self.prev_models]
        preds = np.hstack(preds)
        return self.meta_model.predict(preds)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.score_func(y, y_pred)

class MetaModelEnsemblerBaggedRegressor:
    def __init__(self, bagged_models, meta_model, score_func):
        self.prev_models = bagged_models
        self.n_models = len(bagged_models)
        self.meta_model = clone(meta_model)
        self.model = None
        self.score_func = score_func
        self.oof_preds = None
        self.test_preds = None
        self.oof_score = None
        self.test_score = None
        self.weights = None

    def fit(self, y_train, y_test):
        # use previous model oof stack as a meta feature
        prev_oof_feats = [model.oof_preds.reshape(-1, 1) for model in self.prev_models]
        prev_test_feats = [model.test_preds.reshape(-1, 1) for model in self.prev_models]
        X_train = np.hstack(prev_oof_feats)
        X_test = np.hstack(prev_test_feats)

        self.model = L0BaggingModelRegressor(
                self.meta_model,
                score_func=self.score_func,
                n_splits=self.prev_models[0].n_splits,
                random_state=self.prev_models[0].random_state,
                scaler=None, 
                feature_info=None,
                feature_mask=None
            )
        
        self.model.fit(X_train, y_train, X_test, y_test)

        # weight for plotting
        coefs = np.array([m.coef_ for m in self.model.models])
        self.weights = np.mean(coefs, axis=0)
        
        self.oof_preds = self.model.oof_preds
        self.test_preds = self.model.test_preds

        if self.score_func:
            self.oof_score = self.model.oof_score
            self.test_score = self.model.test_score
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Call `.fit()` first to compute ensemble weights.")
        preds = [model.predict(X).reshape(-1, 1) for model in self.prev_models]
        preds = np.hstack(preds)
        return self.model.predict(preds)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.score_func(y, y_pred)

def get_name(model):
    name = model.__class__.__name__
    model_abbreviations = {
        "RandomForestRegressor": "RF",
        "ExtraTreesRegressor": "ET",
        "GradientBoostingRegressor": "GB",
        "DecisionTreeRegressor": "DT",
        "KNeighborsRegressor": "KNN",
        "SVR": "SVR",
        "XGBRegressor": "XGB",
        "LGBMRegressor": "LGB",
        "CatBoostRegressor": "CB",
        "MLPRegressor": "MLP",
        "AdaBoostRegressor": "ADA",
        "HistGradientBoostingRegressor": "HGB",
        "GaussianProcessRegressor": "GPR",
        "Ridge": "Ridge",
    }
    return model_abbreviations.get(name, name)

def to_grayscale_hex(normalized_value, min_gray=0, max_gray=200):
    gray = int(max_gray - normalized_value * (max_gray - min_gray))
    return f"#{gray:02x}{gray:02x}{gray:02x}"

class StackingRegressor:
    def __init__(self, config, score_func, meta_model=None, maximize_score=False, is_bagged=False, 
                 n_splits=5, random_state=42, verbose=False,
                 scaler=None, feature_info=None, feature_mask=None):
        self.config = config
        self.score_func = score_func
        self.n_splits = n_splits
        self.random_state = random_state
        self.all_models = []
        self.meta_model = meta_model
        self.maximize_score = maximize_score
        self.ensembler = None
        self.is_bagged = is_bagged
        self.verbose = verbose

        # if building from scratch, use scaler and feature info
        self.scaler = scaler
        self.feature_info = feature_info  # Feature information for scaling
        self.feature_mask = feature_mask  # Mask to skip features if needed

    def fit(self, X_train, y_train, X_test, y_test):
        prev_models = []

        for i, cfg in enumerate(self.config):
            level_models = []
            feature_skip = cfg.get("feature_skip", True)
            
            models = cfg["models"]
            if self.verbose:
                models = tqdm(cfg["models"], desc=f"Level {i} Models")
            for base_model in models:
                if self.verbose:
                    model_name = get_name(base_model)
                    models.set_description(f"Level {i} - {model_name}")
                if not prev_models:
                    # Level 0
                    if isinstance(base_model, L0BaggingModelRegressor): 
                        level_models.append(base_model)
                    else:
                        model = L0BaggingModelRegressor(base_model=base_model,
                                               score_func=self.score_func,
                                               n_splits=self.n_splits,
                                               random_state=self.random_state,
                                               scaler=self.scaler,
                                               feature_info=self.feature_info,
                                               feature_mask=self.feature_mask)
                        model.fit(X_train, y_train, X_test, y_test)
                        level_models.append(model)
                else:
                    # Level 1+
                    if isinstance(base_model, L1BaggingModelRegressor): 
                        level_models.append(base_model)
                    else:
                        model = L1BaggingModelRegressor(base_model=base_model,
                                               prev_models=prev_models,
                                               score_func=self.score_func,
                                               feature_skip=feature_skip,
                                               n_splits=self.n_splits,
                                               random_state=self.random_state,
                                               scaler=self.scaler,
                                               feature_info=self.feature_info,
                                               feature_mask=self.feature_mask)
                        model.fit(X_train, y_train, X_test, y_test)
                        level_models.append(model)
            prev_models = level_models
            self.all_models.append(level_models)

        if self.meta_model:
            if self.is_bagged:
                self.ensembler = MetaModelEnsemblerBaggedRegressor(prev_models, self.meta_model, self.score_func)
            else:
                self.ensembler = MetaModelEnsemblerRegressor(prev_models, self.meta_model, self.score_func)
        else:
            self.ensembler = WeightedEnsemblerRegressor(prev_models, self.score_func, self.maximize_score)
        self.ensembler.fit(y_train, y_test)

    def predict(self, X):
        if self.ensembler is None:
            raise ValueError("Call `.fit()` first to train the ensemble model.")
        return self.ensembler.predict(X)

    def score(self, X, y):
        if self.ensembler is None:
            raise ValueError("Call `.fit()` first to train the ensemble model.")
        return self.ensembler.score(X, y)

    def visualize_tree(self, calculate_shap=True, max_models_per_level=None,
                       labels=None, rankdir='TB', ranksep='0.5', dpi ='300', format='png'):
        if self.ensembler is None:
            raise ValueError("Call `.fit()` first to train the ensemble model.")
        
        dot = graphviz.Digraph(format=format, engine='dot')
        dot.attr(rankdir=rankdir, ranksep=ranksep, fontname='Times')
        dot.graph_attr['dpi'] = dpi
        
        # first level only
        if labels is not None:
            with dot.subgraph() as s:
                s.attr(rank='same')
                num_models = len(self.all_models[0])
                if max_models_per_level and num_models > max_models_per_level + 1:
                    half = max_models_per_level // 2
                    visible_indices = list(range(half)) + list(range(num_models - (max_models_per_level - half), num_models))
                else:
                    visible_indices = list(range(num_models))

                for i in visible_indices:
                    if i < len(labels):
                        s.node(f'label_{i}', str(labels[i]), shape='plaintext', fontsize='14')

        for level_idx, models in enumerate(self.all_models):
            num_models = len(models)
            with dot.subgraph() as s:
                s.attr(rank='same')
    
                # Truncate model list if needed
                if max_models_per_level and num_models > (max_models_per_level + 1):
                    half = max_models_per_level // 2
                    first_part = models[:half]
                    second_part = models[-(max_models_per_level - half):]
                    is_truncated = True
                else:
                    first_part = models
                    second_part = []
                    is_truncated = False
    
                offset = 0
    
                # Add first part nodes
                for i, model in enumerate(tqdm(first_part, desc=f"Level {level_idx} Models", leave=True)):
                    model_name = get_name(model.base_model)
                    node_id = f'{level_idx}_{i}'
                    s.node(node_id, f'{model_name}\nVal: {model.oof_score:.3f}\nTest: {model.test_score:.3f}',
                           shape='ellipse', style='filled', fillcolor='lightgrey')
    
                    if level_idx > 0:
                        prev_models = self.all_models[level_idx - 1]
                        prev_display = prev_models
                        if max_models_per_level and len(prev_models) > max_models_per_level:
                            half_prev = max_models_per_level // 2
                            prev_display = prev_models[:half_prev] + prev_models[-(max_models_per_level - half_prev):]
    
                        if calculate_shap:
                            feature_importance = np.array(model.calculate_meta_feature_importance()[:len(prev_display)])
                            color_importance = feature_importance / (np.max(feature_importance) + 1e-8)
                            edge_colors = [to_grayscale_hex(val) for val in color_importance]
                        else:
                            edge_colors = [to_grayscale_hex(1.0) for _ in range(len(prev_display))]
    
                        for h in range(len(prev_display)):
                            dot.edge(f'{level_idx - 1}_{h}', node_id, color=edge_colors[h])

                    elif labels is not None: # First level
                        if i < len(labels) and i in visible_indices:
                            label_node = f'label_{i}'
                            dot.edge(label_node, node_id, style='invis')

                # Chain first_part nodes to preserve left-to-right order
                for i in range(len(first_part) - 1):
                    dot.edge(f'{level_idx}_{i}', f'{level_idx}_{i+1}', style='invis')

                offset += len(first_part)
    
                # Add second part nodes
                for i, model in enumerate(tqdm(second_part, desc=f"Level {level_idx} Models", leave=True)):
                    node_id = f'{level_idx}_{offset + i}'
                    model_name = get_name(model.base_model)
                    s.node(node_id, f'{model_name}\nVal: {model.oof_score:.3f}\nTest: {model.test_score:.3f}',
                           shape='ellipse', style='filled', fillcolor='lightgrey')
    
                    if level_idx > 0:
                        prev_models = self.all_models[level_idx - 1]
                        prev_display = prev_models
                        if max_models_per_level and len(prev_models) > max_models_per_level:
                            half_prev = max_models_per_level // 2
                            prev_display = prev_models[:half_prev] + prev_models[-(max_models_per_level - half_prev):]
    
                        if calculate_shap:
                            feature_importance = np.array(model.calculate_meta_feature_importance()[:len(prev_display)])
                            color_importance = feature_importance / (np.max(feature_importance) + 1e-8)
                            edge_colors = [to_grayscale_hex(val) for val in color_importance]
                        else:
                            edge_colors = [to_grayscale_hex(1.0) for _ in range(len(prev_display))]
    
                        for h in range(len(prev_display)):
                            dot.edge(f'{level_idx - 1}_{h}', node_id, color=edge_colors[h])
                        
                    elif labels is not None: # First level
                        if max_models_per_level is not None:
                            half = max_models_per_level // 2
                            id = num_models - (max_models_per_level - half) + i
                        else:
                            id = offset + i
                        if id < len(labels) and id in visible_indices:
                            label_node = f'label_{id}'
                            dot.edge(label_node, node_id, style='invis')
                
                # Chain second_part nodes to preserve left-to-right order
                for i in range(len(second_part) - 1):
                    dot.edge(f'{level_idx}_{offset + i}', f'{level_idx}_{offset + i + 1}', style='invis')

                # Add ellipsis node
                if is_truncated and max_models_per_level is not None:
                    hidden_count = num_models - max_models_per_level
                    ellipsis_id = f'{level_idx}_ellipsis'
                    s.node(ellipsis_id, f'(+{hidden_count})', shape='plaintext', fontsize='20')
    
                    # Invisible edges to "center" the ellipsis
                    if len(first_part) > 0:
                        dot.edge(f'{level_idx}_{len(first_part) - 1}', ellipsis_id, style='invis')
                    if len(second_part) > 0:
                        dot.edge(ellipsis_id, f'{level_idx}_{offset}', style='invis')

                    # Visible dashed edges for clarity
                    if len(first_part) > 0:
                        dot.edge(f'{level_idx}_{len(first_part) - 1}', ellipsis_id,
                                style='dashed', arrowhead='none')
                    if len(second_part) > 0:
                        dot.edge(ellipsis_id, f'{level_idx}_{offset}',
                                style='dashed', arrowhead='none')
    
        # Add final ensemble node
        val_score = self.ensembler.oof_score
        test_score = self.ensembler.test_score
        dot.node('ensemble', f'Ensemble\nVal: {val_score:.3f}\nTest: {test_score:.3f}',
                 shape='box', style='filled', color='lightblue')
    
        # Connect last level to ensemble
        last_level_idx = len(self.all_models) - 1
        last_models = self.all_models[last_level_idx]
        if max_models_per_level and len(last_models) > max_models_per_level:
            half_last = max_models_per_level // 2
            display_last = last_models[:half_last] + last_models[-(max_models_per_level - half_last):]
        else:
            display_last = last_models
    
        weights = np.array(self.ensembler.weights[:len(display_last)])
        max_weight = np.max(np.abs(weights)) + 1e-8
        for i, weight in enumerate(weights):
            node_id = f'{last_level_idx}_{i}'
            edge_color = to_grayscale_hex(abs(weight) / max_weight)
            dot.edge(node_id, 'ensemble', label=f'{weight:.2f}', color=edge_color)
        return dot