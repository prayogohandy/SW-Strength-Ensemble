import pandas as pd

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Hw/Lw'] = df['Hw'] / df['Lw']
    df['Lw/tw'] = df['Lw'] / df['tw']
    df['rhofyh'] = df['fyh'] * df['rho_h']
    df['rhofyv'] = df['fyv'] * df['rho_v']
    df['rhofyb'] = df['fyb'] * df['rho_b']
    df['Ab'] = df['bf'] * df['tf'] * 2
    df['Ag'] = (df['Lw'] - 2 * df['tf']) * df['tw'] + df['Ab']
    df['Pc'] = df['Ag'] * df["fc'"] / 1000
    df['ALR'] = df['P']/df['Pc']
    return df

def step_to_format(step):
    if step == 0: return "%.0f"
    step_str = f"{step:.10f}".rstrip("0")
    if "." in step_str:
        decimals = len(step_str.split(".")[1])
        if decimals == 0: return "%d"
        return f"%.{decimals}f"
    return "%d"
