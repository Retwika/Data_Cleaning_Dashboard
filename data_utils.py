import pandas as pd
import numpy as np
import re

def drop_high_missing_rows(df, threshold=0.5):
    thresh = int(df.shape[1] * threshold)
    return df.dropna(thresh=thresh+1)

def count_outliers(df):
    outliers_count = 0
    for col in df.select_dtypes(include=[float, int]).columns:
        s = df[col].dropna()
        if len(s) > 0:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers_count += ((s < lower) | (s > upper)).sum()
    return outliers_count

def get_missing_cols(df):
    return list(df.columns[df.isnull().any()])

def summarize_column(df, col):
    s = df[col]
    def to_native(val):
        if hasattr(val, 'item'):
            return val.item()
        return val
    summary = {
        'dtype': str(s.dtype),
        'missing_pct': s.isnull().mean() * 100,
        'n_missing': s.isnull().sum(),
        'n_unique': s.nunique(dropna=True),
        'sample_values': [to_native(v) for v in s.dropna().unique()[:5]],
        'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
        'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
        'mode': s.mode().iloc[0] if not s.mode().empty else None,
        'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None,
        'median': s.median() if pd.api.types.is_numeric_dtype(s) else None
    }
    return summary

def detect_missing_runs(s: pd.Series) -> dict:
    is_na = s.isna().astype(int)
    changes = is_na.diff().fillna(0)!=0
    groups = changes.cumsum()
    runs = s.groupby(groups).apply(lambda grp: len(grp) if grp.isna().all() else 0)
    return {
        "longest_gap": int(runs.max()),
        "isolated_count": int((runs==1).sum())
    }

def detect_case_issues(s: pd.Series) -> float:
    mask = s.dropna().str.match(r'.*[A-Z].*') & s.dropna().str.match(r'.*[a-z].*')
    return mask.sum() / len(s.dropna())

def normalize_case(s: pd.Series, mode: str = 'lower') -> pd.Series:
    return getattr(s.str, mode)()

def parse_money_series(s: pd.Series) -> pd.Series:
    mult = {'K':1e3,'M':1e6,'B':1e9, "T": 1e12}
    def parse_val(val):
        if pd.isnull(val):
            return np.nan
        m = re.search(r'([-+]?[0-9]*\.?[0-9]+)\s*([KMBT])?', str(val))
        if m:
            num = float(m.group(1))
            factor = mult.get(m.group(2), 1)
            return num * factor
        return np.nan
    return s.apply(parse_val) 