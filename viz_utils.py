import altair as alt
import pandas as pd

def plot_missing_bar(df):
    missing_pct = df.isnull().mean() * 100
    data = pd.DataFrame({'column': missing_pct.index, 'missing_pct': missing_pct.values})
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('column:N', title='Column'),
        y=alt.Y('missing_pct:Q', title='% Missing'),
        color=alt.Color('missing_pct:Q', scale=alt.Scale(scheme='reds'))
    ).properties(title='Missing Value Percentage by Column')
    return chart

def plot_histogram(df, col):
    data = df[[col]].dropna()
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=30), title=col),
        y=alt.Y('count()', title='Count')
    ).properties(title=f"Histogram: {col}")
    return chart

def plot_bar_top_values(s, col):
    top_vals = s.value_counts(dropna=False).head(10).reset_index()
    top_vals.columns = ['value', 'count']
    chart = alt.Chart(top_vals).mark_bar().encode(
        x=alt.X('value:N', title=col),
        y=alt.Y('count:Q', title='Count')
    ).properties(title=f"Top 10 Values: {col}")
    return chart

def plot_missing_heatmap(df):
    missing_matrix = df.isnull().astype(int)
    data = missing_matrix.reset_index().melt(id_vars='index', var_name='column', value_name='missing')
    chart = alt.Chart(data).mark_rect().encode(
        x=alt.X('index:O', title='Row'),
        y=alt.Y('column:N', title='Column'),
        color=alt.Color('missing:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title='Missing'))
    ).properties(title='Missingness Heatmap (yellow = missing)')
    return chart

def plot_missing_timeline(s: pd.Series):
    present = (~s.isna()).astype(int)
    df_runs = pd.DataFrame({'index': s.index, 'present': present})
    chart = alt.Chart(df_runs).mark_line().encode(
        x=alt.X('index:O', title='Index'),
        y=alt.Y('present:Q', title='Present (1) / Missing (0)', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(values=[0,1], labelExpr="datum.value == 1 ? 'Present' : 'Missing'"))
    ).properties(title=f"Missing Timeline: {s.name if s.name else 'Column'}")
    return chart

def plot_value_counts(s: pd.Series, top_n=10):
    vc = s.value_counts().head(top_n).reset_index()
    vc.columns = ['value', 'count']
    chart = alt.Chart(vc).mark_bar().encode(
        x=alt.X('value:N', title=s.name if s.name else 'Value'),
        y=alt.Y('count:Q', title='Count')
    ).properties(title=f"Value Counts: {s.name if s.name else 'Value'}")
    return chart

def plot_boxplot(df, col):
    data = df[[col]].dropna()
    chart = alt.Chart(data).mark_boxplot(
        outliers=True,
        color='#1f77b4',
        opacity=0.8
    ).encode(
        y=alt.Y(f'{col}:Q', title=col),
        color=alt.value('#1f77b4')
    ).properties(title=f"Boxplot: {col}")
    return chart